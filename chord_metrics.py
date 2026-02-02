"""
Chord-Specific Evaluation Metrics for Jazz Lead Sheet Recognition

Addresses the problem that CER/SER treat all errors equally:
- C7 → Cmaj7 (wrong extension) penalized same as C7 → F#7 (wrong root)
- But musically these are very different errors!

New Metrics:
1. Root Detection F1 - treats root detection as precision/recall problem
2. Quality Accuracy - major, minor, diminished, augmented (once roots aligned)
3. Extension Accuracy - 7, maj7, min7, 9, 13, etc. (once roots aligned)
"""

import re
from typing import List, Tuple, Dict, Optional, NamedTuple
from dataclasses import dataclass
from collections import Counter


@dataclass
class ParsedChord:
    """Parsed chord components."""
    original: str
    root: Optional[str]  # e.g., "C", "F#", "Bb"
    quality: Optional[str]  # e.g., "maj", "min", "dim", "aug", "" (for dominant)
    extension: Optional[str]  # e.g., "7", "maj7", "9", "13"
    modifiers: List[str]  # e.g., ["b9", "#11"]
    bass: Optional[str]  # e.g., "G" for slash chord C/G
    is_valid: bool  # Whether parsing succeeded


def parse_chord(chord_str: str) -> ParsedChord:
    """
    Parse a chord string into its components.

    Format examples from **mxhm:
    - C:maj7 -> root=C, quality=maj, extension=7
    - G:7 -> root=G, quality="" (dominant), extension=7
    - D-:min7 -> root=Db, quality=min, extension=7
    - A:min7(b5) -> root=A, quality=min, extension=7, modifiers=[b5]
    - F:maj7(9,13) -> root=F, quality=maj, extension=7, modifiers=[9, 13]
    - C:7/G -> root=C, quality="", extension=7, bass=G
    - Bb:dim7 -> root=Bb, quality=dim, extension=7

    Args:
        chord_str: Raw chord string

    Returns:
        ParsedChord with extracted components
    """
    chord_str = chord_str.strip()

    # Handle empty or placeholder chords
    if not chord_str or chord_str in ['.', '*', 'N.C.', 'NC', 'N.C', 'rest']:
        return ParsedChord(
            original=chord_str,
            root=None, quality=None, extension=None,
            modifiers=[], bass=None, is_valid=False
        )

    # Extract bass note if present (slash chord)
    bass = None
    if '/' in chord_str:
        parts = chord_str.split('/')
        chord_str = parts[0]
        bass = normalize_pitch(parts[1]) if len(parts) > 1 else None

    # Extract root and type
    if ':' in chord_str:
        root_part, type_part = chord_str.split(':', 1)
    else:
        # No colon - might just be a root (e.g., "C" implies major triad)
        root_part = chord_str
        type_part = ""

    # Parse root with accidentals
    root = normalize_pitch(root_part)

    # Parse modifiers in parentheses
    modifiers = []
    if '(' in type_part:
        mod_match = re.search(r'\(([^)]+)\)', type_part)
        if mod_match:
            mod_str = mod_match.group(1)
            modifiers = [m.strip() for m in mod_str.split(',')]
            type_part = type_part[:type_part.index('(')]

    # Parse quality and extension from type
    quality, extension = parse_chord_type(type_part)

    return ParsedChord(
        original=chord_str,
        root=root,
        quality=quality,
        extension=extension,
        modifiers=modifiers,
        bass=bass,
        is_valid=root is not None
    )


def normalize_pitch(pitch_str: str) -> Optional[str]:
    """
    Normalize pitch to standard format.

    Handles:
    - C, D, E, F, G, A, B (uppercase)
    - # and - for sharps and flats
    - Converts - to b for consistency

    Returns:
        Normalized pitch string (e.g., "C", "F#", "Bb") or None if invalid
    """
    pitch_str = pitch_str.strip()
    if not pitch_str:
        return None

    # Extract letter (first character, uppercase)
    letter = pitch_str[0].upper()
    if letter not in 'ABCDEFG':
        return None

    # Extract accidental
    accidental = ""
    rest = pitch_str[1:]
    if '#' in rest:
        accidental = "#"
    elif '-' in rest or 'b' in rest.lower():
        accidental = "b"

    return letter + accidental


def parse_chord_type(type_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse chord type into quality and extension.

    Examples:
    - "maj7" -> ("maj", "7")
    - "min7" -> ("min", "7")
    - "7" -> ("", "7")  # dominant
    - "dim7" -> ("dim", "7")
    - "aug" -> ("aug", None)
    - "min7(b5)" -> ("min", "7")  # half-diminished
    - "sus4" -> ("sus4", None)
    - "none" -> (None, None)

    Returns:
        Tuple of (quality, extension)
    """
    type_str = type_str.strip().lower()

    if not type_str or type_str == "none":
        return (None, None)

    # Quality patterns (order matters - longer patterns first)
    quality_patterns = [
        ('maj', 'maj'),
        ('min', 'min'),
        ('m', 'min'),  # shorthand
        ('dim', 'dim'),
        ('aug', 'aug'),
        ('sus4', 'sus4'),
        ('sus2', 'sus2'),
        ('hdim', 'hdim'),  # half-diminished
    ]

    quality = ""  # Default: dominant/major (no quality modifier)
    remaining = type_str

    for pattern, qual_name in quality_patterns:
        if type_str.startswith(pattern):
            quality = qual_name
            remaining = type_str[len(pattern):]
            break

    # Extract extension (numbers like 7, 9, 11, 13, 6)
    extension = None
    ext_match = re.search(r'(\d+)', remaining)
    if ext_match:
        extension = ext_match.group(1)

    return (quality, extension)


def extract_chords_from_mxhm(mxhm_content: str) -> List[str]:
    """
    Extract chord symbols from **mxhm spine content.

    Filters out:
    - Dots (.) which are duration/rest markers
    - Barlines (=)
    - Spine terminators (*-)
    - Empty lines

    Args:
        mxhm_content: Content from **mxhm spine

    Returns:
        List of chord symbol strings
    """
    chords = []
    for line in mxhm_content.split('\n'):
        line = line.strip()
        # Skip non-chord lines
        if not line or line == '.' or line.startswith('=') or line.startswith('*'):
            continue
        # Skip continuation dots within a line
        if line == '.':
            continue
        chords.append(line)
    return chords


def compute_root_f1(pred_chords: List[str], gt_chords: List[str]) -> Dict[str, float]:
    """
    Compute Root Detection F1 score using multiple alignment strategies.

    Three strategies:
    1. Position-based: Compare same index (strict, breaks on insertions)
    2. Bag-of-roots: Multiset matching (ignores order completely)
    3. Aligned (LCS): Edit-distance alignment (tolerates insertions/deletions)

    Args:
        pred_chords: List of predicted chord strings
        gt_chords: List of ground truth chord strings

    Returns:
        Dict with metrics for each strategy
    """
    pred_parsed = [parse_chord(c) for c in pred_chords]
    gt_parsed = [parse_chord(c) for c in gt_chords]

    # Extract valid roots
    pred_roots = [p.root for p in pred_parsed if p.root is not None]
    gt_roots = [g.root for g in gt_parsed if g.root is not None]

    # === Strategy 1: Position-based (strict) ===
    min_len = min(len(pred_parsed), len(gt_parsed))
    pos_correct = 0
    for i in range(min_len):
        if (pred_parsed[i].root is not None and
            gt_parsed[i].root is not None and
            pred_parsed[i].root == gt_parsed[i].root):
            pos_correct += 1

    pos_precision = pos_correct / len(pred_roots) * 100 if pred_roots else 0.0
    pos_recall = pos_correct / len(gt_roots) * 100 if gt_roots else 0.0
    pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall) if (pos_precision + pos_recall) > 0 else 0.0

    # === Strategy 2: Bag-of-roots (multiset intersection) ===
    pred_root_counts = Counter(pred_roots)
    gt_root_counts = Counter(gt_roots)
    # Intersection: min count for each root
    bag_correct = sum((pred_root_counts & gt_root_counts).values())

    bag_precision = bag_correct / len(pred_roots) * 100 if pred_roots else 0.0
    bag_recall = bag_correct / len(gt_roots) * 100 if gt_roots else 0.0
    bag_f1 = 2 * bag_precision * bag_recall / (bag_precision + bag_recall) if (bag_precision + bag_recall) > 0 else 0.0

    # === Strategy 3: Aligned via LCS (Longest Common Subsequence) ===
    align_correct = _lcs_count(pred_roots, gt_roots)

    align_precision = align_correct / len(pred_roots) * 100 if pred_roots else 0.0
    align_recall = align_correct / len(gt_roots) * 100 if gt_roots else 0.0
    align_f1 = 2 * align_precision * align_recall / (align_precision + align_recall) if (align_precision + align_recall) > 0 else 0.0

    # === Strategy 4: Windowed/Fuzzy Position (±window tolerance) ===
    window = 3  # Look ±3 positions for a match
    window_correct, window_used_pred = _windowed_match(pred_roots, gt_roots, window)

    window_precision = window_correct / len(pred_roots) * 100 if pred_roots else 0.0
    window_recall = window_correct / len(gt_roots) * 100 if gt_roots else 0.0
    window_f1 = 2 * window_precision * window_recall / (window_precision + window_recall) if (window_precision + window_recall) > 0 else 0.0

    return {
        # Position-based (strict)
        'precision': pos_precision,
        'recall': pos_recall,
        'f1': pos_f1,
        'correct': pos_correct,
        # Bag-of-roots (order-independent)
        'bag_precision': bag_precision,
        'bag_recall': bag_recall,
        'bag_f1': bag_f1,
        'bag_correct': bag_correct,
        # Aligned via LCS
        'align_precision': align_precision,
        'align_recall': align_recall,
        'align_f1': align_f1,
        'align_correct': align_correct,
        # Windowed/Fuzzy Position (±3 tolerance) - RECOMMENDED
        'window_precision': window_precision,
        'window_recall': window_recall,
        'window_f1': window_f1,
        'window_correct': window_correct,
        # Counts
        'pred_count': len(pred_roots),
        'gt_count': len(gt_roots),
        'count_diff': len(pred_chords) - len(gt_chords),
    }


def _windowed_match(pred_roots: List[str], gt_roots: List[str], window: int = 3) -> Tuple[int, set]:
    """
    Match GT roots to pred roots with a position tolerance window.

    For each GT root at position i, look for a matching pred root
    at positions [i-window, i+window]. Each pred root can only be
    matched once (greedy, closest position first).

    Args:
        pred_roots: List of predicted root notes
        gt_roots: List of ground truth root notes
        window: Position tolerance (default ±3)

    Returns:
        Tuple of (number of matches, set of used pred indices)
    """
    if not pred_roots or not gt_roots:
        return 0, set()

    used_pred_indices = set()
    matches = 0

    for gt_idx, gt_root in enumerate(gt_roots):
        # Search window around gt position, closest first
        search_order = []
        for offset in range(window + 1):
            if offset == 0:
                search_order.append(gt_idx)
            else:
                search_order.extend([gt_idx - offset, gt_idx + offset])

        # Find first matching pred root in window that hasn't been used
        for pred_idx in search_order:
            if pred_idx < 0 or pred_idx >= len(pred_roots):
                continue
            if pred_idx in used_pred_indices:
                continue
            if pred_roots[pred_idx] == gt_root:
                matches += 1
                used_pred_indices.add(pred_idx)
                break

    return matches, used_pred_indices


def _lcs_count(seq1: List[str], seq2: List[str]) -> int:
    """
    Compute length of Longest Common Subsequence (LCS).

    This finds the maximum number of elements that match in order,
    allowing gaps (insertions/deletions).

    Example:
        seq1 = [C, G, G, Am, F]  (pred with extra G)
        seq2 = [C, G, Am, F]     (gt)
        LCS  = [C, G, Am, F]     (length 4)

    Returns:
        Number of matched elements in optimal alignment
    """
    if not seq1 or not seq2:
        return 0

    m, n = len(seq1), len(seq2)
    # dp[i][j] = LCS length for seq1[:i] and seq2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def compute_quality_accuracy(pred_chords: List[str], gt_chords: List[str]) -> Dict[str, float]:
    """
    Compute chord quality accuracy (once roots are aligned).

    Quality categories: major, minor, diminished, augmented, sus, etc.
    Only evaluates positions where both roots match.

    Args:
        pred_chords: List of predicted chord strings
        gt_chords: List of ground truth chord strings

    Returns:
        Dict with accuracy and breakdown by quality type
    """
    pred_parsed = [parse_chord(c) for c in pred_chords]
    gt_parsed = [parse_chord(c) for c in gt_chords]

    # Only evaluate where roots match
    min_len = min(len(pred_parsed), len(gt_parsed))

    total_root_matches = 0
    quality_correct = 0
    quality_breakdown = Counter()
    quality_errors = Counter()  # (gt_quality, pred_quality) pairs

    for i in range(min_len):
        pred, gt = pred_parsed[i], gt_parsed[i]

        # Only evaluate where roots match
        if (pred.root is not None and gt.root is not None and
            pred.root == gt.root):
            total_root_matches += 1

            # Compare quality (normalize empty string to "dom" for dominant)
            pred_qual = pred.quality if pred.quality else "dom"
            gt_qual = gt.quality if gt.quality else "dom"

            if pred_qual == gt_qual:
                quality_correct += 1
                quality_breakdown[gt_qual] += 1
            else:
                quality_errors[(gt_qual, pred_qual)] += 1

    accuracy = quality_correct / total_root_matches * 100 if total_root_matches > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct': quality_correct,
        'total_root_matches': total_root_matches,
        'breakdown': dict(quality_breakdown),
        'top_errors': quality_errors.most_common(5),
    }


def compute_extension_accuracy(pred_chords: List[str], gt_chords: List[str]) -> Dict[str, float]:
    """
    Compute chord extension accuracy (once roots are aligned).

    Extensions: 7, maj7, 6, 9, 11, 13, etc.
    Only evaluates positions where both roots match.

    Args:
        pred_chords: List of predicted chord strings
        gt_chords: List of ground truth chord strings

    Returns:
        Dict with accuracy and breakdown by extension type
    """
    pred_parsed = [parse_chord(c) for c in pred_chords]
    gt_parsed = [parse_chord(c) for c in gt_chords]

    min_len = min(len(pred_parsed), len(gt_parsed))

    total_root_matches = 0
    extension_correct = 0
    extension_breakdown = Counter()
    extension_errors = Counter()

    for i in range(min_len):
        pred, gt = pred_parsed[i], gt_parsed[i]

        if (pred.root is not None and gt.root is not None and
            pred.root == gt.root):
            total_root_matches += 1

            # Compare extension
            pred_ext = pred.extension if pred.extension else "none"
            gt_ext = gt.extension if gt.extension else "none"

            if pred_ext == gt_ext:
                extension_correct += 1
                extension_breakdown[gt_ext] += 1
            else:
                extension_errors[(gt_ext, pred_ext)] += 1

    accuracy = extension_correct / total_root_matches * 100 if total_root_matches > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct': extension_correct,
        'total_root_matches': total_root_matches,
        'breakdown': dict(extension_breakdown),
        'top_errors': extension_errors.most_common(5),
    }


def compute_full_chord_accuracy(pred_chords: List[str], gt_chords: List[str]) -> Dict[str, float]:
    """
    Compute full chord match metrics (root + quality + extension all correct).

    Position-based comparison with precision/recall/F1 to handle count differences.
    """
    pred_parsed = [parse_chord(c) for c in pred_chords]
    gt_parsed = [parse_chord(c) for c in gt_chords]

    min_len = min(len(pred_parsed), len(gt_parsed))

    full_correct = 0
    total_valid_gt = 0
    total_valid_pred = 0

    # Count valid chords in each
    for p in pred_parsed:
        if p.root is not None:
            total_valid_pred += 1
    for g in gt_parsed:
        if g.root is not None:
            total_valid_gt += 1

    # Position-based matching
    for i in range(min_len):
        pred, gt = pred_parsed[i], gt_parsed[i]

        if gt.root is not None and pred.root is not None:
            if (pred.root == gt.root and
                pred.quality == gt.quality and
                pred.extension == gt.extension):
                full_correct += 1

    # Compute P/R/F1
    precision = full_correct / total_valid_pred * 100 if total_valid_pred > 0 else 0.0
    recall = full_correct / total_valid_gt * 100 if total_valid_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correct': full_correct,
        'pred_count': total_valid_pred,
        'gt_count': total_valid_gt,
    }


def analyze_alignment(pred_chords: List[str], gt_chords: List[str]) -> Dict:
    """
    Analyze alignment between predicted and GT chords.

    Helps determine if count mismatches are a significant issue
    and what alignment strategy to use.

    Returns:
        Analysis including counts, mismatch details, and recommendations
    """
    pred_count = len(pred_chords)
    gt_count = len(gt_chords)

    count_diff = pred_count - gt_count

    # Parse both
    pred_parsed = [parse_chord(c) for c in pred_chords]
    gt_parsed = [parse_chord(c) for c in gt_chords]

    # Get root sequences for visualization
    pred_roots = [p.root if p.root else '?' for p in pred_parsed]
    gt_roots = [g.root if g.root else '?' for g in gt_parsed]

    return {
        'pred_count': pred_count,
        'gt_count': gt_count,
        'count_diff': count_diff,
        'counts_match': pred_count == gt_count,
        'pred_roots': pred_roots[:20],  # First 20 for display
        'gt_roots': gt_roots[:20],
    }


def compute_all_chord_metrics(pred_mxhm: str, gt_mxhm: str) -> Dict:
    """
    Compute all chord metrics from **mxhm spine content.

    This is the main entry point for chord evaluation.

    Args:
        pred_mxhm: Predicted **mxhm spine content
        gt_mxhm: Ground truth **mxhm spine content

    Returns:
        Dict with all chord metrics
    """
    pred_chords = extract_chords_from_mxhm(pred_mxhm)
    gt_chords = extract_chords_from_mxhm(gt_mxhm)

    return {
        'alignment': analyze_alignment(pred_chords, gt_chords),
        'root_f1': compute_root_f1(pred_chords, gt_chords),
        'quality': compute_quality_accuracy(pred_chords, gt_chords),
        'extension': compute_extension_accuracy(pred_chords, gt_chords),
        'full_chord': compute_full_chord_accuracy(pred_chords, gt_chords),
        'pred_chord_count': len(pred_chords),
        'gt_chord_count': len(gt_chords),
    }


def print_chord_metrics(metrics: Dict, verbose: bool = True):
    """Pretty print chord metrics."""
    print("\n" + "=" * 60)
    print("CHORD-SPECIFIC METRICS")
    print("=" * 60)

    # Alignment
    align = metrics['alignment']
    print(f"\nAlignment Analysis:")
    print(f"  Predicted chords: {align['pred_count']}")
    print(f"  GT chords: {align['gt_count']}")
    print(f"  Difference: {align['count_diff']:+d}")
    print(f"  Counts match: {'Yes' if align['counts_match'] else 'No'}")

    # Root F1 - all four strategies
    root = metrics['root_f1']
    print(f"\nRoot Detection (4 strategies):")
    print(f"  ┌───────────────────┬───────────┬──────────┬──────────┐")
    print(f"  │ Strategy          │ Precision │  Recall  │    F1    │")
    print(f"  ├───────────────────┼───────────┼──────────┼──────────┤")
    print(f"  │ Position-based    │  {root['precision']:6.2f}%  │  {root['recall']:6.2f}% │  {root['f1']:6.2f}% │")
    print(f"  │ Bag-of-roots      │  {root['bag_precision']:6.2f}%  │  {root['bag_recall']:6.2f}% │  {root['bag_f1']:6.2f}% │")
    print(f"  │ Aligned (LCS)     │  {root['align_precision']:6.2f}%  │  {root['align_recall']:6.2f}% │  {root['align_f1']:6.2f}% │")
    print(f"  │ Windowed (±3) *   │  {root['window_precision']:6.2f}%  │  {root['window_recall']:6.2f}% │  {root['window_f1']:6.2f}% │")
    print(f"  └───────────────────┴───────────┴──────────┴──────────┘")
    print(f"  * Windowed: Tolerates ±3 position shift for line alignment errors")
    print(f"  Counts: {root['pred_count']} predicted, {root['gt_count']} GT")

    # Quality
    qual = metrics['quality']
    print(f"\nQuality Accuracy (where roots match):")
    print(f"  Accuracy: {qual['accuracy']:.2f}%")
    print(f"  Correct: {qual['correct']}/{qual['total_root_matches']}")
    if verbose and qual['top_errors']:
        print(f"  Top errors (GT→Pred):")
        for (gt_q, pred_q), count in qual['top_errors'][:3]:
            print(f"    {gt_q}→{pred_q}: {count}")

    # Extension
    ext = metrics['extension']
    print(f"\nExtension Accuracy (where roots match):")
    print(f"  Accuracy: {ext['accuracy']:.2f}%")
    print(f"  Correct: {ext['correct']}/{ext['total_root_matches']}")
    if verbose and ext['top_errors']:
        print(f"  Top errors (GT→Pred):")
        for (gt_e, pred_e), count in ext['top_errors'][:3]:
            print(f"    {gt_e}→{pred_e}: {count}")

    # Full chord
    full = metrics['full_chord']
    print(f"\nFull Chord Match:")
    print(f"  Precision: {full['precision']:.2f}% ({full['correct']}/{full['pred_count']} predicted)")
    print(f"  Recall:    {full['recall']:.2f}% ({full['correct']}/{full['gt_count']} GT)")
    print(f"  F1:        {full['f1']:.2f}%")

    print("=" * 60)


# Test with some examples
if __name__ == "__main__":
    # Test chord parsing
    test_chords = [
        "C:maj7",
        "G:7",
        "D-:min7",
        "A:min7(b5)",
        "F:maj7(9,13)",
        "C:7/G",
        "Bb:dim7",
        "E:aug",
        "F#:sus4",
        ".",
    ]

    print("Chord Parsing Test:")
    print("-" * 60)
    for chord in test_chords:
        parsed = parse_chord(chord)
        print(f"{chord:20} -> root={parsed.root}, qual={parsed.quality}, "
              f"ext={parsed.extension}, mods={parsed.modifiers}, bass={parsed.bass}")

    # Test metrics with sample data
    print("\n\nMetrics Test 1: Quality/Extension errors (no alignment issues)")
    print("-" * 60)

    gt_sample1 = """C:maj7
G:7
D-:min7
A:min7(b5)
F:maj7"""

    pred_sample1 = """C:maj7
G:maj7
Db:min7
A:min7
F:7"""

    metrics1 = compute_all_chord_metrics(pred_sample1, gt_sample1)
    print_chord_metrics(metrics1)

    # Test 2: Insertion error (shows difference between strategies)
    print("\n\nMetrics Test 2: INSERTION ERROR (extra G inserted)")
    print("-" * 60)
    print("GT:   C    G    Am   F     (4 chords)")
    print("Pred: C    G    G    Am   F  (5 chords - extra G)")
    print("")

    gt_sample2 = """C:maj7
G:7
A:min7
F:maj7"""

    pred_sample2 = """C:maj7
G:7
G:7
A:min7
F:maj7"""

    metrics2 = compute_all_chord_metrics(pred_sample2, gt_sample2)
    print_chord_metrics(metrics2)

    print("\nInterpretation:")
    print("  - Position-based: Cascade failure after insertion")
    print("  - Bag-of-roots: All roots found (extra G counted)")
    print("  - Aligned (LCS): 4/4 GT roots matched in sequence")
    print("  - Windowed (±3): Tolerates position shifts, finds matches nearby")

    # Test 3: Line alignment shift (shows windowed matching advantage)
    print("\n\nMetrics Test 3: LINE ALIGNMENT SHIFT (off by 2 positions)")
    print("-" * 60)
    print("GT:   C    G    Am   F    Dm   E7   (6 chords)")
    print("Pred: X    X    C    G    Am   F    (shifted by 2, different first 2)")
    print("")

    gt_sample3 = """C:maj7
G:7
A:min7
F:maj7
D:min7
E:7"""

    pred_sample3 = """Bb:maj7
Eb:7
C:maj7
G:7
A:min7
F:maj7"""

    metrics3 = compute_all_chord_metrics(pred_sample3, gt_sample3)
    print_chord_metrics(metrics3)

    print("\nInterpretation:")
    print("  - Position-based: Only matches at same position (likely low)")
    print("  - Windowed (±3): Finds C, G, Am, F shifted by 2 positions")
    print("  - This mimics real-world line alignment errors in OCR")
