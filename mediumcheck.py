"""
Search for mystery tokens in actual kern data files.
"""

from pathlib import Path
import re

def search_in_kern_files(data_dir, search_strings):
    """Search for specific strings in all kern files."""
    
    kern_files = list(Path(data_dir).rglob("*.krn")) + list(Path(data_dir).rglob("*.txt"))
    
    print(f"\nSearching in: {data_dir}")
    print(f"Found {len(kern_files)} potential kern files")
    
    results = {s: [] for s in search_strings}
    
    for kern_file in kern_files:
        try:
            with open(kern_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                for search_str in search_strings:
                    if search_str in content:
                        # Find all occurrences with context
                        lines = content.split('\n')
                        for line_num, line in enumerate(lines):
                            if search_str in line:
                                results[search_str].append({
                                    'file': str(kern_file),
                                    'line_num': line_num,
                                    'line': line.strip(),
                                })
        except Exception as e:
            pass  # Skip files that can't be read
    
    return results


def main():
    print("="*80)
    print("SEARCHING FOR MYSTERY TOKENS IN KERN FILES")
    print("="*80)
    
    mystery_strings = ['efG', ' l', '\tl', 'l\t', 'l\n', '*G-:']
    
    # Search in both datasets
    for data_dir in [
        'data/jazzmus_systems',
        'data/jazzmus_fullpage',
    ]:
        if not Path(data_dir).exists():
            print(f"\n⚠️  Directory not found: {data_dir}")
            continue
        
        results = search_in_kern_files(data_dir, mystery_strings)
        
        print(f"\n{'='*80}")
        print(f"RESULTS FOR: {data_dir}")
        print(f"{'='*80}")
        
        for search_str in mystery_strings:
            if results[search_str]:
                print(f"\n--- Found '{search_str}' in {len(results[search_str])} places ---")
                
                # Show first 5 occurrences
                for i, occurrence in enumerate(results[search_str][:5]):
                    print(f"\n  Occurrence {i+1}:")
                    print(f"    File: {occurrence['file']}")
                    print(f"    Line {occurrence['line_num']}: {occurrence['line']}")
                
                if len(results[search_str]) > 5:
                    print(f"\n  ... and {len(results[search_str]) - 5} more occurrences")
            else:
                print(f"\n--- '{search_str}' NOT FOUND ---")


if __name__ == "__main__":
    main()