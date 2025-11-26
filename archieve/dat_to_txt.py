#!/usr/bin/env python3
# extract_dat_to_txt.py
import argparse
import json
import os
import pickle
from pathlib import Path

def try_torch_load(p):
    try:
        import torch
        return torch.load(p, map_location="cpu")
    except Exception:
        return None

def try_numpy_load(p):
    try:
        import numpy as np
        return np.load(p, allow_pickle=True)
    except Exception:
        return None

def try_pickle_load(p):
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def try_json_load(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def obj_to_iterable(obj):
    """Return a flat iterable of strings from arbitrary Python objects."""
    lines = []

    def flat(x):
        # Accept strings directly
        if isinstance(x, (str, bytes, os.PathLike)):
            s = x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else str(x)
            s = s.strip()
            if s:
                lines.append(s)
            return

        # NumPy arrays
        try:
            import numpy as np
            if isinstance(x, np.ndarray):
                # If array of strings/objects, iterate; else convert scalars
                if x.dtype.kind in ("U", "S", "O"):
                    for y in x.ravel():
                        flat(y)
                else:
                    for y in x.ravel():
                        flat(str(y))
                return
        except Exception:
            pass

        # Torch tensors -> tolist
        try:
            import torch
            if isinstance(x, torch.Tensor):
                flat(x.detach().cpu().numpy())
                return
        except Exception:
            pass

        # Lists/tuples/sets
        if isinstance(x, (list, tuple, set)):
            for y in x:
                flat(y)
            return

        # Dicts -> values (keys often metadata)
        if isinstance(x, dict):
            # If dict looks like {"items": [...]}, prefer that
            candidates = None
            for k in ("items", "paths", "files", "ids", "samples", "data"):
                if k in x and isinstance(x[k], (list, tuple)):
                    candidates = x[k]
                    break
            if candidates is not None:
                flat(candidates)
            else:
                for v in x.values():
                    flat(v)
            return

        # Fallback: stringify scalars / unknowns
        try:
            s = str(x).strip()
            if s:
                lines.append(s)
        except Exception:
            pass

    flat(obj)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for s in lines:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def load_any(dat_path):
    """Try multiple loaders; return (obj, loader_name) or (None, None)."""
    for loader, name in (
        (try_torch_load, "torch"),
        (try_numpy_load, "numpy"),
        (try_pickle_load, "pickle"),
        (try_json_load, "json"),
    ):
        obj = loader(dat_path)
        if obj is not None:
            return obj, name
    return None, None

def main():
    ap = argparse.ArgumentParser(description="Extract text lines from .dat files into .txt")
    ap.add_argument("--root", required=True, help="Root directory to search for .dat files")
    ap.add_argument("--pattern", default="*.dat", help="Glob pattern for dat files (default: *.dat)")
    ap.add_argument("--key", default=None, help="If loaded object is a dict, extract this key only")
    ap.add_argument("--aggregate-out", default=None, help="Write all extracted lines to a single TXT file")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be written without writing files")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    dat_files = sorted(root.rglob(args.pattern))

    if not dat_files:
        print(f"[!] No files matched {args.pattern} under {root}")
        return

    aggregate = []

    for dat_path in dat_files:
        obj, loader = load_any(dat_path)
        if obj is None:
            print(f"[x] Could not load: {dat_path}")
            continue

        # Optional key extraction for dicts
        if args.key and isinstance(obj, dict):
            if args.key in obj:
                obj = obj[args.key]
            else:
                print(f"[!] Key '{args.key}' not in dict for {dat_path}; continuing with all values.")

        lines = obj_to_iterable(obj)
        print(f"[{loader:6}] {dat_path} -> {len(lines)} lines")

        if args.aggregate_out:
            aggregate.extend(lines)
        else:
            out_path = dat_path.with_suffix(".txt")
            if args.dry_run:
                print(f"     (dry-run) would write: {out_path}")
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    for ln in lines:
                        f.write(f"{ln}\n")

    if args.aggregate_out:
        # Dedup aggregate while preserving order
        seen = set()
        uniq = []
        for s in aggregate:
            if s not in seen:
                seen.add(s)
                uniq.append(s)

        out_path = Path(args.aggregate_out).resolve()
        if args.dry_run:
            print(f"(dry-run) would write aggregate {out_path} with {len(uniq)} lines")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for ln in uniq:
                    f.write(f"{ln}\n")
            print(f"[done] aggregate -> {out_path} ({len(uniq)} lines)")

if __name__ == "__main__":
    main()

