def compare_vocabularies(checkpoint_path: str, fold: int = 0):
    """
    Compare vocabulary from checkpoint with full-page dataset vocabulary.
    
    Args:
        checkpoint_path: Path to system-level checkpoint
        fold: Dataset fold to use
    """
    import gin
    from collections import Counter
    import torch
    
    print("=" * 80)
    print("VOCABULARY COMPARISON")
    print("=" * 80)
    
    # Load checkpoint vocabulary
    print(f"\n1. Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # The vocabulary should be in the state dict or hyperparameters
    if 'hyper_parameters' in checkpoint and 'w2i' in checkpoint['hyper_parameters']:
        ckpt_w2i = checkpoint['hyper_parameters']['w2i']
        ckpt_i2w = checkpoint['hyper_parameters']['i2w']
    elif 'w2i' in checkpoint:
        ckpt_w2i = checkpoint['w2i']
        ckpt_i2w = checkpoint['i2w']
    else:
        print("Available keys in checkpoint:", checkpoint.keys())
        if 'hyper_parameters' in checkpoint:
            print("Hyper_parameters keys:", checkpoint['hyper_parameters'].keys())
        raise ValueError("Could not find vocabulary in checkpoint!")
    
    print(f"   Checkpoint vocab size: {len(ckpt_w2i)}")
    
    # Load full-page dataset vocabulary
    print(f"\n2. Building full-page dataset vocabulary (fold {fold})...")
    
    # Need to parse gin config first (use same config as training)
    # For now, we'll build it directly
    from jazzmus.dataset.full_page_smt_dataset import FullPageGrandStaffDataset
    from jazzmus.dataset.smt_dataset import GrandStaffDataset
    
    datamodule = FullPageGrandStaffDataset(data_path="data/jazzmus_fullpage", fold=fold, batch_size=1)
    fullpage_w2i = datamodule.train_set.w2i
    fullpage_i2w = datamodule.train_set.i2w
    
    print(f"   Full-page vocab size: {len(fullpage_w2i)}")
    
    # Convert to sets for comparison
    ckpt_tokens = set(ckpt_w2i.keys())
    fullpage_tokens = set(fullpage_w2i.keys())
    
    # Find differences
    only_in_ckpt = ckpt_tokens - fullpage_tokens
    only_in_fullpage = fullpage_tokens - ckpt_tokens
    common_tokens = ckpt_tokens & fullpage_tokens
    
    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nCheckpoint vocab size:  {len(ckpt_w2i)}")
    print(f"Full-page vocab size:   {len(fullpage_w2i)}")
    print(f"Common tokens:          {len(common_tokens)}")
    print(f"Only in checkpoint:     {len(only_in_ckpt)}")
    print(f"Only in full-page:      {len(only_in_fullpage)}")
    
    # Show tokens only in checkpoint
    if only_in_ckpt:
        print(f"\n{'=' * 80}")
        print(f"TOKENS ONLY IN CHECKPOINT ({len(only_in_ckpt)} tokens)")
        print(f"{'=' * 80}")
        sorted_ckpt_only = sorted(only_in_ckpt)
        if len(sorted_ckpt_only) <= 50:
            for token in only_in_ckpt:
                print(f"  '{token}'")
        else:
            print(f"First 25:")
            for token in sorted_ckpt_only[:25]:
                print(f"  '{token}'")
            print(f"\nLast 25:")
            for token in sorted_ckpt_only[-25:]:
                print(f"  '{token}'")
            print(f"\n... and {len(sorted_ckpt_only) - 50} more")
    
    # Show tokens only in full-page
    if only_in_fullpage:
        print(f"\n{'=' * 80}")
        print(f"NEW TOKENS IN FULL-PAGE ({len(only_in_fullpage)} tokens)")
        print(f"{'=' * 80}")
        sorted_fp_only = sorted(only_in_fullpage)
        if len(sorted_fp_only) <= 50:
            for token in only_in_fullpage:
                print(f"  '{token}'")
        else:
            print(f"First 25:")
            for token in sorted_fp_only[:25]:
                print(f"  '{token}'")
            print(f"\nLast 25:")
            for token in sorted_fp_only[-25:]:
                print(f"  '{token}'")
            print(f"\n... and {len(sorted_fp_only) - 50} more")
    
    # Show some common tokens as sanity check
    print(f"\n{'=' * 80}")
    print(f"SAMPLE COMMON TOKENS (showing first 20)")
    print(f"{'=' * 80}")
    for token in sorted(common_tokens)[:20]:
        print(f"  '{token}'")
    
    # Check if indices match for common tokens
    print(f"\n{'=' * 80}")
    print(f"INDEX ALIGNMENT CHECK")
    print(f"{'=' * 80}")
    misaligned = []
    for token in list(common_tokens)[:10]:  # Check first 10 common tokens
        ckpt_idx = ckpt_w2i[token]
        fp_idx = fullpage_w2i[token]
        match = "✓" if ckpt_idx == fp_idx else "✗"
        print(f"  {match} '{token}': ckpt[{ckpt_idx}] vs fullpage[{fp_idx}]")
        if ckpt_idx != fp_idx:
            misaligned.append(token)
    
    if misaligned:
        print(f"\n⚠️  WARNING: {len(misaligned)} tokens have different indices!")
        print("This will cause issues when loading the checkpoint.")
    else:
        print("\n✓ Sample tokens have matching indices")
    
    # Summary recommendation
    print(f"\n{'=' * 80}")
    print("RECOMMENDATION")
    print(f"{'=' * 80}")
    if len(only_in_fullpage) > 0:
        print("⚠️  Full-page dataset has NEW tokens not in the checkpoint!")
        print("    Options:")
        print("    1. Retrain from scratch with full-page vocab")
        print("    2. Extend checkpoint embedding layer to include new tokens")
        print("    3. Filter full-page data to only use checkpoint vocab")
    else:
        print("✓ Full-page dataset vocab is a subset of checkpoint vocab")
        print("  You can safely use the checkpoint!")
    
    return {
        'ckpt_w2i': ckpt_w2i,
        'fullpage_w2i': fullpage_w2i,
        'only_in_ckpt': only_in_ckpt,
        'only_in_fullpage': only_in_fullpage,
        'common_tokens': common_tokens
    }
    
compare_vocabularies("weights/smt_sys_best/smt_pre_syn_medium.ckpt")