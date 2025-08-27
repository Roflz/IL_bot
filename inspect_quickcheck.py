import argparse, glob, os, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="quickcheck_out", help="folder with decoded_batch*.npy & target_batch*.npy")
    args = ap.parse_args()

    dec_files = sorted(glob.glob(os.path.join(args.dir, "decoded_batch*.npy")))
    tgt_files = sorted(glob.glob(os.path.join(args.dir, "target_batch*.npy")))
    if not dec_files or not tgt_files:
        raise SystemExit(f"No npy files found in {args.dir}. Did quickcheck run?")

    print(f"Found {len(dec_files)} decoded files and {len(tgt_files)} target files")
    for di, (df, tf) in enumerate(zip(dec_files, tgt_files)):
        dec = np.load(df)
        tgt = np.load(tf)
        print(f"\n[{di}] {os.path.basename(df)} vs {os.path.basename(tf)}")
        print(f" decoded: shape={dec.shape}, dtype={dec.dtype}")
        print(f" target : shape={tgt.shape}, dtype={tgt.dtype}")
        # Columns: [time_ms, type_id, x, y, button_id, key_id, scroll_dx, scroll_dy]
        # Simple stats (x/y only)
        def stats(a, name):
            x = a[..., 2]; y = a[..., 3]
            print(f"  {name}: x[min={x.min():.1f}, max={x.max():.1f}, mean={x.mean():.1f}]  "
                  f"y[min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f}]")
        stats(dec, "decoded")
        stats(tgt, "target")

        # Show a tiny peek of the first sample’s first 5 rows
        s = 0
        print("  head (sample 0, first 5 actions):")
        print("   decoded:\n", dec[s, :5])
        print("   target :\n", tgt[s, :5])

if __name__ == "__main__":
    main()
