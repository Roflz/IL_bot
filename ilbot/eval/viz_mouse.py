import argparse, glob, os, numpy as np
import matplotlib.pyplot as plt

# action_type mapping used in your model:
# 0=move, 1=click, 2=key_press, 3=key_release, 4=scroll

def valid_mask(sample):
    """Treat an action row as valid if any absolute value > 0 (your training mask logic)."""
    return (np.abs(sample).sum(axis=-1) > 0)

def load_pair(folder, batch_index):
    dpath = sorted(glob.glob(os.path.join(folder, "decoded_batch*.npy")))[batch_index]
    tpath = sorted(glob.glob(os.path.join(folder, "target_batch*.npy")))[batch_index]
    decoded = np.load(dpath)  # (B, 100, 8)
    target  = np.load(tpath)  # (B, 100, 8)
    return decoded, target, os.path.basename(dpath), os.path.basename(tpath)

def plot_xy(ax, x, y, label, mark_clicks=None):
    ax.plot(x, y, linewidth=1.5, label=label)
    if mark_clicks is not None and mark_clicks.any():
        ax.scatter(x[mark_clicks], y[mark_clicks], s=20, marker="o", label=f"{label} clicks")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="quickcheck_out", help="folder with decoded/target npy files")
    ap.add_argument("--batch", type=int, default=0, help="which decoded_batch*.npy/target_batch*.npy index")
    ap.add_argument("--sample", type=int, default=0, help="which sample inside the batch (row in B)")
    ap.add_argument("--max_steps", type=int, default=100, help="limit number of actions to show")
    ap.add_argument("--show_clicks", action="store_true", help="mark click actions (type_id==1)")
    ap.add_argument("--save", type=str, default=None, help="optional path to save PNG; if omitted, just show()")
    args = ap.parse_args()

    decoded, target, dname, tname = load_pair(args.dir, args.batch)
    if args.sample >= decoded.shape[0]:
        raise SystemExit(f"Sample index {args.sample} out of range for batch {dname} (B={decoded.shape[0]})")

    dec = decoded[args.sample]  # (100, 8)
    tgt = target[args.sample]   # (100, 8)

    # mask padded rows
    m_dec = valid_mask(dec)
    m_tgt = valid_mask(tgt)
    m = m_dec & m_tgt  # keep actions valid in both for a fair overlay

    # cap to max_steps
    idx = np.where(m)[0][:args.max_steps]

    if idx.size == 0:
        raise SystemExit("No valid actions to plot (mask empty). Try another sample or batch.")

    # Extract
    x_dec, y_dec = dec[idx, 2], dec[idx, 3]
    x_tgt, y_tgt = tgt[idx, 2], tgt[idx, 3]

    # Click markers (optional)
    click_dec = None
    click_tgt = None
    if args.show_clicks:
        type_dec = dec[idx, 1].astype(int)
        type_tgt = tgt[idx, 1].astype(int)
        click_dec = (type_dec == 1)
        click_tgt = (type_tgt == 1)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_xy(ax, x_tgt, y_tgt, label="target", mark_clicks=click_tgt)
    plot_xy(ax, x_dec, y_dec, label="pred",   mark_clicks=click_dec)

    ax.set_title(f"Mouse path overlay — {dname} / {tname} — sample {args.sample}")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    # Optionally flip Y if your screen coords have origin at top-left
    ax.invert_yaxis()

    plt.tight_layout()
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True) if os.path.dirname(args.save) else None
        plt.savefig(args.save, dpi=150)
        print(f"Saved {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
