import os, numpy as np, json, argparse, math

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
args = parser.parse_args()

def main():
    p = args.data_dir
    final_dir = os.path.abspath(p)
    if not os.path.exists(final_dir):
        print(f"❌ Final data dir not found: {final_dir}")
        return
    gs_path = os.path.join(final_dir, "gamestate_sequences.npy")
    if not os.path.exists(gs_path):
        print(f"❌ File not found: {gs_path}")
        print("   Did the build script finish? See console logs for errors.")
        return
    gs = np.load(gs_path)      # (N,10,128)
    ai = np.load(os.path.join(final_dir, "action_input_sequences.npy"))    # (N,10,100,8)
    at_v1 = np.load(os.path.join(final_dir, "action_targets.npy"))            # (N,100,8)
    a_v2 = np.load(os.path.join(final_dir, "actions_v2.npy"))                # (N,100,7)
    vm   = np.load(os.path.join(final_dir, "valid_mask.npy"))                # (N,100)

    print(f"gamestate_sequences: {gs.shape}")
    print(f"action_input_sequences (v1): {ai.shape}")
    print(f"action_targets (v1): {at_v1.shape}")
    print(f"actions_v2: {a_v2.shape}")
    print(f"valid_mask: {vm.shape}  valid_frac={vm.mean():.3f}")

    # V2 ranges on valid rows only
    valid = vm.astype(bool)
    time_raw = a_v2[...,0][valid]
    x    = a_v2[...,1][valid]
    y    = a_v2[...,2][valid]

    def rng(name, arr):
        if arr.size == 0:
            print(f"  {name}: n=0")
            return
        print(f"  {name}: min={np.min(arr):.3f}, p50={np.percentile(arr,50):.3f}, max={np.max(arr):.3f}")

    # Decode time according to manifest (log1p(seconds) or plain seconds)
    man = {}
    man_path = os.path.join(p, "dataset_manifest.json")
    if os.path.exists(man_path):
        with open(man_path, "r") as f:
            man = json.load(f)
    tmeta = man.get("time", {})
    transform = tmeta.get("transform", "none")
    # for readability we show both the stored scale and the decoded seconds
    print("\nRanges:")
    if time_raw.size == 0:
        print("  time: n=0")
    else:
        rng("time_raw", time_raw)
        if transform == "log1p":
            time_sec = np.expm1(time_raw)
            rng("time_sec(decoded)", time_sec)
        else:
            rng("time_sec", time_raw)
    rng("x", x); rng("y", y)

    # Discrete distributions (V2)
    btn = a_v2[...,3][valid].astype(int)
    ka  = a_v2[...,4][valid].astype(int)
    kid = a_v2[...,5][valid & (a_v2[...,4] > 0)].astype(int)
    sy  = a_v2[...,6][valid].astype(int)

    def topk_count(name, arr, k=10):
        if arr.size == 0:
            print(f"{name}: n=0 (no events)")
            return
        vals, counts = np.unique(arr, return_counts=True)
        order = np.argsort(-counts)
        total = counts.sum()
        pairs = [(int(vals[i]), float(counts[i]/total)) for i in order[:k]]
        print(f"{name}: n={total}  dist={pairs}")

    topk_count("click (0=None,1=L,2=R,3=M)", btn)
    topk_count("key_action (0=None,1=Press,2=Release)", ka)
    topk_count("key_id (only where key_action>0)", kid)
    topk_count("scroll_y (0=down,1=none,2=up)", sy)

    # Window length stats = number of valid events per window
    winlens = vm.sum(-1)
    print(f"\nWindow lengths: min={winlens.min()}, p25={np.percentile(winlens,25)}, p50={np.percentile(winlens,50)}, p75={np.percentile(winlens,75)}, max={winlens.max()}")

    # ---------- Action-input timestamp field (context) ----------
    # ai[...,0] is the timestamp column as written by the builder (padding == 0)
    ai_time = ai[...,0].reshape(-1)
    ai_time_nz = ai_time[ai_time != 0]
    print("\nAction input timestamp field (ai[...,0]) over nonzero entries:")
    rng("ai_time_raw", ai_time_nz)

    if man:
        ai_sem = man.get("action_input", {}).get("timestamp_semantics")
        if ai_sem:
            print(f"Manifest: action_input.timestamp_semantics = {ai_sem}")

    # ---------- Gamestate time features ----------
    # Try to find a feature map to locate timestamp/time_since_interaction columns.
    session_root = os.path.abspath(os.path.join(p, os.pardir))
    fmap_path = os.path.join(session_root, "05_mappings", "feature_mappings.json")
    if os.path.exists(fmap_path):
        with open(fmap_path, "r") as f:
            fmap = json.load(f)  # list of {index, name} or similar
        # Try to find indices by name (be tolerant to naming)
        def find_idx(sub):
            for item in fmap:
                name = (item.get("name") or "").lower()
                if sub in name:
                    return int(item["index"])
            return None
        ts_idx  = find_idx("timestamp")
        tsi_idx = find_idx("time_since_interaction")
        pd_idx  = find_idx("phase_duration")

        print("\nGamestate time features (as stored in sequences):")
        if ts_idx is not None:
            ts = gs[..., ts_idx]  # (N,10)
            # Per-window deltas
            dt_win = ts[:, -1] - ts[:, 0]
            rng("gs.timestamp_window_delta", dt_win)
            # Per-step deltas
            dt_step = np.diff(ts, axis=1).reshape(-1)
            dt_step_nz = dt_step[dt_step != 0]
            print("Per-frame delta of gs.timestamp:")
            rng("gs.timestamp_step_delta", dt_step_nz)
        if tsi_idx is not None:
            tsi = gs[..., tsi_idx].reshape(-1)
            rng("gs.time_since_interaction", tsi[tsi != 0])
        if pd_idx is not None:
            pd = gs[..., pd_idx].reshape(-1)
            rng("gs.phase_duration", pd[pd != 0])
    else:
        print("\n(feature_mappings.json not found; skipping gamestate time stats)")

if __name__ == "__main__":
    main()
