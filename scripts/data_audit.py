# scripts/data_audit.py
import os, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def qstats(x):
    x = np.asarray(x).reshape(-1)
    return {
        "count": int(x.size),
        "nan": int(np.isnan(x).sum()),
        "min": float(np.nanmin(x)) if x.size else None,
        "p10": float(np.nanpercentile(x,10)) if x.size else None,
        "p25": float(np.nanpercentile(x,25)) if x.size else None,
        "median": float(np.nanmedian(x)) if x.size else None,
        "mean": float(np.nanmean(x)) if x.size else None,
        "p75": float(np.nanpercentile(x,75)) if x.size else None,
        "p90": float(np.nanpercentile(x,90)) if x.size else None,
        "max": float(np.nanmax(x)) if x.size else None,
    }

def main(data_dir="data/recording_sessions/20250831_113719/06_final_training_data", out_dir="data_profile"):
    p = Path(data_dir)
    # Anchor report directory relative to the data root to avoid CWD surprises:
    # <repo>/data/recording_sessions/.../<run>/06_final_training_data  ->  <repo>/data/<out_dir>
    data_root = Path(data_dir).resolve().parents[2]   # .../<repo>/data
    if data_root.name != "data":
        raise RuntimeError(f"Expected data root to be .../data, got {data_root}")
    out = (data_root / out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    gs = np.load(p/"gamestate_sequences.npy")      # [N,T,Dg]
    ai = np.load(p/"action_input_sequences.npy")   # [N,T,A,Fa]
    tg = np.load(p/"action_targets.npy")           # [N,A,7] (time_ms, x, y, button, key_action, key_id, scroll_y)

    N,T,Dg = gs.shape
    _,_,A,Fa = ai.shape
    assert tg.shape == (N,A,7), f"targets shape {tg.shape} != (N,A,7)"

    # Inactive convention check (0 vs -1)
    btn_inactive_is_zero = np.all((tg[...,3] == 0) | (tg[...,3] > 0))
    key_inactive_is_zero = np.all((tg[...,4] == 0) | (tg[...,4] > 0))
    kid_inactive_is_zero = np.all((tg[...,5] == 0) | (tg[...,5] > 0))

    # Valid mask (mirror your current logic): time_seconds>0 OR any nonzero mark
    time_seconds = tg[...,0] / 1000.0
    marks = np.stack([
        (tg[...,3] != 0),
        (tg[...,4] != 0),
        (tg[...,5] != 0),
        (tg[...,6] != 0),
    ], axis=-1)
    any_mark = marks.any(axis=-1)
    valid = (time_seconds > 0) | any_mark
    


    # Event derivation (canonical: 0=CLICK,1=KEY,2=SCROLL,3=MOVE default)
    ev = np.full((N,A), 3, dtype=np.int64)
    ev = np.where(tg[...,3] != 0, 0, ev)
    ev = np.where(tg[...,4] != 0, 1, ev)
    ev = np.where(tg[...,6] != 0, 2, ev)

    # Screen bounds (declare + verify via norm.json; create if missing)
    norm_path = out/"norm.json"
    obs_x = float(np.nanmax(tg[...,1])) if tg.size else 0.0
    obs_y = float(np.nanmax(tg[...,2])) if tg.size else 0.0
    eps = 1e-6
    
    # Initialize bounds
    x_max = obs_x
    y_max = obs_y
    
    if norm_path.exists():
        try:
            norm = json.loads(norm_path.read_text())
            x_max = float(norm["x_max"]); y_max = float(norm["y_max"])
        except Exception as e:
            raise RuntimeError(f"Failed to read/parse {norm_path}: {e}")
        # Do NOT auto-update; raise if observed exceeds declared bounds
        if obs_x > x_max + eps or obs_y > y_max + eps:
            raise RuntimeError(
                f"Observed XY exceeds declared bounds in {norm_path}: "
                f"observed_x_max={obs_x:.6f} (declared {x_max:.6f}), "
                f"observed_y_max={obs_y:.6f} (declared {y_max:.6f}). "
                "Increment x_max/y_max in norm.json explicitly or replace it, then rerun."
            )
    else:
        # First run: create norm.json with observed maxima (explicit contract)
        norm_path.write_text(json.dumps({"x_max": x_max, "y_max": y_max}, indent=2))

    # Core profiles
    report = {
        "shapes": {"gamestate": [int(N),int(T),int(Dg)], "action_input": [int(N),int(T),int(A),int(Fa)], "targets": [int(N),int(A),7]},
        "inactive_convention": {"button_zero_inactive": bool(btn_inactive_is_zero), "key_action_zero_inactive": bool(key_inactive_is_zero), "key_id_zero_inactive": bool(kid_inactive_is_zero)},
        "valid_mask": {"fraction": float(valid.mean()), "count_true": int(valid.sum()), "count_total": int(valid.size)},
        "events": {
            "counts": {k:int((ev==k).sum()) for k in [0,1,2,3]},
            "fractions": {k: float((ev==k).mean()) for k in [0,1,2,3]},
            "names": {0:"CLICK",1:"KEY",2:"SCROLL",3:"MOVE"},
        },
        "time_seconds_overall": qstats(time_seconds[valid]),
        "time_seconds_by_event": {name: qstats(time_seconds[(ev==eid) & valid]) for eid,name in {0:"CLICK",1:"KEY",2:"SCROLL",3:"MOVE"}.items()},
        "x_norm_overall": qstats((tg[...,1]/x_max)[valid]),
        "y_norm_overall": qstats((tg[...,2]/y_max)[valid]),
    }

    # XY by event (only where meaningful: CLICK or MOVE typically)
    for eid, name in {0:"CLICK",3:"MOVE"}.items():
        m = (ev==eid) & valid
        report[f"x_norm_{name.lower()}"] = qstats((tg[...,1]/x_max)[m])
        report[f"y_norm_{name.lower()}"] = qstats((tg[...,2]/y_max)[m])

    # Actions per sample
    actions_per_sample = valid.sum(axis=1)  # [N]
    report["actions_per_sample"] = qstats(actions_per_sample)

    # Time coverage per sample (sum of deltas)
    time_coverage_seconds = time_seconds * valid
    time_coverage_seconds = time_coverage_seconds.sum(axis=1)   # [N]
    report["time_coverage_seconds_per_sample"] = qstats(time_coverage_seconds)

    report["xy_bounds_observed"] = {"min_x": float(np.nanmin(tg[...,1])) if tg.size else 0.0,
                                    "max_x": obs_x,
                                    "min_y": float(np.nanmin(tg[...,2])) if tg.size else 0.0,
                                    "max_y": obs_y}
    report["xy_bounds_declared"] = {"x_max": x_max, "y_max": y_max}

    # Save JSON
    with open(out/"data_profile.json","w") as f:
        json.dump(report, f, indent=2)

    # Plots (saved as PNGs)
    def save_hist(arr, title, fname, bins=100, xlim=None):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0: return
        plt.figure()
        plt.hist(arr, bins=bins)
        plt.title(title); plt.xlabel(title); plt.ylabel("count")
        if xlim: plt.xlim(*xlim)
        plt.tight_layout()
        plt.savefig(out/fname); plt.close()

    save_hist(time_seconds[valid], "time_seconds (valid)", "time_seconds_valid.png", bins=200)
    save_hist((tg[...,1]/x_max)[valid], "x_norm [0,1] (valid)", "x_norm_valid.png", bins=200, xlim=(0,1))
    save_hist((tg[...,2]/y_max)[valid], "y_norm [0,1] (valid)", "y_norm_valid.png", bins=200, xlim=(0,1))
    for eid,name in {0:"CLICK",1:"KEY",2:"SCROLL",3:"MOVE"}.items():
        m = (ev==eid) & valid
        save_hist(time_seconds[m], f"time_seconds by event={name}", f"time_seconds_{name.lower()}.png", bins=200)
    for eid,name in {0:"CLICK",3:"MOVE"}.items():
        m = (ev==eid) & valid
        save_hist((tg[...,1]/x_max)[m], f"x_norm [0,1] by event={name}", f"x_norm_{name.lower()}.png", bins=200, xlim=(0,1))
        save_hist((tg[...,2]/y_max)[m], f"y_norm [0,1] by event={name}", f"y_norm_{name.lower()}.png", bins=200, xlim=(0,1))

    print(json.dumps(report, indent=2))
    print(f"\n✅ Wrote profile to {out/'data_profile.json'} and histograms to {out}/")

if __name__ == "__main__":
    # Allow override via env var
    main(os.environ.get("DATA_DIR", "data/recording_sessions/20250831_113719/06_final_training_data"))
