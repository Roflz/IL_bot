#!/usr/bin/env python3
"""
Build Offline Training Data CLI

This script reproduces exactly the artifacts produced by the legacy Phase 1 script,
using the modularized shared pipeline. It preserves byte-for-byte output where feasible.
"""

import argparse, sys, os, json, pathlib, shutil, numpy as np
# Compat: code below uses `_json.*`; make it an alias of `json`.
_json = json

# Add parent directory to path for imports
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from ilbot.pipeline.shared_pipeline import (
    load_gamestates, load_actions, load_existing_features, load_feature_mappings,
    load_gamestates_metadata, load_raw_action_data, load_action_targets,
    extract_features_from_gamestates, extract_raw_action_data, convert_raw_actions_to_tensors,
    create_temporal_sequences, create_screenshot_paths,
    normalize_features, normalize_input_sequences, normalize_action_data,
    save_training_data, save_final_training_data, validate_data_files,
    derive_encodings_from_data, derive_encodings_from_raw_actions,
    save_organized_training_data
)
from ilbot.pipeline.shared_pipeline.sequences import trim_sequences, create_temporal_sequences, create_screenshot_paths
from ilbot.pipeline.shared_pipeline.io_offline import build_gamestates_metadata, load_gamestates as load_gamestates_sorted


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Build offline training data using shared pipeline modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use existing extracted features (recommended)
  python tools/build_offline_training_data.py --use-existing-features
  
  # Extract features from gamestates (slower)
  python tools/build_offline_training_data.py --extract-features
  
  # Custom data directory
  python tools/build_offline_training_data.py --data-dir /path/to/data --use-existing-features
  
  # Custom trimming (trim 10 from start, 30 from end)
  python tools/build_offline_training_data.py --trim-start 10 --trim-end 30
  
  # Minimal trimming (just safety buffer)
  python tools/build_offline_training_data.py --trim-start 5 --trim-end 0
  
  # No trimming (keep all data)
  python tools/build_offline_training_data.py --trim-start 0 --trim-end 0
        """
    )
    
    parser.add_argument('--data-dir', default='data',
                        help='Root data directory (parent of recording_sessions)')
    parser.add_argument(
        '--session',
        default=None,
        help='Recording session id (e.g. 20250824_183745). If omitted, the newest folder under data/recording_sessions is used.'
    )
    parser.add_argument(
        '--session-dir',
        default=None,
        help='Full path to a session directory. Overrides --session.'
    )
    
    # Fresh every time; keep flags for compatibility but ignore them
    parser.add_argument('--use-existing-features', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--extract-features', action='store_true', help=argparse.SUPPRESS)
    
    parser.add_argument(
        '--output-dir',
        default='data/training_data',
        help='Output directory for training data (default: data/training_data)'
    )
    
    # per-session subfolders are derived below; no more global defaults
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate data files, don\'t process'
    )
    
    # Trimming control arguments
    parser.add_argument(
        '--trim-start',
        type=int,
        default=5,
        help='Number of gamestates to trim from the start of the session (default: 5, minimum safety buffer)'
    )
    parser.add_argument(
        '--trim-end',
        type=int,
        default=20,
        help='Number of gamestates to trim from the end of the session (default: 20, session boundaries)'
    )
    
    parser.add_argument("--emit_v2", action="store_true",
                        help="Also emit event-centric V2 targets + manifest.")
    parser.add_argument("--time_div", type=float, default=1000.0, help="Divide raw milliseconds by this to get seconds")
    parser.add_argument("--time_clip", type=float, default=10.0, help="Clip Δt seconds to at most this")
    parser.add_argument("--time_transform", choices=["none","log1p"], default="none",
        help="Transform for time target: store raw seconds or log1p(seconds)")
    # V1 → V2 conversion needs to know how V1 time was scaled (your pipeline uses ms/180)
    parser.add_argument("--v1_time_scale", type=float, default=180.0,
        help="V1 time is stored in ms/this_scale (default 180). Used only when converting V1→V2.")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BUILD OFFLINE TRAINING DATA")
    print("=" * 60)
    
    print("Validating session inputs...")
    # Resolve session directory
    base = pathlib.Path(args.data_dir)
    if args.session_dir:
        session_root = pathlib.Path(args.session_dir)
    else:
        sessions_root = base / "recording_sessions"
        if args.session:
            session_root = sessions_root / args.session
        else:
            # pick newest folder
            candidates = sorted(sessions_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                print("❌ No recording_sessions found.")
                return 1
            session_root = candidates[0]
    session_root = session_root.resolve()
    print(f"Session: {session_root}")

    # Derive all per-session paths
    gamestates_dir = session_root / "gamestates"
    actions_csv    = session_root / "actions.csv"
    raw_dir        = session_root / "01_raw_data"
    trimmed_dir    = session_root / "02_trimmed_data"
    normalized_dir = session_root / "03_normalized_data"
    mappings_dir   = session_root / "05_mappings"
    final_dir      = session_root / "06_final_training_data"

    # Validate minimal inputs for the session
    validation_results = {"gamestates": gamestates_dir.exists(), "actions_csv": actions_csv.exists()}
    ok = all(validation_results.values())
    for k,v in validation_results.items():
        print(f" - {k}: {'OK' if v else 'MISSING'}")
    if not ok:
        print("❌ Missing required session inputs.")
        return 1
    
    if args.validate_only:
        print("Validation complete. Exiting.")
        return 0

    # Always build fresh — wipe output subdirs then rebuild
    def _clean_dir(p: pathlib.Path):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    print("\n🧹 Cleaning output folders (fresh build)…")
    for p in (raw_dir, trimmed_dir, normalized_dir, mappings_dir, final_dir):
        _clean_dir(p)

    try:
        print("\n🔍 Extracting features from session gamestates (fresh)…")
        process_fresh(session_root, gamestates_dir, actions_csv,
                      raw_dir, trimmed_dir, normalized_dir, mappings_dir, final_dir,
                      args.trim_start, args.trim_end,
                      emit_v2=args.emit_v2,
                      time_div=args.time_div,
                      time_clip=args.time_clip,
                      time_transform=args.time_transform,
                      v1_time_scale=args.v1_time_scale)
        
        print("\n✅ Training data build completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Training data build failed: {e}")
        raise


def process_fresh(session_root, gamestates_dir, actions_csv,
                  raw_dir, trimmed_dir, normalized_dir, mappings_dir, final_dir,
                  trim_start=0, trim_end=0,
                  emit_v2=False, time_div=1000.0, time_clip=3.0,
                  time_transform="log1p", v1_time_scale=180.0):
    """Fresh, session-scoped build. No fallbacks; no external metadata."""
    # Load session gamestates (sorted) for use throughout this function
    gamestates = load_gamestates_sorted(str(gamestates_dir))
    
    # ---------- Helpers ----------
    def _match_start_index(raw_np: np.ndarray, trimmed_np: np.ndarray, tol: float = 1e-6) -> int:
        """Find where the first few rows of trimmed_np occur inside raw_np."""
        if raw_np.ndim != 2 or trimmed_np.ndim != 2:
            return 0
        K = min(3, trimmed_np.shape[0])
        tgt = trimmed_np[:K]
        N = raw_np.shape[0] - K + 1
        for s in range(max(N, 0)):
            if not np.allclose(raw_np[s], tgt[0], atol=tol, rtol=0):
                continue
            if K == 1 or np.allclose(raw_np[s:s+K], tgt, atol=tol, rtol=0):
                return s
        return 0

    def _write_slice_info(path: str, start_idx_raw: int, end_idx_raw: int, count: int,
                          first_ts: int, last_ts: int) -> None:
        info = {
            "start_idx_raw": int(start_idx_raw) if start_idx_raw is not None else None,
            "end_idx_raw": int(end_idx_raw) if end_idx_raw is not None else None,
            "count": int(count),
            "first_timestamp": int(first_ts) if first_ts is not None else None,
            "last_timestamp": int(last_ts) if last_ts is not None else None,
        }
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            _json.dump(info, f, indent=2)



    # ---- 1) Extract features fresh -------------------------------------------------
    print("🔍 Extracting features from session gamestates (fresh)…")
    # IMPORTANT: do NOT write global mappings; we will save under the session mappings dir.
    features, feature_mappings, feature_timestamps = extract_features_from_gamestates(
        gamestates, save_mappings=False
    )

    # ---- 2) Build session-local metadata from JSONs --------------------------------
    gamestates_metadata = build_gamestates_metadata(str(gamestates_dir))
    (mappings_dir / "gamestates_metadata.json").write_text(_json.dumps(gamestates_metadata, indent=2))

    # ---- 3) Raw action data --------------------------------------------------------
    print("Extracting raw action data…")
    raw_action_data = extract_raw_action_data(gamestates, str(actions_csv), align_to_gamestates=True)
    (raw_dir / "raw_action_data.json").write_text(_json.dumps(raw_action_data, indent=2))

    # ---- 4) Trimming (features + action_data) --------------------------------------
    print(f"Applying data trimming (start: {trim_start}, end: {trim_end})…")
    trimmed_features, trimmed_raw_action_data, start_idx, end_idx = trim_sequences(
        features, raw_action_data, trim_start, trim_end
    )
    # DO NOT mutate raw metadata. Keep a raw copy and make a trimmed view when needed.
    gamestates_metadata_raw = gamestates_metadata
    gamestates_metadata_trim = gamestates_metadata_raw[start_idx: len(features) - end_idx]

    # ---- 5) Normalize features -----------------------------------------------------
    print("Normalizing features…")
    # Save session-local feature mappings (no globals).
    (mappings_dir / "feature_mappings.json").write_text(_json.dumps(feature_mappings, indent=2))
    normalized_features = normalize_features(trimmed_features, str(mappings_dir / "feature_mappings.json"))
    # Save raw features + ids
    np.save(raw_dir / "state_features.npy", features)
    (mappings_dir / "id_mappings.json").write_text(_json.dumps(feature_timestamps, indent=2))
    
    # ---- 7) Build sequences --------------------------------------------------------
    print("Creating temporal sequences from raw features…")
    input_sequences, action_input_sequences, target_sequences = create_temporal_sequences(
        trimmed_features, trimmed_raw_action_data
    )
    
    # Create temporal sequences from normalized features
    print("Creating temporal sequences from normalized features...")
    normalized_input_sequences, _, _ = create_temporal_sequences(
        normalized_features, trimmed_raw_action_data
    )
    
    # Use raw sequences as the default input sequences
    # (input_sequences is already set above)
    
    # Normalize action data
    print("Normalizing action data...")
    normalized_action_data = normalize_action_data(trimmed_raw_action_data, normalized_features)

    # Create normalized sequences from normalized action data
    print("Creating normalized action sequences...")
    _, normalized_action_input_sequences, normalized_target_sequences = create_temporal_sequences(
        normalized_features, normalized_action_data
    )

    # (optional but helpful) persist the trimmed raw actions for inspection
    try:
        (trimmed_dir / "trimmed_raw_action_data.json").write_text(_json.dumps(trimmed_raw_action_data, indent=2))
    except Exception:
        pass
    
    # ---- 8) Screenshot paths & metadata -------------------------------------------
    # For screenshots counts, the trimmed view is what the UI expects.
    screenshot_paths = create_screenshot_paths(gamestates_metadata_trim, str(session_root / "runelite_screenshots"))
    
    # Load id_mappings if available (existing-features mode)
    id_mappings_path = pathlib.Path(mappings_dir / "id_mappings.json")
    if id_mappings_path.exists():
        try:
            with open(id_mappings_path, 'r') as f:
                id_mappings = _json.load(f)
            print("Loaded id_mappings from existing features")
        except Exception as e:
            print(f"⚠ Warning: Failed to load id_mappings: {e}")
            id_mappings = {}
    else:
        id_mappings = {}
    
    # Build trimmed action data to match feature trim window
    trimmed_raw_action_data = raw_action_data[start_idx : start_idx + len(trimmed_features)]

    save_organized_training_data(str(raw_dir), str(trimmed_dir), str(normalized_dir),
                                 str(mappings_dir), str(final_dir),
                                 features,
                                 trimmed_features, normalized_features,
                                 input_sequences, normalized_input_sequences,
                                 target_sequences, action_input_sequences,
                                 raw_action_data, trimmed_raw_action_data,
                                 normalized_action_data, feature_mappings,
                                 id_mappings,
                                 gamestates_metadata_raw,
                                 screenshot_paths,
                                 normalized_target_sequences,
                                 normalized_action_input_sequences
                                 )



    # ----- Build V2 outputs + manifest (optional) -----
    if emit_v2:
        print("\n🧩 Building V2 event-centric targets + manifest…")
        # Convert finalized V1 targets → V2 (no undefined helpers, no frame_ts).
        at_v1_path = final_dir / "action_targets.npy"
        if not at_v1_path.exists():
            raise FileNotFoundError(f"Missing {at_v1_path} to build V2 targets.")
        at_v1 = np.load(at_v1_path)  # shape (N, 100, 8), legacy layout

        actions_v2, valid_mask = _v1_to_v2(
            at_v1,
            time_div=time_div,
            time_clip=time_clip,
            time_transform=time_transform,
            v1_time_scale=v1_time_scale
        )

        np.save(final_dir / "actions_v2.npy", actions_v2.astype(np.float32))
        np.save(final_dir / "valid_mask.npy", valid_mask.astype(np.bool_))
        print(f"  ✓ actions_v2.npy: {actions_v2.shape}")
        print(f"  ✓ valid_mask.npy: {valid_mask.shape}")

        # Build a minimal manifest (no hard-coded shapes)
        gs = np.load(final_dir / "gamestate_sequences.npy")
        ai = np.load(final_dir / "action_input_sequences.npy")
        manifest = {
            "targets_version": "v2",
            "enum_sizes": {"button": 4, "key_action": 3, "key_id": 151, "scroll_y": 3},
            "shapes": {
                "gamestate_sequences": list(gs.shape),
                "action_input_sequences": list(ai.shape),
                "actions_v2": list(actions_v2.shape),
                "valid_mask": list(valid_mask.shape),
            },
            "time": {"div": float(time_div), "clip": float(time_clip), "transform": str(time_transform)},
            "targets_alignment": "within_window_deltas",
            "action_input": {
                "shape": [int(gs.shape[0]), int(gs.shape[1]), int(ai.shape[2]), int(ai.shape[3])],
                "timestamp_semantics": "absolute_or_session_relative_ms",  # padding=0
            },
            # keep both keys for downstream compatibility; 'click' is the canonical name
            "enums": {"click": 4, "button": 4, "key_action": 3, "key_id": 151, "scroll_y": 3},
            "time": {"transform": time_transform, "div": time_div, "clip": time_clip}
        }
        with open(final_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
            _json.dump(manifest, f, indent=2)
        print("  ✓ dataset_manifest.json (no hard-coded shapes)")

        # Update metadata with V2 time info
        meta_path = final_dir / "metadata.json"
        try:
            meta = _json.loads((final_dir / "metadata.json").read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        meta.setdefault("targets_v2", {})
        meta["targets_v2"]["layout"] = ["time", "x", "y", "click", "key_action", "key_id", "scroll_y"]
        meta["targets_v2"]["time"] = {
            "unit": "sec",
            "div": float(time_div),
            "clip": float(time_clip),
            "transform": str(time_transform),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            _json.dump(meta, f, indent=2)

    print("\n🎯 ORGANIZED TRAINING DATA SAVED!")
    print("============================================================")

    # -------------- RAW SLICE INFO --------------
    raw_N = int(features.shape[0])
    _write_slice_info(
        str(raw_dir / "slice_info.json"),
        0, raw_N - 1, raw_N,
        first_ts=gamestates_metadata_raw[0]["absolute_timestamp"],
        last_ts=gamestates_metadata_raw[raw_N - 1]["absolute_timestamp"]
    )
    
    # -------------- TRIMMED SLICE INFO --------------
    # Use RAW metadata for absolute references to avoid index errors
    s_trim = start_idx
    e_trim = start_idx + len(trimmed_features) - 1
    _write_slice_info(
        str(trimmed_dir / "slice_info.json"),
        s_trim, e_trim, len(trimmed_features),
        first_ts=gamestates_metadata_raw[s_trim]["absolute_timestamp"],
        last_ts=gamestates_metadata_raw[e_trim]["absolute_timestamp"]
    )
    
    # -------------- FINAL TRAINING SLICE INFO --------------
    seq_B = int(input_sequences.shape[0])                 # 34
    first_seq_raw_idx = start_idx                         # 5 in your log
    last_target_raw_idx = start_idx + len(trimmed_features) - 1  # 5+45-1 = 49
    _write_slice_info(
        str(final_dir / "slice_info.json"),
        first_seq_raw_idx,
        last_target_raw_idx,
        seq_B,
        first_ts=gamestates_metadata_raw[first_seq_raw_idx]["absolute_timestamp"],
        last_ts=gamestates_metadata_raw[last_target_raw_idx]["absolute_timestamp"]
    )

    # -------------- NORMALIZED SLICE INFO --------------
    # Normalized features are a 1:1 transform of trimmed features; reuse start/end.
    _write_slice_info(
        str(normalized_dir / "slice_info.json"),
        s_trim, e_trim, len(trimmed_features),
        first_ts=gamestates_metadata_raw[s_trim]["absolute_timestamp"],
        last_ts=gamestates_metadata_raw[e_trim]["absolute_timestamp"]
    )

    # end of process_fresh

def _save_np(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

def _v1_to_v2(at_v1, time_div, time_clip, time_transform, v1_time_scale):
    """
    Convert V1 action targets (N,100,8) to V2 event-centric targets (N,100,7)
    V1 layout (per metadata): [timestamp, type, x, y, button, key, scroll_dx, scroll_dy]
    V2 layout we emit:        [time,      x,   y, click, key_action, key_id, scroll_y]
    """
    # ----- read correct V1 columns -----
    t_raw  = at_v1[..., 0].astype(np.float32)   # absolute or session-relative timestamp in (ms / v1_time_scale)
    ty     = at_v1[..., 1].astype(np.int64)     # V1 action type (0=move,1=click,2=key press,3=key release,4=scroll)
    x      = at_v1[..., 2].astype(np.float32)
    y      = at_v1[..., 3].astype(np.float32)
    b1     = at_v1[..., 4].astype(np.int64)     # V1 button code
    kid_v1 = at_v1[..., 5].astype(np.int64)     # V1 key id
    sy_v1  = at_v1[..., 7].astype(np.float32)   # V1 scroll dy in {-1,0,+1}

    # ----- build Δt per row along action axis -----
    # Convert to ms, diff along the 100-slot axis, keep last Δt = previous Δt, clamp non-negative
    ts_ms = t_raw * float(v1_time_scale)                     # (N,100) in ms
    dt_ms = np.diff(ts_ms, axis=1, append=ts_ms[..., -1:])
    if dt_ms.shape[-1] >= 2:
        dt_ms[..., -1] = dt_ms[..., -2]
    dt_ms = np.clip(dt_ms, 0.0, None)
    # scale to seconds and apply optional clip/log1p
    dt_sec = dt_ms / float(time_div)
    if time_clip is not None:
        dt_sec = np.clip(dt_sec, 0.0, float(time_clip))
    t_store = np.log1p(dt_sec).astype(np.float32) if time_transform == "log1p" else dt_sec.astype(np.float32)

    # ----- click mapping (maintain consistency with V1) -----
    click = b1.astype(np.int64)  # Keep original button codes: 0=none, 1=left, 2=right, 3=middle

    key_action = np.zeros_like(ty)              # 0=None, 1=Press, 2=Release
    key_action[ty == 2] = 1
    key_action[ty == 3] = 2
    key_id = np.where(key_action > 0, kid_v1, 0).astype(np.int64)

    # ----- scroll mapping (correct values: -1=down, 0=none, 1=up) -----
    scroll_y = np.where(sy_v1 < 0, -1, np.where(sy_v1 > 0, 1, 0)).astype(np.int64)

    a_v2 = np.stack([t_store, x, y,
                     click.astype(np.float32),
                     key_action.astype(np.float32),
                     key_id.astype(np.float32),
                     scroll_y.astype(np.float32)], axis=-1).astype(np.float32)
    # Valid iff any event signal present (don't use absolute t for validity)
    valid_mask = ((ty > 0) | (b1 > 0) | (kid_v1 > 0) | (sy_v1 != 0))
    return a_v2, valid_mask






if __name__ == "__main__":
    sys.exit(main())
