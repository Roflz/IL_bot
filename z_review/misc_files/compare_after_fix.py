#!/usr/bin/env python3
# Compare out_current.npy with outputs after refactor (logits + decode)
import argparse, numpy as np, torch, sys
from model.imitation_hybrid_model import create_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamestate",  default="data\\recording_sessions\\20250827_040359\\06_final_training_data\\gamestate_sequences.npy")
    ap.add_argument("--actions_in", default="data\\recording_sessions\\20250827_040359\\06_final_training_data\\action_input_sequences.npy")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--baseline", default="data\\recording_sessions\\20250827_040359\\06_final_training_data\\out_current.npy")
    ap.add_argument("--save-fixed", default="data\\recording_sessions\\20250827_040359\\06_final_training_data\\out_fixed.npy")
    args = ap.parse_args()

    G = np.load(args.gamestate)[:args.samples]
    A = np.load(args.actions_in)[:args.samples]
    base = np.load(args.baseline)

    model = create_model()
    model.eval()

    with torch.no_grad():
        try:
            heads = model(torch.from_numpy(G).float(),
                          torch.from_numpy(A).float(),
                          return_logits=True)  # requires your Step 1 refactor
        except TypeError:
            print("Model does not support return_logits=True yet. Implement Step 1 first.")
            sys.exit(1)
        if not hasattr(model, "action_decoder") or not hasattr(model.action_decoder, "decode"):
            print("Decoder lacks `decode(heads)`. Implement Step 1 first.")
            sys.exit(1)
        out_fixed = model.action_decoder.decode(heads)  # (B,100,8)

    out_fixed_np = out_fixed.cpu().numpy()
    if out_fixed_np.shape != base.shape:
        print("Shape mismatch:", out_fixed_np.shape, "vs", base.shape)
        sys.exit(1)

    diff = (out_fixed_np - base).astype(np.float32)
    print("=== Comparison (fixed vs current) ===")
    print("abs mean:", np.abs(diff).mean(), " | abs max:", np.abs(diff).max())
    # Show a tiny slice for eyeballing
    print("sample[0, :5, :]:\nfixed:\n", out_fixed_np[0, :5, :], "\ncurrent:\n", base[0, :5, :])

    np.save(args.save_fixed, out_fixed_np)
    print(f"Saved fixed outputs → {args.save_fixed}")

if __name__ == "__main__":
    main()
