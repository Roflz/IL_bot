#!/usr/bin/env python3
# Save current model's legacy (B,100,8) outputs as out_current.npy
import argparse, numpy as np, torch
from model.imitation_hybrid_model import create_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamestate",  default="data\\recording_sessions\\20250827_040359\\06_final_training_data\\gamestate_sequences.npy")
    ap.add_argument("--actions_in", default="data\\recording_sessions\\20250827_040359\\06_final_training_data\\action_input_sequences.npy")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--out", default="data\\recording_sessions\\20250827_040359\\06_final_training_data\\out_current.npy")
    args = ap.parse_args()

    G = np.load(args.gamestate)[:args.samples]
    A = np.load(args.actions_in)[:args.samples]

    model = create_model()
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(G).float(),
                    torch.from_numpy(A).float())  # expects legacy (B,100,8)
    np.save(args.out, out.cpu().numpy())
    print(f"Saved {args.out} with shape {out.shape}")

if __name__ == "__main__":
    main()
