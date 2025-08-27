#!/usr/bin/env python3
import argparse, re, json, numpy as np, torch, os, sys, textwrap

def grep_with_context(path, pattern, context=3):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    pat = re.compile(pattern)
    hits = []
    for i, line in enumerate(lines):
        if pat.search(line):
            start = max(0, i-context)
            end   = min(len(lines), i+context+1)
            snippet = "".join(f"{j+1:4d}: {lines[j]}" for j in range(start, end))
            hits.append(snippet)
    return hits

def scan_model(model_file):
    patterns = {
        "argmax":  r"argmax\(",
        "round":   r"\.round\(",
        "sign":    r"\.sign\(",
        "softmax": r"softmax\(",
        "tanh":    r"tanh\(",
        "sigmoid": r"sigmoid\(",
    }
    out = {}
    for name, pat in patterns.items():
        out[name] = grep_with_context(model_file, pat, context=3)
    return out

def load_stats(npy_path):
    arr = np.load(npy_path, mmap_mode="r")
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
    }

def show_autograd_proofs():
    print("\n=== AUTOGRAD PROOFS ===")
    torch.manual_seed(0)

    def show_gradients(name, loss_fn):
        x = torch.randn(4, 5, requires_grad=True)  # "logits" toy
        loss = loss_fn(x)
        try:
            loss.backward()
        except Exception as e:
            print(f"[{name}] backward ERROR: {e}")
            return
        grad_sum = x.grad.abs().sum().item() if x.grad is not None else None
        print(f"[{name}] loss={float(loss):.6f} | grad_sum(|∂loss/∂x|)={grad_sum}")

    # Categorical: BAD (softmax→argmax) vs GOOD (CE on logits)
    y_true = torch.tensor([1, 0, 3, 2])
    def loss_bad_argmax(logits):
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1)    # non-diff
        return torch.mean((pred_idx.float() - y_true.float())**2)

    def loss_good_ce(logits):
        return torch.nn.functional.cross_entropy(logits, y_true)

    show_gradients("Categorical BAD (softmax→argmax + MSE)", loss_bad_argmax)
    show_gradients("Categorical GOOD (CrossEntropy on logits)", loss_good_ce)

    # Coordinates: BAD (round) vs GOOD (regression)
    target_xy = torch.tensor([[100.0, 200.0, 50.0, 10.0, 900.0]])
    def loss_bad_round(logits):
        xy_logits = logits[:, :2]
        xy = torch.sigmoid(xy_logits) * 1000.0
        xy_rounded = torch.round(xy)              # non-diff
        tgt = target_xy[:, :2].expand_as(xy_rounded)
        return torch.mean((xy_rounded - tgt)**2)

    def loss_good_regression(logits):
        xy = logits[:, :2]                        # raw reals
        tgt = target_xy[:, :2].expand_as(xy)
        return torch.nn.functional.smooth_l1_loss(xy, tgt)

    show_gradients("Coords BAD (sigmoid→scale→round + MSE)", loss_bad_round)
    show_gradients("Coords GOOD (SmoothL1 on reals)", loss_good_regression)

    # Scroll: BAD (sign) vs GOOD (3-class CE)
    scroll_labels = torch.tensor([0, 1, 2, 1])
    def loss_bad_sign(logits):
        s = torch.tanh(logits[:, 0])
        s_disc = torch.sign(s)
        mapped = torch.tensor([-1.0, 0.0, 1.0, 0.0])
        return torch.mean((s_disc - mapped)**2)

    def loss_good_scroll_ce(logits):
        tri_logits = torch.stack([logits[:,0], logits[:,1], logits[:,2]], dim=-1)
        return torch.nn.functional.cross_entropy(tri_logits, scroll_labels)

    show_gradients("Scroll BAD (tanh→sign + MSE)", loss_bad_sign)
    show_gradients("Scroll GOOD (3-class CE on logits)", loss_good_scroll_ce)

def main():
    ap = argparse.ArgumentParser(description="Proof pack: scan code, inspect data, and show autograd gradient proofs.")
    ap.add_argument("--model-file", default="model\\imitation_hybrid_model.py")
    ap.add_argument("--gamestate",  default="data\\recording_sessions\\20250827_040359\\06_final_training_data\\gamestate_sequences.npy")
    ap.add_argument("--actions_in", default="data\\recording_sessions\\20250827_040359\\06_final_training_data\\action_input_sequences.npy")
    ap.add_argument("--actions_y",  default="data\\recording_sessions\\20250827_040359\\06_final_training_data\\action_targets.npy")
    ap.add_argument("--save-report", default="data\\recording_sessions\\20250827_040359\\06_final_training_data\\proof_report.json")
    args = ap.parse_args()

    print("=== 1) CODE SCAN ===")
    scan = scan_model(args.model_file)
    for k, snippets in scan.items():
        print(f"\n-- {k.upper()} ({len(snippets)}) --")
        for s in snippets:
            print(s.rstrip(), "\n")

    print("=== 2) DATA SHAPES & STATS ===")
    report = {"scan": {k: len(v) for k,v in scan.items()}, "data": {}}
    for name, path in [("gamestate", args.gamestate),
                       ("action_input", args.actions_in),
                       ("action_targets", args.actions_y)]:
        if os.path.exists(path):
            st = load_stats(path)
            report["data"][name] = st
            print(f"{name:15s}: shape={st['shape']}, dtype={st['dtype']}, "
                  f"min={st['min']:.4f}, max={st['max']:.4f}, mean={st['mean']:.4f}, std={st['std']:.4f}")
        else:
            print(f"{name:15s}: NOT FOUND @ {path}")

    show_autograd_proofs()

    with open(args.save_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report → {args.save_report}")

if __name__ == "__main__":
    main()
