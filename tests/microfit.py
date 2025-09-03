# tests/microfit.py
import os, time, torch
from torch.optim import AdamW
from ilbot.config import Config
from ilbot.data.dataset import make_loaders
from ilbot.model.model import SequentialImitationModel
from ilbot.training.losses import AdvancedUnifiedEventLoss
from ilbot.data.contracts import derive_event_targets_from_marks
import torch.nn.functional as F

def main(steps=200):
    data_dir = os.environ.get("DATA_DIR", "data/recording_sessions/20250831_113719/06_final_training_data")
    cfg = Config(data_dir=data_dir, batch_size=32, num_workers=0)
    train_loader, _, dc = make_loaders(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequentialImitationModel(dc, hidden_dim=128, horizon_s=0.6).to(device)
    loss_fn = AdvancedUnifiedEventLoss(qc_weight=0.0).to(device)
    opt = AdamW(model.parameters(), lr=2.5e-4)
    
    it = iter(train_loader)
    
    # Estimate event class weights from a few batches (inverse frequency)
    with torch.no_grad():
        counts = torch.zeros(4, device=device)
        seen = 0
        for _ in range(3):
            try: b = next(it)
            except StopIteration: it = iter(train_loader); b = next(it)
            vm = b["valid_mask"].to(device)
            ev = derive_event_targets_from_marks(b["targets"].to(device))  # [B,A]
            for k in range(4):
                counts[k] += ((ev==k) & vm).sum()
            seen += vm.sum()
        freq = (counts / counts.sum().clamp_min(1)).clamp_min(1e-6)
        w = (1.0 / freq); w = w / w.sum() * 4.0  # normalized inverse-frequency
        loss_fn.set_event_class_weights(w)
        print("event class weights:", w.tolist())
    ema = {}

    def upd(name, val, a=0.9): ema[name] = val if name not in ema else a*ema[name] + (1-a)*val

    t0 = time.time()
    for step in range(1, steps+1):
        try: batch = next(it)
        except StopIteration: it = iter(train_loader); batch = next(it)

        batch = {k:(v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}

        opt.zero_grad(set_to_none=True)
        preds = model(batch["temporal_sequence"], batch["action_sequence"], batch["valid_mask"])
        total, comps = loss_fn(preds, batch)
        total.backward(); opt.step()

        for k in ["event_ce","x_gaussian_nll","y_gaussian_nll","timing_pinball"]:
            upd(k, float(comps[k]))
        if step % 20 == 0:
            td_med = preds["time_delta_q"][...,1].mean().item()
            td_std = preds["time_delta_q"][...,1].std().item()
            apg = preds["sequence_length"].float()
            print(f"[{step:03d}] event:{ema['event_ce']:.3f} x:{ema['x_gaussian_nll']:.3f} y:{ema['y_gaussian_nll']:.3f} time:{ema['timing_pinball']:.3f} | dT μ={td_med:.4f}s σ={td_std:.4f}s | acts μ={apg.mean():.1f} σ={apg.std():.1f}")

    print(f"✅ Micro-fit finished in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
