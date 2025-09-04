# apps/train.py
import argparse
import os
from ilbot.config import Config
from ilbot.training.train_loop import run_training

def main():
    parser = argparse.ArgumentParser(description="Train imitation learning model")
    parser.add_argument("--data-dir", type=str, 
                       default="data/recording_sessions/20250831_113719/06_final_training_data",
                       help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--run-name", type=str, default=None, help="Run name (default: auto timestamp)")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable AMP")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable AMP")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--event-weighting", choices=["none","inverse","inv_sqrt"], default="none")
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--xy-weight", type=float, default=0.1)
    parser.add_argument("--logsig-min", type=float, default=-3.0)
    parser.add_argument("--logsig-max", type=float, default=0.0)
    
    args = parser.parse_args()
    
    # Build config
    cfg = Config(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        run_name=args.run_name,
        amp=args.amp,
        seed=args.seed,
        patience=args.patience,
        event_weighting=args.event_weighting,
        focal_gamma=args.focal_gamma,
        xy_weight=args.xy_weight,
        logsig_min=args.logsig_min,
        logsig_max=args.logsig_max,
    )
    
    print(f"Training configuration:")
    print(f"  Data dir: {cfg.data_dir}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.lr}")
    print(f"  AMP: {cfg.amp}")
    print(f"  Grad clip: {cfg.grad_clip}")
    print(f"  Seed: {cfg.seed}")
    print(f"  Patience: {cfg.patience}")
    print(f"\nEvent weighting: {cfg.event_weighting} | Focal gamma: {cfg.focal_gamma}")
    print(f"XY weight: {cfg.xy_weight}")
    print(f"XY logσ clamp: [{cfg.logsig_min}, {cfg.logsig_max}]")
    print()
    
    # Run training
    summary = run_training(cfg)
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print("="*60)

if __name__ == "__main__":
    main()
