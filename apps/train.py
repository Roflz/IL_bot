# apps/train.py
from ilbot.config import Config
from ilbot.data.dataset import make_loaders

def main():
    cfg = Config(data_dir="data/recording_sessions/20250831_113719/06_final_training_data")
    train_loader, val_loader, dc = make_loaders(cfg)
    print("Data ready. Shapes:")
    b = next(iter(train_loader))
    for k,v in b.items():
        print(k, tuple(v.shape))
    print("data_config:", dc)
    print("✅ Phase 0/1 complete")

if __name__ == "__main__":
    main()
