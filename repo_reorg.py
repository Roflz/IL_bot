#!/usr/bin/env python3
import argparse, shutil, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Keep these folders at top-level (you said you use them)
PROTECT_DIRS = {
    "botgui", "shared_pipeline", "training_browser",
    "checkpoints", "data", "logs", "models", "quickcheck_out"
}

# Map *files* we know to their new homes
MOVE_FILES = {
    "imitation_hybrid_model.py": "ilbot/models/imitation_hybrid_model.py",
    "setup_training.py":         "ilbot/training/setup.py",
    "train_model.py":            "ilbot/training/train_loop.py",
    "inference_quickcheck.py":   "ilbot/inference/quickcheck.py",
    # Optional if present
    "evaluation_framework.py":   "ilbot/eval/metrics.py",
    "plot_mouse_overlay.py":     "ilbot/eval/viz_mouse.py",
    "action_tensor_loss.py":     "ilbot/models/losses.py",
}

# Files to create (wrappers + inits)
CREATE_FILES = {
    "apps/train.py": """from ilbot.training.train_loop import main
if __name__ == "__main__":
    main()
""",
    "apps/quickcheck.py": """from ilbot.inference.quickcheck import main
if __name__ == "__main__":
    main()
""",
    "ilbot/__init__.py": ".__version__ = '0.1.0'\n",
    "ilbot/data/__init__.py": "",
    "ilbot/models/__init__.py": "",
    "ilbot/training/__init__.py": "",
    "ilbot/inference/__init__.py": "",
    "ilbot/eval/__init__.py": "",
    "ilbot/utils/__init__.py": "",
}

PYPROJECT = """[project]
name = "ilbot"
version = "0.1.0"
description = "OSRS imitation learning bot"
requires-python = ">=3.10"
dependencies = ["torch>=2.2","numpy","tqdm","matplotlib","scikit-learn"]

[tool.setuptools]
packages = ["ilbot","apps"]
"""

GITIGNORE_ADD = """
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
*.egg-info/
.venv/
.env

# IDE/OS
.vscode/
.idea/
.DS_Store
Thumbs.db

# Artifacts / data
checkpoints/
quickcheck_out/
logs/
training_results/
data/**/*.npy
!data/sample_data/**
"""

def ensure_dirs_and_wrappers(dry):
    for rel in CREATE_FILES:
        (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
    for rel, content in CREATE_FILES.items():
        p = ROOT / rel
        if p.exists(): 
            continue
        print(f"[create] {p}")
        if not dry:
            p.write_text(content, encoding="utf-8")

def write_pyproject_and_gitignore(dry):
    pp = ROOT / "pyproject.toml"
    if not pp.exists():
        print(f"[create] {pp}")
        if not dry:
            pp.write_text(PYPROJECT, encoding="utf-8")
    gi = ROOT / ".gitignore"
    if gi.exists():
        txt = gi.read_text(encoding="utf-8")
        if GITIGNORE_ADD.strip() not in txt:
            print(f"[patch ] .gitignore (append)")
            if not dry:
                gi.write_text(txt.rstrip() + "\n\n" + GITIGNORE_ADD.strip() + "\n", encoding="utf-8")
    else:
        print(f"[create] {gi}")
        if not dry:
            gi.write_text(GITIGNORE_ADD.strip() + "\n", encoding="utf-8")

def move_known_files(dry):
    for src_rel, dst_rel in MOVE_FILES.items():
        src = ROOT / src_rel
        if not src.exists():
            continue
        dst = ROOT / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[move  ] {src} -> {dst}")
        if not dry:
            shutil.move(str(src), str(dst))

def handle_model_dir(dry):
    model_dir = ROOT / "model"
    if model_dir.exists() and model_dir.is_dir():
        # Keep only imitation_hybrid_model.py (already moved above if present)
        target_legacy = ROOT / "z_review" / "model_legacy"
        target_legacy.mkdir(parents=True, exist_ok=True)
        print(f"[stash ] model/ -> {target_legacy}")
        if not dry:
            shutil.move(str(model_dir), str(target_legacy))
        # if we moved whole dir, but need the file and it's inside legacy, copy it back into ilbot
        ihm_legacy = target_legacy / "model" / "imitation_hybrid_model.py"
        ihm_new = ROOT / "ilbot" / "models" / "imitation_hybrid_model.py"
        if ihm_legacy.exists() and not ihm_new.exists():
            print(f"[copy  ] {ihm_legacy} -> {ihm_new}")
            if not dry:
                ihm_new.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(ihm_legacy), str(ihm_new))

def stash_unused(dry):
    zreview = ROOT / "z_review"
    zreview.mkdir(exist_ok=True)
    for item in ROOT.iterdir():
        name = item.name
        if name.startswith(".") or name in {
            "ilbot","apps","z_review","repo_reorg.py","pyproject.toml",".git","README.md",".gitignore",
        } | PROTECT_DIRS | set(MOVE_FILES.keys()):
            continue
        # skip files we already moved/created
        if item.is_dir():
            dest = zreview / "misc_dirs" / name
        else:
            dest = zreview / "misc_files" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[stash ] {item} -> {dest}")
        if not dry:
            shutil.move(str(item), str(dest))
    # move tests -> z_review/tests
    tests = ROOT / "tests"
    if tests.exists():
        dest = zreview / "tests"
        print(f"[stash ] tests/ -> {dest}")
        if not dry:
            shutil.move(str(tests), str(dest))

def patch_imports_in(file_path: Path, dry: bool):
    if not file_path.exists():
        return
    txt = file_path.read_text(encoding="utf-8")
    orig = txt
    replacements = [
        (r"\bfrom\s+setup_training\s+import\b", "from ilbot.training.setup import"),
        (r"\bimport\s+setup_training\b", "from ilbot.training import setup as setup_training"),
        (r"\bfrom\s+imitation_hybrid_model\s+import\b", "from ilbot.models.imitation_hybrid_model import"),
        (r"\bfrom\s+evaluation_framework\s+import\b", "from ilbot.eval.metrics import"),
        (r"\bfrom\s+plot_mouse_overlay\s+import\b", "from ilbot.eval.viz_mouse import"),
    ]
    for pat, repl in replacements:
        txt = re.sub(pat, repl, txt)

    if txt != orig:
        print(f"[patch ] imports in {file_path}")
        if not dry:
            file_path.write_text(txt, encoding="utf-8")

def patch_moved_files(dry):
    for rel in ["ilbot/training/train_loop.py", "ilbot/inference/quickcheck.py"]:
        patch_imports_in(ROOT / rel, dry)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Perform changes")
    args = ap.parse_args()
    dry = not args.apply

    ensure_dirs_and_wrappers(dry)
    move_known_files(dry)
    handle_model_dir(dry)
    stash_unused(dry)
    patch_moved_files(dry)
    write_pyproject_and_gitignore(dry)

    print("\nDone.", "(dry-run)" if dry else "")
    print("Run training via:   python -m apps.train  --data_dir ...")
    print("Run quickcheck via: python -m apps.quickcheck --data_dir ...")

if __name__ == "__main__":
    main()
