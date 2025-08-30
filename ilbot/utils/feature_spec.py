from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, List, Tuple

from torch import special

def _read_json_required(p: Path, what: str) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"{what} not found at: {p}")
    return json.loads(p.read_text())

def load_feature_spec(data_dir: Path) -> dict:
    """
    Returns a spec dict with:
      - group_indices: {group_name: [int indices]}
      - cat_fields: [{"name": str, "indices": [int], "vocab_size": int}]
      - total_cat_vocab: int
      - cat_offsets: [int]  # length = number of categorical columns, prefix-sum offsets into a shared embedding table
      - unknown_index_per_field: [int]  # unknown id per field (always last bin)
    Priority:
      1) feature_mappings.json (authoritative layout & types)
      2) dataset_manifest.json (fallback)
      3) last-resort: assume everything non-negative & large is categorical (not ideal)
    """
    data_dir = Path(data_dir)
    mappings_dir = (data_dir.parent / "05_mappings")
    fmap_path = mappings_dir / "feature_mappings.json"
    idmap_path = mappings_dir / "id_mappings.json"
    man_path   = data_dir / "dataset_manifest.json"

    # Read exactly from the fixed locations
    fmap = _read_json_required(fmap_path, "feature_mappings.json")
    man  = _read_json_required(man_path, "dataset_manifest.json")
    idmaps = json.loads(idmap_path.read_text()) if idmap_path.exists() else {}

    # 1) Build index groups:
    #    Prefer mappings["groups"]["indices"]; else manifest["feature_groups"]; else derive from per-column metadata.
    if "groups" in fmap and "indices" in fmap["groups"]:
        groups = fmap["groups"]["indices"]
        group_indices = {k: list(map(int, v)) for k, v in groups.items()}
    else:
        fg = man.get("feature_groups", {})
        group_indices = {k: list(map(int, v)) for k, v in fg.items()} if fg else {}
        if not group_indices and isinstance(fmap, list) and fmap and "feature_index" in fmap[0]:
            gi = {"categorical": [], "continuous": [], "boolean": [], "counts": [], "angles": [], "time": []}
            def add(k, i): gi[k].append(int(i))
            for col in fmap:
                i  = int(col["feature_index"])
                dt = str(col.get("data_type","")).lower()
                nm = str(col.get("feature_name","")).lower()
                if dt in {"boolean","bool"}:
                    add("boolean", i)
                elif "angle" in dt or "angle" in nm:
                    add("angles", i)
                elif "time" in dt or "timestamp" in nm:
                    add("time", i)
                elif dt in {"count","skill_level","skill_xp"} or "count" in nm or nm.endswith("_xp"):
                    add("counts", i)
                elif dt.endswith("_id") or dt in {"item_id","object_id","npc_id","tab_id","animation_id","key_id","hashed_string","phase_type"} or nm.endswith("_id") or nm in {"action_type","item_name","target","phase_type"}:
                    add("categorical", i)
                else:
                    add("continuous", i)
            group_indices = gi

    # Sanity: require at least one non-empty group; otherwise the encoder will build no chunks.
    if not any(len(group_indices.get(k, [])) for k in ("categorical","continuous","boolean","counts","angles","time")):
        raise ValueError(
            "feature_spec: 'group_indices' is empty. "
            "Populate ../05_mappings/feature_mappings.json['groups']['indices'] or "
            "dataset_manifest.json['feature_groups'], or ensure per-column metadata exists."
        )

    # 2) Categorical fields & vocab sizes
    cat_fields: List[Dict] = []
    if isinstance(fmap, dict) and "categorical_fields" in fmap:
        src = fmap["categorical_fields"]
        for fld in src:
            name = fld["name"]; idxs = list(map(int, fld["indices"]))
            vsz = int(fld.get("vocab_size", 0))
            if vsz <= 0 and name in idmaps:
                vsz = int(len(idmaps[name])) + 1
            cat_fields.append({"name": name, "indices": idxs, "vocab_size": vsz})
    else:
        # Derive per-column vocab using your id_mappings keys (exact names from your JSONs)
        #   Inventory: "Inventory.item_ids"
        #   Game Objects: "Game Objects.object_ids"
        #   NPCs: "NPCs.npc_ids"
        #   Tabs: "Tabs.tab_ids"
        #   Player: "Player.player_animation_ids", "Player.player_movement_direction_hashes"
        #   Interaction: "Interaction.action_type_hashes", "Interaction.item_name_hashes", "Interaction.target_hashes"
        #   Phase: "Phase Context.phase_type_hashes"
        name_for = []
        for col in fmap if isinstance(fmap, list) else []:
            i  = int(col["feature_index"])
            dt = str(col.get("data_type","")).lower()
            nm = str(col.get("feature_name","")).lower()
            name = None
            if dt == "slot_id":
                name = "Bank.slot_ids"
            elif dt == "item_id":
                name = "Inventory.item_ids"
            elif dt == "object_id":
                name = "Game Objects.object_ids"
            elif dt == "npc_id":
                name = "NPCs.npc_ids"
            elif dt == "tab_id":
                name = "Tabs.tab_ids"
            elif dt == "animation_id":
                name = "Player.player_animation_ids"
            elif nm == "player_movement_direction":
                name = "Player.player_movement_direction_hashes"
            elif nm in {"action_type","item_name","target"}:
                name = f"Interaction.{nm}_hashes"
            elif nm == "phase_type":
                name = "Phase Context.phase_type_hashes"
            # Build cat_fields entry per categorical column
            if name:
                vocab = idmaps
                for part in name.split("."):
                    vocab = vocab.get(part, {})
                vsz = int(len(vocab)) + 1 if isinstance(vocab, dict) else 0
                cat_fields.append({"name": name, "indices": [i], "vocab_size": vsz})

    # 3) Offsets for a shared embedding table (sum of field vocabs)
    total_cat_vocab = sum(f["vocab_size"] for f in cat_fields) if cat_fields else 0
    cat_offsets = []
    running = 0
    for f in cat_fields:
        for _ in f["indices"]:
            cat_offsets.append(running)
        running += f["vocab_size"]
    unknown_index_per_field = [f["vocab_size"] - 1 for f in cat_fields for _ in f["indices"]]

    spec = {
        "group_indices": group_indices,
        "cat_fields": cat_fields,
        "total_cat_vocab": total_cat_vocab,
        "cat_offsets": cat_offsets,
        "unknown_index_per_field": unknown_index_per_field,
    }
        # --- Debug summary: show how each feature was categorized (with vocab size) ---
    # try:
    #     # Map feature_index -> (feature_name, data_type) from your mappings list
    #     name_by_idx = {}
    #     if isinstance(fmap, list):
    #         for col in fmap:
    #             idx = int(col.get("feature_index"))
    #             nm  = str(col.get("feature_name", ""))
    #             dt  = str(col.get("data_type", ""))
    #             name_by_idx[idx] = (nm, dt)

    #     # Build: categorical index -> vocab size (from cat_fields)
    #     cat_vocab_by_idx = {}
    #     for f in cat_fields:
    #         vsz = int(f.get("vocab_size", 0))
    #         for i in f["indices"]:
    #             cat_vocab_by_idx[int(i)] = vsz

    #     # group -> indices  ==>  index -> group
    #     group_by_idx = {}
    #     for g, idxs in (group_indices or {}).items():
    #         for i in idxs:
    #             group_by_idx[int(i)] = g

    #     # Union of anything we know about (names or groups)
    #     all_idx = sorted(set(name_by_idx.keys()) | set(group_by_idx.keys()))

    #     print("=== Feature grouping summary ===")
    #     print("idx\tgroup\tname\tdata_type\tcat_vocab\tvocab_size")
    #     for i in all_idx:
    #         g  = group_by_idx.get(i, "UNASSIGNED")
    #         nm, dt = name_by_idx.get(i, ("", ""))
    #         if g == "categorical":
    #             vsz = cat_vocab_by_idx.get(i, 0)
    #             has_vocab = "yes" if i in cat_vocab_by_idx else "no"
    #             print(f"{i}\t{g}\t{nm}\t{dt}\t{has_vocab}\t{vsz}")
    #         else:
    #             print(f"{i}\t{g}\t{nm}\t{dt}\t\t")
    #     print("=== End feature summary ===")
    # except Exception as e:
    #     print(f"[feature_spec] summary print failed: {e}")

    return spec

