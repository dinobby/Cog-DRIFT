"""Data preprocessing for adaptive curriculum training.

Each unique problem is grouped with all available variant forms
(four_choice / ten_choice / fill_in / open_ended).  Training starts at the
easiest available variant and adaptively upgrades according to reward.

Difficulty order: four_choice < ten_choice < fill_in < open_ended
"""
import json
import os
from pathlib import Path
import datasets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DIFFICULTY_ORDER = ["four_choice", "ten_choice", "fill_in", "open_ended"]
INSTRUCTION_FOLLOWING = "\n\nThink step by step and put the final answer within \\boxed{}."


def _get_base_question(q: str) -> str:
    """Strip variant-specific suffixes to recover the raw problem text.

    Handles:
    - Multiple-choice options ("\nA. ...")
    - Fill-in-the-blank marker ("\nThe answer should look like: ...")
    """
    for marker in ("\nA.", "\nThe answer should look like:", "\nFill the blank"):
        idx = q.find(marker)
        if idx != -1:
            return q[:idx].strip()
    return q.strip()


def _initial_variant(available: list[str]) -> str:
    """Return the easiest available variant."""
    for v in DIFFICULTY_ORDER:
        if v in available:
            return v
    raise ValueError(f"No known variant in {available}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    _SRC_DIR = Path(__file__).parent
    _REPO_ROOT = _SRC_DIR.parent

    data_path_oe  = _SRC_DIR / "data" / "BMH_train.json"
    data_path_fi  = _SRC_DIR / "data" / "BMH_train_fill_in.json"
    data_path_4c  = _SRC_DIR / "data" / "BMH_train_four_choice.json"
    data_path_10c = _SRC_DIR / "data" / "BMH_train_ten_choice.json"

    local_save_dir = _REPO_ROOT / "processed_data" / "BMH_Adaptive"
    os.makedirs(local_save_dir, exist_ok=True)

    data_source = "BigMathHard"

    with open(data_path_oe)  as f: data_oe  = json.load(f)
    with open(data_path_fi)  as f: data_fi  = json.load(f)
    with open(data_path_4c)  as f: data_4c  = json.load(f)
    with open(data_path_10c) as f: data_10c = json.load(f)

    # -----------------------------------------------------------------------
    # Build base_question → {variant_type: sample} lookup
    # -----------------------------------------------------------------------
    # base_map[base_q][variant_type] = raw sample dict
    base_map: dict[str, dict] = {}

    for vtype, samples in [
        ("open_ended",  data_oe),
        ("fill_in",     data_fi),
        ("four_choice", data_4c),
        ("ten_choice",  data_10c),
    ]:
        for s in samples:
            base = _get_base_question(s["question"])
            if base not in base_map:
                base_map[base] = {}
            base_map[base][vtype] = s

    print(f"Total unique base questions: {len(base_map)}")

    # -----------------------------------------------------------------------
    # Assign stable question IDs and build variant lookup JSON
    # variants_lookup[question_id] = {
    #     "available": [sorted list of variant types],
    #     "variants": {variant_type: {"prompt": [...], "gold_answer": "..."}}
    # }
    # -----------------------------------------------------------------------
    variants_lookup: dict[str, dict] = {}
    train_list: list[dict] = []

    combo_counts: dict[tuple, int] = {}

    for q_idx, (base_q, variants) in enumerate(base_map.items()):
        q_id = f"q_{q_idx}"
        available = sorted(variants.keys(), key=lambda v: DIFFICULTY_ORDER.index(v)
                           if v in DIFFICULTY_ORDER else 99)
        combo_counts[tuple(sorted(variants.keys()))] = \
            combo_counts.get(tuple(sorted(variants.keys())), 0) + 1

        # Build per-variant prompt dicts
        variant_data: dict[str, dict] = {}
        for vtype, sample in variants.items():
            q_text = sample["question"] + INSTRUCTION_FOLLOWING
            variant_data[vtype] = {
                "prompt": [{"role": "user", "content": q_text}],
                "gold_answer": sample["gold_answer"],
            }

        variants_lookup[q_id] = {
            "available": available,
            "variants": variant_data,
        }

        # Training row starts at the easiest available variant
        init_vtype = _initial_variant(available)
        init_variant = variant_data[init_vtype]

        train_list.append({
            "type":         init_vtype,   # current active variant
            "question_id":  q_id,
            "prompt":       init_variant["prompt"],
            "gold_answer":  init_variant["gold_answer"],
        })

    print("\nVariant combination counts:")
    for combo, cnt in sorted(combo_counts.items(), key=lambda x: -x[1]):
        print(f"  {combo}: {cnt}")

    # -----------------------------------------------------------------------
    # Build HuggingFace train dataset
    # -----------------------------------------------------------------------
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt       = example.pop("prompt")
            gold_answer  = example.pop("gold_answer")
            q_id         = example.pop("question_id")
            vtype        = example["type"]
            data = {
                "data_source":  data_source,
                "type":         vtype,
                "prompt":       prompt,
                "ability":      "math",
                "reward_model": {"style": "rule", "ground_truth": gold_answer},
                "extra_info":   {"split": split, "index": idx, "question_id": q_id},
            }
            return data
        return process_fn

    train_dataset = datasets.Dataset.from_list(train_list)
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    val_list = []

    math500 = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test").to_list()
    for sample in math500:
        val_list.append({
            "data_source": "MATH500",
            "type": "open_ended",
            "question": sample["problem"],
            "gold_answer": sample["answer"],
        })

    def make_val_map_fn(split):
        def process_fn(example, idx):
            question    = example.pop("question")
            gold_answer = example.pop("gold_answer")
            effective_ds = example.pop("data_source", data_source)
            data = {
                "data_source":  effective_ds,
                "type":         example["type"],
                "prompt":       [{"role": "user", "content": question + INSTRUCTION_FOLLOWING}],
                "ability":      "math",
                "reward_model": {"style": "rule", "ground_truth": gold_answer},
                "extra_info":   {"split": split, "index": idx, "question_id": ""},
            }
            return data
        return process_fn

    val_dataset = datasets.Dataset.from_list(val_list)
    val_dataset = val_dataset.map(function=make_val_map_fn("val"), with_indices=True)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    local_dir = str(local_save_dir)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))

    # Save full variants lookup (used by AdaptiveRLHFDataset at training time)
    variants_path = os.path.join(local_dir, "variants_lookup.json")
    with open(variants_path, "w", encoding="utf-8") as f:
        json.dump(variants_lookup, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(train_dataset)} training rows → {local_dir}/train.parquet")
    print(f"Saved {len(val_dataset)} validation rows → {local_dir}/val.parquet")
    print(f"Saved variants_lookup with {len(variants_lookup)} questions → {variants_path}")

    # Print starting variant distribution
    from collections import Counter
    start_types = Counter(row["type"] for row in train_list)
    print("\nStarting variant distribution in train set:")
    for vt in DIFFICULTY_ORDER:
        print(f"  {vt}: {start_types.get(vt, 0)}")

    # Save train example for inspection
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    example = val_dataset[0]
    with open(os.path.join(local_dir, "val_example.json"), "w") as f:
        json.dump(example, f, indent=2)
