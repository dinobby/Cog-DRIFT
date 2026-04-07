import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import last_boxed_only_string, remove_boxed, is_math_equiv

DEPLOYMENT = "gpt-5.3-chat"
API_VERSION = "2024-12-01-preview"
ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
SUBSCRIPTION_KEY = os.environ.get("AZURE_OPENAI_KEY", "")

NUM_VOTES = 3
MAX_CONCURRENT = 30
SAVE_EVERY = 10      # save progress after every N completed samples

client = AsyncAzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=SUBSCRIPTION_KEY,
)


def extract_boxed(text: str) -> str:
    return remove_boxed(last_boxed_only_string(text))


async def get_answer(question: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        resp = await client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert math solver. "
                        "Solve the problem step by step and put your final answer in \\boxed{}."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_completion_tokens=20000,
        )
    return resp.choices[0].message.content


async def process_sample(
    sample: dict, semaphore: asyncio.Semaphore
) -> tuple[dict, bool]:
    """
    Returns (sample_with_metadata, is_valid).
    is_valid = majority-voted GPT answer matches gold_answer.
    """
    question = sample["question"]
    gold = sample["gold_answer"]

    # Fetch NUM_VOTES answers concurrently
    raw_answers = await asyncio.gather(
        *[get_answer(question, semaphore) for _ in range(NUM_VOTES)],
        return_exceptions=True,
    )

    extracted = []
    for ans in raw_answers:
        if isinstance(ans, Exception):
            print(f"\n[WARN] answer error for id={sample['id']}: {ans}")
        else:
            extracted.append(extract_boxed(ans))

    if not extracted:
        return sample, False

    majority_answer, majority_count = Counter(extracted).most_common(1)[0]

    try:
        is_valid = is_math_equiv(gold, majority_answer)
    except Exception:
        is_valid = False
    print(f"majority_answer: {majority_answer}, gold: {gold} is_valid: {is_valid}")
    enriched = {
        **sample,
        "gpt_answers": extracted,
        "gpt_majority_answer": majority_answer,
        "gpt_majority_count": majority_count,
    }
    return enriched, is_valid


async def main(data_path: Path, output_path: Path, progress_path: Path):
    with open(data_path) as f:
        data: list[dict] = json.load(f)

    total = len(data)
    print(f"Dataset loaded: {total} samples")

    # ── Resume from checkpoint ──────────────────────────────────────────────
    processed_ids: set[int] = set()
    valid_samples: list[dict] = []

    if os.path.exists(progress_path):
        with open(progress_path) as f:
            ckpt = json.load(f)
        processed_ids = set(ckpt["processed_ids"])
        print(f"Checkpoint found: {len(processed_ids)} samples already processed.")

    if os.path.exists(output_path):
        with open(output_path) as f:
            valid_samples = json.load(f)
        print(f"Loaded {len(valid_samples)} previously validated samples.")

    remaining = [s for s in data if s["id"] not in processed_ids]
    print(f"Samples remaining: {len(remaining)}\n")

    if not remaining:
        print("Nothing left to process.")
    else:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        lock = asyncio.Lock()

        completed = 0
        newly_valid = 0

        async def handle(sample: dict, pbar: tqdm):
            nonlocal completed, newly_valid
            enriched, is_valid = await process_sample(sample, semaphore)
            async with lock:
                processed_ids.add(sample["id"])
                if is_valid:
                    valid_samples.append(enriched)
                    newly_valid += 1
                completed += 1
                pbar.update(1)
                pbar.set_postfix(valid=len(valid_samples), filtered=completed - newly_valid)

                # Periodic checkpoint save
                if completed % SAVE_EVERY == 0:
                    _save(valid_samples, processed_ids, output_path, progress_path)

        with tqdm(total=len(remaining), desc="Processing") as pbar:
            await asyncio.gather(*[handle(s, pbar) for s in remaining])

        _save(valid_samples, processed_ids, output_path, progress_path)

    # ── Final statistics ────────────────────────────────────────────────────
    valid = len(valid_samples)
    filtered_out = total - valid
    print("\n" + "=" * 50)
    print("FINAL STATISTICS")
    print("=" * 50)
    print(f"  Total samples          : {total}")
    print(f"  Processed this run     : {len(processed_ids)}")
    print(f"  Valid (kept)           : {valid}  ({valid / total * 100:.1f}%)")
    print(f"  Filtered out (noisy)   : {filtered_out}  ({filtered_out / total * 100:.1f}%)")
    print(f"  Output file            : {output_path}")
    print("=" * 50)


def _save(valid_samples: list[dict], processed_ids: set[int],
          output_path: Path, progress_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid_samples, f, ensure_ascii=False, indent=2)
    with open(progress_path, "w") as f:
        json.dump({"processed_ids": list(processed_ids)}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter hard samples using Azure OpenAI majority-vote verification."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to pass_at_k_equals_0.json produced by get_hard_samples.py",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for validated samples (default: <input_dir>/valid_samples.json)",
    )
    parser.add_argument(
        "--progress", type=Path, default=None,
        help="Checkpoint file path (default: <input_dir>/.filter_progress.json)",
    )
    args = parser.parse_args()

    input_dir = args.input.parent
    output_path = args.output or (input_dir / "BMH_full.json")
    progress_path = args.progress or (input_dir / ".filter_progress.json")

    asyncio.run(main(args.input, output_path, progress_path))
