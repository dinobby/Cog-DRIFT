## [Cog-DRIFT: Exploration on Adaptively Reformulated Instances Enables Learning from Hard Reasoning Problems](https://arxiv.org/abs/2604.04767)

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2604.04767)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/dinobby/Cog-DRIFT-Dataset)

[Justin Chih-Yao Chen](https://dinobby.github.io/), [Archiki Prasad](https://archiki.github.io/), [Zaid Khan](https://zaidkhan.me/), [Joykirat Singh](https://joykirat18.github.io/), [Runchu Tian](https://rachum-thu.github.io/), [Elias Stengel-Eskin](https://esteng.github.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)


### Overview

![Overview of Cog-DRIFT](https://i.imgur.com/7vfhnQK.png)

This repository contains the implementation of Cog-DRIFT, a RL framework that reformulates hard problems into easier, structured variants (MCQ and cloze), then curriculum-train models from easy → hard to unlock new learning signals. This project is built on top of [verl](https://github.com/volcengine/verl). The pipeline has three main components:

1. **Identify hard problems** (`src/get_hard_samples.py`) — Run pass@k rollouts on a base model to collect problems it cannot solve.
2. **Generate question variants** (`src/rewrite_questions.py`) — Rewrite open-ended problems into multiple-choice and fill-in-the-blank formats.
3. **Train with adaptive curriculum** (`start_qwen4b.sh` / `start_llama3b.sh`) — The trainer starts each problem at its easiest available format and upgrades the format as the model improves.


### Installation

```bash
conda create -n verl python=3.10.12
conda activate verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
pip install math-verify[antlr4_9_3]
```


### Data Preparation

We have the preprocessed parquet data in `./processsed_data`, as well as the intermediate data on [HuggingFace](https://huggingface.co/datasets/dinobby/Cog-DRIFT-Dataset), so you can skip this section and directly start training. If you want to reconstruct the exact variant files used in the paper, please follow the steps below.

#### Step 1 — Find hard samples (pass@k = 0)

```bash
python src/get_hard_samples.py \
    --model Qwen/Qwen3-4B-Instruct \
    --dataset BMH \
    --k 64 \
    --gpus 0,1,2,3
```

This saves a state file and a final `pass_at_64_equals_0.json` under `./data`.

Then you can filter noisy data using GPT:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_KEY="your-api-key"

python src/filter_questions.py \
    --input ./data/BMH_pass_at_64_equals_0.json
```

This will produce a `BMH_full.json`, which can be split into train and test split via:
```bash
python src/split_data.py
```
Now you will get `src/data/BMH_train.json` and `src/data/BMH_test.json`.

#### Step 2 — Generate question variants

```bash
# Four-choice MC
python src/rewrite_questions.py \
    --mode four_choice \
    --input src/data/BMH_train.json \
    --tensor_parallel_size 4

# Ten-choice MC
python src/rewrite_questions.py \
    --mode ten_choice \
    --input src/data/BMH_train.json \
    --tensor_parallel_size 4

# Fill-in-the-blank
python src/rewrite_questions.py \
    --mode fill_in \
    --input src/data/BMH_train.json \
    --tensor_parallel_size 4
```

Output files are written alongside the input (`BMH_train_four_choice.json`, etc.).

#### Step 3 — Preprocess into training parquet

```bash
python src/preprocess_data.py
# Outputs: processed_data/BMH_Adaptive/{train.parquet, val.parquet, variants_lookup.json}
```


### Training

#### Qwen3-4B

```bash
bash start_qwen4b.sh
```

#### Llama-3.2-3B

```bash
bash start_llama3b.sh
```

Both scripts auto-detect the repo root and read `HF_HOME` from the environment (defaulting to `~/.cache/huggingface`). Override any Hydra config key by appending it:

```bash
bash start_qwen4b.sh trainer.total_epochs=50 data.train_batch_size=16
```

To use [W&B](https://wandb.ai) logging, run `wandb login` before training. To disable it, change `trainer.logger=['console']` in the script.


### Evaluation

```bash
python src/eval.py \
    --model Qwen/Qwen3-4B-Instruct \
    --dataset BMH_test \
    --k 3 \
    --gpus 0,1,2,3
```

Supported datasets: `BMH_test`, `AIME2024`, `AIME2025`, `GPQA`, `OmniMATH`, `Date`.

Add `--save_output` to save all results to `outputs/`.


### Project Structure

```
Cog-DRIFT/
├── src/
│   ├── split_data.py          # Train/test split for BMH_full.json
│   ├── get_hard_samples.py    # Pass@k rollout to find unsolvable problems
│   ├── filter_questions.py    # Filter noisy data using GPT
│   ├── rewrite_questions.py   # Generate MC / fill-in-the-blank variants
│   ├── preprocess_data.py     # Build adaptive curriculum parquet
│   ├── eval.py                # Evaluation on standard math benchmarks
│   └── utils.py               # Math equivalence utilities
├── verl/                      # verl training framework (modified)
│   ├── utils/dataset/
│   │   └── adaptive_rl_dataset.py   # Adaptive dataset with per-question state
│   └── experimental/dataset/
│       └── adaptive_sampler.py      # Curriculum sampler logic
├── start_qwen4b.sh            # Launch script for Qwen3-4B
├── start_llama3b.sh           # Launch script for Llama-3.2-3B
└── processed_data/            # Generated by preprocess_data.py
```


### Citation

If you find this work useful, please cite:

```bibtex
@article{chen2026cogdrift,
  title   = {Cog-DRIFT: Exploration on Adaptively Reformulated Instances Enables Learning from Hard Reasoning Problems},
  author  = {Chen, Justin Chih-Yao and Prasad, Archiki and Khan, Zaid and Singh, Joykirat and Tian, Runchu and Stengel-Eskin, Elias and Bansal, Mohit},
  year    = {2026},
  journal={arXiv preprint arXiv:2604.04767},
}
```
