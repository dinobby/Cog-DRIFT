# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adaptive curriculum sampler: upgrades questions through difficulty levels
when the model's rolling-average reward exceeds a threshold."""

import logging
from collections import defaultdict
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import RandomSampler

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler

# DIFFICULTY_ORDER is used for metric keys; import as constant to avoid
# coupling to the class identity (load_extern_type creates a separate module
# object, so isinstance checks against the package-imported class would fail).
DIFFICULTY_ORDER = ["four_choice", "ten_choice", "fill_in", "open_ended"]

logger = logging.getLogger(__name__)


def _is_adaptive_dataset(dataset) -> bool:
    """Duck-type check: does this dataset support adaptive curriculum?"""
    return hasattr(dataset, "upgrade_item") and hasattr(dataset, "get_variant_counts")


class AdaptiveCurriculumSampler(AbstractCurriculumSampler):
    """Curriculum sampler that upgrades questions through multiple difficulty levels.

    Difficulty order: four_choice < ten_choice < fill_in < open_ended

    After each training step the sampler:
    1. Computes the mean reward over all rollouts for every question that
       appeared in the batch.
    2. Maintains a rolling window of per-question batch-mean rewards.
    3. When the rolling mean exceeds ``adaptive_threshold`` the question is
       permanently upgraded to the next harder available variant.
    4. Returns a rich metrics dict that is logged to wandb **every step**,
       regardless of whether any upgrade happened.

    Logged metrics (all under ``adaptive/``):
        n_upgraded_this_step          – upgrades performed in this step
        total_upgraded                – cumulative upgrades across all steps
        upgraded_to_{type}            – cumulative upgrades that landed on *type*
        dataset_count/{type}          – total questions currently at each level
        batch_count/{type}            – questions of each type that appeared
                                        in this batch
        reward/{type}/mean|max|min    – reward statistics per type in this batch

    Config keys (all under ``data.``):
        adaptive_threshold (float): Reward threshold for upgrade. Default 0.6.
        adaptive_reward_window (int): Rolling window size (in batches).
            Default 1 (upgrade as soon as current-batch mean > threshold).
        seed (int): Random seed. Default 1.
    """

    def __init__(self, data_source, data_config: DictConfig):
        self._dataset = data_source
        self._threshold: float = data_config.get("adaptive_threshold", 0.5)
        self._window: int = max(1, data_config.get("adaptive_reward_window", 1))

        if not _is_adaptive_dataset(data_source):
            logger.warning(
                "[AdaptiveCurriculumSampler] data_source does not expose "
                "upgrade_item / get_variant_counts – upgrade logic disabled."
            )

        seed = data_config.get("seed", 1)
        generator = torch.Generator()
        generator.manual_seed(seed)
        self._random_sampler = RandomSampler(data_source=data_source, generator=generator)

        # question_id → rolling list of per-batch mean rewards (length ≤ _window)
        self._reward_history: dict[str, list[float]] = defaultdict(list)

        # Cumulative counters for wandb
        self._total_upgraded: int = 0
        self._upgraded_to: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------

    def __iter__(self):
        return iter(self._random_sampler)

    def __len__(self) -> int:
        return len(self._dataset)

    # ------------------------------------------------------------------
    # AbstractCurriculumSampler interface
    # ------------------------------------------------------------------

    def update(self, batch: DataProto) -> Optional[dict]:
        """Process the completed batch, trigger upgrades, and return metrics."""

        # ---- gather per-item scores, types, question_ids ------------------
        if "token_level_scores" not in batch.batch:
            return None

        seq_scores = batch.batch["token_level_scores"].sum(dim=-1).tolist()
        types = batch.non_tensor_batch.get("type", None)
        extra_infos = batch.non_tensor_batch.get("extra_info", None)

        if types is None or extra_infos is None:
            return None

        # ---- accumulate per-question scores (four_choice only upgrades) ---
        # We group by question_id for all types so we can report batch rewards,
        # but only attempt upgrades when the current type for that question
        # matches what we see in the batch.
        qid_scores: dict[str, list[float]] = defaultdict(list)
        type_scores: dict[str, list[float]] = defaultdict(list)

        for score, t, ei in zip(seq_scores, types, extra_infos):
            t = str(t)
            type_scores[t].append(float(score))
            if isinstance(ei, dict):
                qid = ei.get("question_id", "")
                if qid:
                    qid_scores[qid].append(float(score))

        # ---- rolling-window upgrade logic ---------------------------------
        n_upgraded_this_step = 0
        if _is_adaptive_dataset(self._dataset):
            for qid, scores in qid_scores.items():
                batch_mean = sum(scores) / len(scores)
                history = self._reward_history[qid]
                history.append(batch_mean)
                if len(history) > self._window:
                    self._reward_history[qid] = history[-self._window:]

                rolling_mean = sum(self._reward_history[qid]) / len(self._reward_history[qid])
                if rolling_mean > self._threshold:
                    new_variant = self._dataset.upgrade_item(qid)
                    if new_variant is not None:
                        n_upgraded_this_step += 1
                        self._total_upgraded += 1
                        self._upgraded_to[new_variant] += 1
                        # Reset history so we don't immediately upgrade again
                        self._reward_history[qid] = []
                        logger.info(
                            f"[Adaptive] Upgraded {qid} → {new_variant} "
                            f"(rolling_mean={rolling_mean:.3f})"
                        )

        # ---- build metrics dict (logged EVERY step) -----------------------
        metrics: dict = {
            "adaptive/n_upgraded_this_step": n_upgraded_this_step,
            "adaptive/total_upgraded":       self._total_upgraded,
        }

        # Cumulative count of upgrades per destination level
        for vt in DIFFICULTY_ORDER:
            metrics[f"adaptive/upgraded_to/{vt}"] = self._upgraded_to.get(vt, 0)

        # Dataset-level variant distribution (how many questions at each level)
        if _is_adaptive_dataset(self._dataset):
            variant_counts = self._dataset.get_variant_counts()
            for vt in DIFFICULTY_ORDER:
                metrics[f"adaptive/dataset_count/{vt}"] = variant_counts.get(vt, 0)

        # Batch-level variant distribution and reward stats
        for vt in DIFFICULTY_ORDER:
            scores_for_type = type_scores.get(vt, [])
            metrics[f"adaptive/batch_count/{vt}"] = len(scores_for_type)
            if scores_for_type:
                metrics[f"adaptive/reward/{vt}/mean"] = sum(scores_for_type) / len(scores_for_type)
                metrics[f"adaptive/reward/{vt}/max"]  = max(scores_for_type)
                metrics[f"adaptive/reward/{vt}/min"]  = min(scores_for_type)

        return metrics
