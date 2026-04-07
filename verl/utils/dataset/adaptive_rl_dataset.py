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
"""Adaptive RLHF dataset: upgrades questions through difficulty levels at runtime."""

import json
import logging
from collections import defaultdict
from typing import Optional

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)

# Difficulty ordering shared between dataset and sampler
DIFFICULTY_ORDER = ["four_choice", "ten_choice", "fill_in", "open_ended"]


class AdaptiveRLHFDataset(RLHFDataset):
    """RLHF dataset that supports multi-level adaptive curriculum learning.

    Each question starts at its easiest available variant and can be upgraded
    through harder variants by calling ``upgrade_item()``.

    Difficulty order: four_choice < ten_choice < fill_in < open_ended

    Config keys (under ``data.``):
        variants_lookup (str): Path to ``variants_lookup.json`` produced by the
            preprocessing script.  Required when ``data.adaptive=True``.
    """

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(data_files, tokenizer, config, processor)

        # ------------------------------------------------------------------
        # Load variants lookup
        # variants_lookup[question_id] = {
        #     "available": [sorted-by-difficulty variant types],
        #     "variants": { variant_type: {"prompt": [...], "gold_answer": "..."} }
        # }
        # ------------------------------------------------------------------
        lookup_path = config.get("variants_lookup", None)
        self._variants_lookup: dict[str, dict] = {}
        if lookup_path:
            with open(lookup_path, "r", encoding="utf-8") as f:
                self._variants_lookup = json.load(f)
            logger.info(
                f"[AdaptiveRLHFDataset] Loaded variants_lookup with "
                f"{len(self._variants_lookup)} questions from {lookup_path}"
            )
        else:
            logger.warning(
                "[AdaptiveRLHFDataset] 'variants_lookup' not set in config; "
                "adaptive upgrades are disabled."
            )

        # ------------------------------------------------------------------
        # Runtime state — must be initialized before _build_qid_index()
        # ------------------------------------------------------------------
        # df_idx → {"prompt": [...], "reward_model": {...}, "type": "..."}
        self._overrides: dict[int, dict] = {}
        # question_id → currently active variant type (pre-populated by _build_qid_index)
        self._current_variant: dict[str, str] = {}

        # ------------------------------------------------------------------
        # Build question_id → (post-filter) dataframe index
        # ------------------------------------------------------------------
        self._qid_to_df_idx: dict[str, int] = {}
        self._build_qid_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_qid_index(self) -> None:
        """Build question_id → df_idx and pre-populate _current_variant."""
        types = self.dataframe["type"]
        for i, (ei, row_type) in enumerate(zip(self.dataframe["extra_info"], types)):
            if isinstance(ei, dict):
                qid = ei.get("question_id", "")
                if qid:
                    self._qid_to_df_idx[qid] = i
                    self._current_variant[qid] = row_type  # initial variant from parquet
        logger.info(
            f"[AdaptiveRLHFDataset] Indexed {len(self._qid_to_df_idx)} "
            "questions in the (filtered) dataset."
        )

    def _get_current_variant(self, question_id: str) -> str:
        """Return the currently active variant for a question_id."""
        return self._current_variant.get(question_id, "open_ended")

    # ------------------------------------------------------------------
    # RLHFDataset override
    # ------------------------------------------------------------------

    def _get_row(self, item: int) -> dict:
        if item in self._overrides:
            merged = dict(self.dataframe[item])
            merged.update(self._overrides[item])
            return merged
        return self.dataframe[item]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upgrade_item(self, question_id: str) -> str | None:
        """Upgrade a question to the next harder available variant.

        Returns:
            The new variant type string if an upgrade was applied,
            ``None`` if the question is already at its hardest available
            variant or is not found in the dataset.
        """
        if question_id not in self._variants_lookup:
            return None
        if question_id not in self._qid_to_df_idx:
            return None  # filtered out

        info = self._variants_lookup[question_id]
        available: list[str] = info["available"]
        current = self._get_current_variant(question_id)

        # Find the next harder variant that is available
        try:
            cur_rank = DIFFICULTY_ORDER.index(current)
        except ValueError:
            cur_rank = -1

        next_variant: str | None = None
        for v in DIFFICULTY_ORDER[cur_rank + 1:]:
            if v in available:
                next_variant = v
                break

        if next_variant is None:
            return None  # already at max difficulty

        df_idx = self._qid_to_df_idx[question_id]
        variant_data = info["variants"][next_variant]

        self._overrides[df_idx] = {
            self.prompt_key: variant_data["prompt"],
            "reward_model": {"style": "rule", "ground_truth": variant_data["gold_answer"]},
            "type": next_variant,
        }
        self._current_variant[question_id] = next_variant
        return next_variant

    def get_variant_counts(self) -> dict[str, int]:
        """Return a count of questions currently at each variant level."""
        counts: dict[str, int] = defaultdict(int)
        for qid in self._qid_to_df_idx:
            counts[self._get_current_variant(qid)] += 1
        return dict(counts)
