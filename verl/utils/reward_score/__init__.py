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
# from . import gsm8k, math, prime_math, prime_code
from cgitb import reset
import re

from verl.utils.import_utils import deprecated


def _compute_format_reward(solution_str):
    """
    Compute format reward based on boxed{} usage.
    
    Rules:
    1. solution_str should contain exactly one \boxed{} or boxed{}
    2. After the boxed{}, there should be no more than 10 characters
    
    Returns:
        float: 0.2 if format is correct, 0.0 otherwise
    """
    if not solution_str or not isinstance(solution_str, str):
        return 0.0
    
    # Pattern to match \boxed{...} or boxed{...}
    # This pattern matches the entire boxed{...} including nested braces
    pattern = r'\\?boxed\{'
    
    # Find all occurrences of boxed{
    matches = list(re.finditer(pattern, solution_str, re.IGNORECASE))
    
    # Rule 1: Must have exactly one boxed{}
    if len(matches) != 1:
        return 0.0
    
    # Find the position after the closing brace of boxed{}
    match = matches[0]
    start_pos = match.end()  # Position right after "boxed{"
    
    # Find the matching closing brace
    brace_count = 1
    pos = start_pos
    while pos < len(solution_str) and brace_count > 0:
        if solution_str[pos] == '{':
            brace_count += 1
        elif solution_str[pos] == '}':
            brace_count -= 1
        pos += 1
    
    # If braces don't match properly, format is invalid
    if brace_count != 0:
        return 0.0
    
    # pos is now right after the closing brace of boxed{}
    # Rule 2: Check if there are more than 10 characters after boxed{}
    remaining_text = solution_str[pos:].strip()
    
    if len(remaining_text) > 10:
        return 0.0
    
    return 0.2


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    format_reward: bool = False,
    format_reward_coef: float = 0.2,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)

    elif data_source in ["BigMathHard", "BMH_test", "MATH500", "AIME2024", "AIME2025", "OmniMATH", "HMMT", "GPQA", "Date"]:
        from math_verify import parse, verify
        from . import math as math_utils

        pred = None
        try:
            string_in_last_boxed = math_utils.last_boxed_only_string(solution_str)
            if string_in_last_boxed is not None:
                pred = math_utils.remove_boxed(string_in_last_boxed)
        except (AssertionError, Exception):
            pred = None

        try:
            correctness_score = 1.0 if (pred is not None and verify(parse(ground_truth), parse(pred))) else 0.0
        except Exception:
            correctness_score = 0.0

        if format_reward:
            fmt_score = format_reward_coef if _compute_format_reward(solution_str) > 0 else 0.0
            res = {
                "correctness_score": float(correctness_score),
                "format_score": float(fmt_score),
                "score": float(correctness_score) + float(fmt_score),
            }
        else:
            res = correctness_score
        

    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
