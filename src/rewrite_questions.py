import json
import os
import argparse
import random
import re
from collections import defaultdict
from utils import is_math_equiv, last_boxed_only_string, remove_boxed
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


def normalize_answer(answer):
    """Normalize answer for comparison."""
    answer = str(answer).strip()
    answer = re.sub(r'\s+', ' ', answer)
    answer = answer.replace('\\text{', '').replace('}', '')
    answer = answer.replace('^{', '^').replace('_{', '_')
    return answer.lower()


def strip_latex_math(s):
    """Strip $/$ math delimiters and expand \\text{content} to content before comparison."""
    s = str(s).strip()
    s = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', s)
    s = re.sub(r'^\$+|\$+$', '', s).strip()
    return s


def validate_multiple_choice(result, original_answer, num_options):
    """Validate a generated multiple-choice question."""
    question = result['question']
    correct_choice = result.get('correct_choice', result.get('gold_answer'))
    original_normalized = normalize_answer(original_answer)

    validation = {
        'passed': False,
        'errors': []
    }

    expected_options = (['A', 'B', 'C', 'D'] if num_options == 4
                       else ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
                       if num_options == 10
                       else ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                             'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'])

    options = {}
    for line in question.split('\n'):
        line = line.strip()
        for opt in expected_options:
            if line.startswith(f"{opt}."):
                options[opt] = line.replace(f"{opt}.", "").strip()
                break

    if len(options) != num_options:
        validation['errors'].append(f"Expected {num_options} options but found {len(options)}")

    if correct_choice not in expected_options:
        validation['errors'].append(f"Invalid correct_choice '{correct_choice}', expected one of {expected_options}")
    elif correct_choice not in options:
        validation['errors'].append(f"Correct choice option '{correct_choice}' not found in parsed options")
    else:
        correct_option_text = options[correct_choice]
        correct_stripped = strip_latex_math(correct_option_text)
        original_stripped = strip_latex_math(original_answer)
        norm_correct = normalize_answer(correct_stripped)
        norm_original = normalize_answer(original_stripped)
        is_equiv = (norm_correct == norm_original) or bool(is_math_equiv(correct_stripped, original_stripped))
        if not is_equiv:
            correct_option_normalized = normalize_answer(correct_option_text)
            original_parts = re.findall(r'\d+\.?\d*|[a-zA-Z]+', original_normalized)
            match_found = any(part in correct_option_normalized for part in original_parts if len(part) > 0)
            if not match_found:
                validation['errors'].append(
                    f"Original answer '{original_answer}' not equivalent to correct option '{correct_choice}: {correct_option_text}'"
                )

    original_stripped = strip_latex_math(original_answer)
    for opt, opt_text in options.items():
        if opt != correct_choice:
            opt_stripped = strip_latex_math(opt_text)
            if (is_math_equiv(opt_stripped, original_stripped)
                    and normalize_answer(opt_stripped) == normalize_answer(original_stripped)):
                validation['errors'].append(
                    f"Original answer '{original_answer}' found in distractor option '{opt}: {opt_text}'"
                )
    
    validation['passed'] = len(validation['errors']) == 0
    return validation


def validate_fill_in(result, original_answer, generated_text=""):
    """Validate a generated fill-in-the-blank question. Fails if output lacks \\boxed{...}."""
    question = result['question']
    validation = {
        'passed': False,
        'errors': []
    }

    if "\\boxed" not in generated_text:
        validation['errors'].append("Model output must contain \\boxed{...}; no extraction from other formats")
        validation['passed'] = False
        return validation

    boxed = last_boxed_only_string(generated_text)
    if boxed and boxed != "N/A":
        mask_answer = remove_boxed(boxed).strip()
        if mask_answer and is_math_equiv(mask_answer, original_answer):
            validation['errors'].append(
                "Masked answer in \\boxed{} must not be identical to the original answer (should contain underscores)"
            )

    if "_" not in question:
        validation['errors'].append("Question should contain underscores (_) to indicate blanks")

    # hint line format: "The answer should look like: <mask>. Fill the blank by giving the full answer."
    hint_match = re.search(r'The answer should look like:\s*(.+)', question)
    if hint_match:
        mask_hint = hint_match.group(1)
        mask_hint = re.sub(r'\.\s*Fill the blank.*', '', mask_hint).strip()

        gold_digit_count = len(re.findall(r'\d', str(original_answer)))
        mask_digit_count = (len(re.findall(r'\d', mask_hint))
                            + len(re.findall(r'_(?!\{)', mask_hint)))

        if gold_digit_count > 0 and mask_digit_count != gold_digit_count:
            validation['errors'].append(
                f"Mask digit count ({mask_digit_count}) != gold answer digit count ({gold_digit_count}); "
                f"gold: '{original_answer}', mask: '{mask_hint}'"
            )

    validation['passed'] = len(validation['errors']) == 0
    return validation

def print_validation_report(validations, mode):
    """Print a validation summary report."""
    total = len(validations)
    passed = sum(1 for v in validations if v['passed'])
    failed = total - passed
    
    print("\n" + "="*70)
    print(f"VALIDATION REPORT - Mode: {mode}")
    print("="*70)
    print(f"Total samples: {total}")
    print(f"Passed: {passed} ({passed/total*100:.2f}%)")
    print(f"Failed: {failed} ({failed/total*100:.2f}%)")
    
    if failed > 0:
        print("\n" + "-"*70)
        print("Failed validations (showing first 10):")
        print("-"*70)
        shown = 0
        for i, v in enumerate(validations):
            if not v['passed'] and shown < 10:
                print(f"\nSample #{i+1}:")
                for error in v['errors']:
                    print(f"  - {error}")
                if 'original_answer' in v:
                    print(f"  gold_answer:        {v['original_answer']}")
                if 'rewritten_question' in v:
                    print(f"  rewritten_question: {v['rewritten_question']}")
                shown += 1
    
    print("="*70 + "\n")
    
    return passed, failed


def is_rollout_state(obj):
    """Check if loaded JSON is a rollout state file (from 1_get_hard_samples.py)."""
    return isinstance(obj, dict) and 'samples' in obj and 'current_round' in obj


def load_data(input_file, filter_num_pass_zero=False):
    """Load input data. If the file is a rollout state and filter_num_pass_zero=True, return only num_pass==0 samples."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not is_rollout_state(data):
        if filter_num_pass_zero:
            # Plain list: assume no num_pass field, use all
            return data, None
        return data, None

    # Rollout state: extract samples
    samples = data['samples']
    current_round = data.get('current_round', '?')
    total_rounds = data.get('total_rounds', '?')

    num_pass_zero = [s for s in samples if s.get('num_pass', 0) == 0]
    num_pass_gt_zero = len(samples) - len(num_pass_zero)

    stats = {
        'is_rollout_state': True,
        'current_round': current_round,
        'total_rounds': total_rounds,
        'total_samples': len(samples),
        'num_pass_zero': len(num_pass_zero),
        'num_pass_gt_zero': num_pass_gt_zero,
    }

    if filter_num_pass_zero:
        return num_pass_zero, stats
    return samples, stats


def create_multiple_choice_prompt(question, gold_answer, num_options, forced_correct_choice=None):
    """Build a prompt asking the model to generate a multiple-choice version of a math problem."""
    if num_options == 4:
        options = "A, B, C, D"
        extra_options = ""
    elif num_options == 10:
        options = "A, B, C, D, E, F, G, H, I, J"
        extra_options = """E. [option E]
F. [option F]
G. [option G]
H. [option H]
I. [option I]
J. [option J]"""
    else:  # 16 options
        options = "A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P"
        extra_options = """E. [option E]
F. [option F]
G. [option G]
H. [option H]
I. [option I]
J. [option J]
K. [option K]
L. [option L]
M. [option M]
N. [option N]
O. [option O]
P. [option P]"""
    
    forced_choice_text = ""
    if forced_correct_choice:
        forced_choice_text = f"6. The correct answer must be option {forced_correct_choice}\n"

    prompt = f"""Please create a multiple-choice question with {num_options} options ({options}) based on the following math problem and its correct answer.

Original Question: {question}
Correct Answer: {gold_answer}

Requirements:
1. Keep the original question
2. Add {num_options} options ({options})
3. One option should be the correct answer
4. The other {num_options - 1} options should be plausible but incorrect distractors
5. Randomly place the correct answer among the options
{forced_choice_text.strip()}

Output format:
Question: [the question]
A. [option A]
B. [option B]
C. [option C]
D. [option D]
{extra_options}
Correct Answer: [letter of correct option]

Only output in this exact format, nothing else."""
    
    return prompt


def parse_multiple_choice_response(response, num_options):
    """Parse a model response into (question_text, options_dict, correct_answer)."""
    lines = response.strip().split('\n')
    question_text = ""
    options = {}
    correct_answer = ""
    
    for line in lines:
        line = line.strip()
        if line.startswith("Question:"):
            question_text = line.replace("Question:", "").strip()
        elif line.startswith("Correct Answer:"):
            correct_answer = line.replace("Correct Answer:", "").strip()
        else:
            # Parse options
            for opt in (['A', 'B', 'C', 'D'] if num_options == 4
                        else ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
                        if num_options == 10
                        else ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']):
                if line.startswith(f"{opt}."):
                    options[opt] = line.replace(f"{opt}.", "").strip()
                    break
    
    if question_text and options:
        return question_text, options, correct_answer
    
    return None, None, None


def create_fill_in_prompt(question, gold_answer):
    """Build a prompt asking the model to generate a masked fill-in-the-blank answer."""
    prompt = f"""Your task is to produce a masked version of the correct answer by replacing some digits with underscores (_).

Requirements:
1. The masked answer is the correct answer with some digits replaced by underscores (_)
2. Preserve LaTeX formatting in the masked answer (e.g., if answer is $\\frac{1}{2}$, mask it as $\\frac{1}{{_}}$ or similar)
3. Mask approximately 50-80% of the digits, keeping at least one digit visible
4. Only mask numbers, not letters or latex symbols

Output format:
\\boxed{{[masked answer with underscores only]}}

Examples:
If the answer is 1003, output: \\boxed{{1__3}} or \\boxed{{__03}}
If the answer is $\\frac{5}{8}$, output: \\boxed{{\\frac{5}{{_}}}} or \\boxed{{\\frac{{_}}{8}}}

Only output the masked answer in \\boxed{{}}, nothing else.

Original Question: {question}
Correct Answer: {gold_answer}
Masked Answer: """    
    return prompt


def parse_fill_in_response(response, original_question):
    """Extract masked answer from \\boxed{...}; returns empty string (failing validation) if absent."""
    response = response.strip()
    masked_answer = ""

    boxed = last_boxed_only_string(response)
    if boxed and boxed != "N/A":
        masked_answer = remove_boxed(boxed).strip()
        
    question = f"{original_question}\nThe answer should look like: {masked_answer}. Fill the blank by giving the full answer."
    return question


def apply_chat_template_to_prompts(prompts, tokenizer):
    """Wrap raw prompt strings in the model's chat template format."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]


def process_fill_in(data, llm, sampling_params, tokenizer):
    """Generate fill-in-the-blank variants for all samples."""
    results = []
    prompts = []
    metadata = []
    
    print("Preparing prompts for fill-in-the-blank questions...")
    for item in data:
        prompt = create_fill_in_prompt(
            item['question'],
            item['gold_answer']
        )
        prompts.append(prompt)
        metadata.append({
            'domain': item['domain'],
            'original_question': item['question'],
            'original_answer': item['gold_answer']
        })
    
    print("Generating responses with vLLM...")
    outputs = llm.generate(apply_chat_template_to_prompts(prompts, tokenizer), sampling_params)
    
    print("Processing responses...")
    for i, output in enumerate(tqdm(outputs)):
        generated_text = output.outputs[0].text.strip()
        metadata[i]['generated_text'] = generated_text

        question = parse_fill_in_response(generated_text, metadata[i]['original_question'])

        result = {
            'domain': metadata[i]['domain'],
            'question': question,
            'gold_answer': metadata[i]['original_answer']
        }
        results.append(result)

    return results, metadata

def process_multiple_choice(data, llm, sampling_params, tokenizer, num_options):
    """Generate multiple-choice variants for all samples."""
    results = []
    prompts = []
    metadata = []
    expected_options = (['A', 'B', 'C', 'D'] if num_options == 4 
                       else ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
                       if num_options == 10
                       else ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                             'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'])
    choice_counts = {opt: 0 for opt in expected_options}
    forced_choice_list = (
        expected_options * (len(data) // len(expected_options))
        + expected_options[: len(data) % len(expected_options)]
    )
    random.shuffle(forced_choice_list)
    
    print(f"Preparing prompts for {num_options}-option multiple choice questions...")
    for i, item in enumerate(data):
        forced_correct_choice = forced_choice_list[i]
        prompt = create_multiple_choice_prompt(
            item['question'],
            item['gold_answer'],
            num_options,
            forced_correct_choice=forced_correct_choice
        )
        prompts.append(prompt)
        metadata.append({
            'domain': item['domain'],
            'original_question': item['question'],
            'original_answer': item['gold_answer'],
            'forced_correct_choice': forced_correct_choice
        })
    
    print("Generating responses with vLLM...")
    outputs = llm.generate(apply_chat_template_to_prompts(prompts, tokenizer), sampling_params)
    
    print("Processing responses...")
    for i, output in enumerate(tqdm(outputs)):
        generated_text = output.outputs[0].text.strip()
        
        question_text, options, correct_answer = parse_multiple_choice_response(generated_text, num_options)
        
        if question_text is None:
            # Fallback: use the generated text as is
            question = generated_text
            correct_choice = metadata[i]['forced_correct_choice'] or "A"  # Default fallback
        else:
            forced_choice = metadata[i]['forced_correct_choice']
            if (forced_choice and correct_answer in options and forced_choice in options
                    and correct_answer != forced_choice):
                options[forced_choice], options[correct_answer] = (
                    options[correct_answer],
                    options[forced_choice],
                )
            correct_choice = forced_choice if forced_choice in expected_options else correct_answer
            question = question_text + "\n"
            for opt in expected_options:
                if opt in options:
                    question += f"{opt}. {options[opt]}\n"
            question = question.strip()
        
        if correct_choice not in expected_options:
            correct_choice = "A"
        choice_counts[correct_choice] += 1
        result = {
            'domain': metadata[i]['domain'],
            'question': question,
            'gold_answer': metadata[i]['original_answer'],
            'correct_choice': correct_choice
        }
        results.append(result)

    total_choices = sum(choice_counts.values())
    print("\nCorrect choice distribution:")
    for opt in expected_options:
        count = choice_counts[opt]
        pct = (count / total_choices * 100) if total_choices else 0
        print(f"  {opt}: {count} ({pct:.2f}%)")
    
    return results, metadata


def main():
    parser = argparse.ArgumentParser(description='Convert math problems to different question formats using vLLM')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['four_choice', 'ten_choice', 'sixteen_choice', 'fill_in'],
                        help='Question format: four_choice, ten_choice, sixteen_choice, fill_in')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSON: plain list of samples, or rollout state (e.g. rollout_state_k64.json); if state, only num_pass=0 are processed')
    parser.add_argument('--model', type=str,
                        default='Qwen/Qwen3-4B-Instruct-2507',
                        help='Model path or HuggingFace ID used to generate question variants')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to process (for testing)')
    parser.add_argument('--k', type=int, default=1,
                        help='Rewrite each question k times; keep results that pass validation and are unique (per original sample)')
    parser.add_argument('--validation-report', type=str, default=None,
                        help='Save validation report to JSON file')
    
    args = parser.parse_args()
    if args.k < 1:
        args.k = 1

    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    args.output = os.path.join(os.path.dirname(args.input), f'{input_stem}_{args.mode}.json')
    
    print(f"Loading data from {args.input}...")
    data, rollout_stats = load_data(args.input, filter_num_pass_zero=True)

    if rollout_stats is not None:
        s = rollout_stats
        print("\n" + "=" * 70)
        print("ROLLOUT STATE STATISTICS (from rollout_state_k*.json)")
        print("=" * 70)
        print(f"  Total samples in state:     {s['total_samples']}")
        print(f"  current_round / total_rounds: {s['current_round']} / {s['total_rounds']}")
        print(f"  num_pass == 0 (hard):       {s['num_pass_zero']}  (will process these only)")
        print(f"  num_pass > 0 (skipped):     {s['num_pass_gt_zero']}")
        if s['total_samples'] > 0:
            pct = 100.0 * s['num_pass_zero'] / s['total_samples']
            print(f"  Hard ratio:                  {pct:.2f}%")
        print("=" * 70 + "\n")
        print(f"Processing only the {len(data)} hard samples (num_pass=0)...")
    else:
        print(f"Loaded {len(data)} samples (plain list; processing all)")
    
    if args.limit:
        print(f"Limiting to {args.limit} samples for testing...")
        data = data[:args.limit]
    
    if len(data) == 0:
        print("No samples to process. Exiting.")
        return
    print(f"Loaded {len(data)} samples")

    n_original = len(data)
    if args.k > 1:
        data_expanded = [item for item in data for _ in range(args.k)]
        print(f"Rewriting each question {args.k} times: {len(data_expanded)} total attempts")
    else:
        data_expanded = data
    
    print(f"Initializing tokenizer and vLLM with model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.9
    )
    
    print(f"Processing data in {args.mode} mode...")
    if args.mode == 'four_choice':
        results, metadata = process_multiple_choice(data_expanded, llm, sampling_params, tokenizer, num_options=4)
    elif args.mode == 'ten_choice':
        results, metadata = process_multiple_choice(data_expanded, llm, sampling_params, tokenizer, num_options=10)
    elif args.mode == 'sixteen_choice':
        results, metadata = process_multiple_choice(data_expanded, llm, sampling_params, tokenizer, num_options=16)
    elif args.mode == 'fill_in':
        results, metadata = process_fill_in(data_expanded, llm, sampling_params, tokenizer)

    if args.k > 1:
        for i in range(len(metadata)):
            metadata[i]['original_sample_idx'] = i // args.k

    print("\nValidating results...")
    validations = []
    for i, result in enumerate(tqdm(results)):
        if args.mode in ['four_choice', 'ten_choice', 'sixteen_choice']:
            num_options = 4 if args.mode == 'four_choice' else 10 if args.mode == 'ten_choice' else 16
            validation = validate_multiple_choice(
                result,
                metadata[i]['original_answer'],
                num_options
            )
        elif args.mode == 'fill_in':
            validation = validate_fill_in(
                result,
                metadata[i]['original_answer'],
                metadata[i].get('generated_text', '')
            )

        validation['sample_index'] = i
        validation['original_question'] = metadata[i]['original_question']
        validation['original_answer'] = metadata[i]['original_answer']
        validation['rewritten_question'] = results[i]['question']
        validations.append(validation)
    
    passed_count, failed_count = print_validation_report(validations, args.mode)

    if args.validation_report:
        print(f"Saving validation report to {args.validation_report}...")
        with open(args.validation_report, 'w', encoding='utf-8') as f:
            json.dump(validations, f, ensure_ascii=False, indent=2)
    
    original_count = len(results)
    if args.k <= 1:
        results = [results[i] for i, v in enumerate(validations) if v['passed']]
        filtered_count = original_count - len(results)
        print(f"\nFiltered out {filtered_count} invalid samples. Remaining: {len(results)}")
    else:
        # Group by original sample; keep only passed, then one result per original question
        passed_by_sample = defaultdict(list)
        for i, v in enumerate(validations):
            if v['passed']:
                si = metadata[i]['original_sample_idx']
                passed_by_sample[si].append((i, results[i]))
        final_results = []
        for si in range(n_original):
            # Keep only the first passed result per original question (one row per original)
            if passed_by_sample[si]:
                _, res = passed_by_sample[si][0]
                final_results.append(res)
        results = final_results
        n_passed = sum(1 for v in validations if v['passed'])
        print(f"\nPassed validation: {n_passed}/{original_count} attempts. After dedup (one per original): {len(results)} results from {n_original} original samples")

    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Processed {len(results)} questions and saved to {args.output}")
    print(f"Validation: {passed_count} passed, {failed_count} failed")


if __name__ == '__main__':
    main()
