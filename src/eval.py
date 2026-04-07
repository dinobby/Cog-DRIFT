import os
import json
from utils import *
from glob import glob
from datetime import datetime
import random
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
from datasets import load_dataset, load_from_disk
import pandas as pd
import re 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Model path/ID.",
    )
    parser.add_argument("--dataset", type=str, default="BMH_test")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="which GPUs to use")
    parser.add_argument("--save_output", action="store_true", help="whether to save the output")
    parser.add_argument("--print_random_n", type=int, default=5, help="print N random model outputs after evaluation (from final k run); set 0 to disable")
    
    args = parser.parse_args()
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    num_gpus = len(args.gpus.split(","))
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        max_model_len=16384,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=num_gpus,
    )

    if args.dataset == "BMH_test":
        with open("/nas-ssd2/cychen/Cog-DRIFT/src/data/BMH_test.json") as f:
            test_data = json.load(f)

    elif args.dataset == "AIME2024":
        test = load_dataset("Maxwell-Jia/AIME_2024", split="train").to_list()
        test_data = []
        for sample in test:
            test_data.append({
                "question": sample['Problem'],
                "gold_answer": sample['Answer']
            })

    elif args.dataset == "AIME2025":
        test1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test").to_list()
        test2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test").to_list()
        test = test1 + test2
        test_data = []
        for sample in test:
            test_data.append({
                "question": sample['question'],
                "gold_answer": sample['answer']
            })

    elif args.dataset == "GPQA":
        test = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train").to_list()
        test_data = []
        for sample in test:
            choices = [
                sample['Correct Answer'],
                sample['Incorrect Answer 1'],
                sample['Incorrect Answer 2'],
                sample['Incorrect Answer 3'],
            ]
            random.shuffle(choices)
            labels = ["A", "B", "C", "D"]
            choices_str = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
            question = (
                f"{sample['Question']}\n\n"
                f"{choices_str}"
            )
            test_data.append({
                "question": question,
                "gold_answer": sample['Correct Answer']
            })

    elif args.dataset == "OmniMATH":
        test = load_dataset("KbsdJames/Omni-MATH", split="test")
        test = test.filter(lambda x: x["difficulty"] > 7.5).to_list()
        test_data = []
        for sample in test:
            test_data.append({
                "question": sample['problem'],
                "gold_answer": sample['answer']
            })

    elif args.dataset == "Date":
        import requests
        response = requests.get("https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/date_understanding/task.json")
        response.raise_for_status()
        test = response.json()["examples"]
        test_data = []
        for sample in test:
            test_data.append({
                "question": sample['input'],
                "gold_answer": next(k for k, v in sample['target_scores'].items() if v == 1)
            })

    prompts = []
    ground_truths = []
    if args.dataset in ["GPQA", "Date"]:
        INSTRUCTION_FOLLOWING = "\n\nThink step by step, and put the full answer text (not the option label) in \\boxed{}."
    else:
        INSTRUCTION_FOLLOWING = "\n\nThink step by step and put the final answer within \\boxed{}."
        
    for sample in test_data:
        msg = [
            {"role": "user", "content": sample['question'] + INSTRUCTION_FOLLOWING}
        ]
                
        prompt = tokenizer.apply_chat_template(msg,
                                                tokenize=False,
                                                add_generation_prompt=True)
        prompts.append(prompt)
        ground_truths.append(sample['gold_answer'])
    
    print(prompts[0])
    print(f"\nTotal samples in test_data: {len(test_data)}")
    
    sampling_params = SamplingParams(temperature=0.7,
                                    max_tokens=8192)

    # Create batched prompts: duplicate each prompt k times
    batched_prompts = []
    prompt_to_sample_idx = []  # Track which sample each prompt belongs to
    
    for i, prompt in enumerate(prompts):
        for k_idx in range(args.k):
            batched_prompts.append(prompt)
            prompt_to_sample_idx.append(i)
    
    print(f"Total prompts in batch: {len(batched_prompts)} (= {len(test_data)} samples × {args.k} runs)")
    
    # Run single batched inference
    print(f"\n{'='*50}")
    print(f"Starting batched inference with {len(batched_prompts)} prompts")
    print(f"{'='*50}")
    
    outputs = llm.generate(batched_prompts, sampling_params)
    
    # Organize results by sample
    # all_results[sample_idx][k_idx] contains the result for sample i, attempt k
    all_results = [[None for _ in range(args.k)] for _ in range(len(test_data))]
    correct_per_run = [0] * args.k  # Track correctness for each k position (for acc@k)
    
    for output_idx, output in enumerate(outputs):
        sample_idx = prompt_to_sample_idx[output_idx]
        k_idx = output_idx % args.k
        
        generated_text = output.outputs[0].text
        
        # Extract answer from \boxed{}
        extracted_answer = last_boxed_only_string(generated_text)
        extracted_answer = remove_boxed(extracted_answer)
        
        # Check correctness using is_math_equiv
        try:
            is_correct = is_math_equiv(ground_truths[sample_idx], extracted_answer)
        except:
            is_correct = False
        
        result = {
            "question": test_data[sample_idx]['question'],
            "ground_truth": ground_truths[sample_idx],
            "generated_text": generated_text,
            "extracted_answer": extracted_answer,
            "correct": is_correct,
            "sample_idx": sample_idx,
            "k_idx": k_idx
        }
        
        all_results[sample_idx][k_idx] = result
        
        if is_correct:
            correct_per_run[k_idx] += 1
    
    # Calculate pass@k: for each sample, if any of the k runs is correct, count as correct
    pass_at_k_correct = 0
    for sample_idx in range(len(test_data)):
        any_correct = False
        for k_idx in range(args.k):
            if all_results[sample_idx][k_idx]['correct']:
                any_correct = True
                break
        if any_correct:
            pass_at_k_correct += 1
    
    # Calculate acc@k: average accuracy across all k runs
    acc_at_k = sum(correct_per_run) / (args.k * len(test_data))
    pass_at_k = pass_at_k_correct / len(test_data)

    # Calculate pass@n for each power of 2 up to k (and k itself)
    pass_at_n_values = []
    n = 1
    n_list = []
    while n < args.k:
        n_list.append(n)
        n *= 2
    n_list.append(args.k)
    # deduplicate while preserving order
    seen = set()
    n_list = [x for x in n_list if not (x in seen or seen.add(x))]

    for n in n_list:
        correct_count = sum(
            1 for sample_idx in range(len(test_data))
            if any(all_results[sample_idx][k_idx]['correct'] for k_idx in range(n))
        )
        pass_at_n_values.append((n, correct_count / len(test_data), correct_count))

    def majority_vote_correct(answers, gold):
        """Cluster answers by math equivalence, return True if largest cluster's answer matches gold."""
        clusters = []  # list of (representative_answer, count)
        for ans in answers:
            matched = False
            for i, (rep, cnt) in enumerate(clusters):
                try:
                    equiv = is_math_equiv(rep, ans)
                except:
                    equiv = (rep == ans)
                if equiv:
                    clusters[i] = (rep, cnt + 1)
                    matched = True
                    break
            if not matched:
                clusters.append((ans, 1))
        best_answer = max(clusters, key=lambda x: x[1])[0]
        try:
            return is_math_equiv(gold, best_answer)
        except:
            return False

    maj_at_n_values = []
    for n in n_list:
        correct_count = 0
        for sample_idx in range(len(test_data)):
            answers = [all_results[sample_idx][k_idx]['extracted_answer'] for k_idx in range(n)]
            gold = ground_truths[sample_idx]
            if majority_vote_correct(answers, gold):
                correct_count += 1
        maj_at_n_values.append((n, correct_count / len(test_data), correct_count))
    
    # Print random samples (all k attempts for selected samples)
    if args.print_random_n > 0:
        print(f"\n{'='*50}")
        print(f"Random samples (showing all {args.k} attempts per sample)")
        print(f"{'='*50}")
        sample_indices = random.sample(range(len(test_data)), min(args.print_random_n, len(test_data)))
        
        for sample_idx in sample_indices:
            print(f"\n{'='*50}")
            print(f"Sample {sample_idx + 1}")
            print(f"Question: {all_results[sample_idx][0]['question']}")
            print(f"Ground truth: {all_results[sample_idx][0]['ground_truth']}")
            print(f"{'='*50}")
            
            for k_idx in range(args.k):
                result = all_results[sample_idx][k_idx]
                status = "✓ CORRECT" if result['correct'] else "✗ WRONG"
                print(f"\nAttempt {k_idx + 1}/{args.k}: {status}")
                print(f"Extracted answer: {result['extracted_answer']}")
                print(f"Generated text:\n{result['generated_text'][:500]}..." if len(result['generated_text']) > 500 else result['generated_text'])

    # Print final results
    print(f"\n{'='*50}")
    print("Final Results")
    print(f"{'='*50}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Total samples: {len(test_data)}")
    print(f"K (number of runs): {args.k}")
    print(f"\nAcc@{args.k}: {acc_at_k:.4f} ({sum(correct_per_run)}/{args.k * len(test_data)})")
    print(f"\nPass@n (using first n of {args.k} attempts):")
    for n, rate, count in pass_at_n_values:
        print(f"  Pass@{n}: {rate:.4f} ({count}/{len(test_data)})")
    print(f"\nMaj@n (majority vote over first n of {args.k} attempts):")
    for n, rate, count in maj_at_n_values:
        print(f"  Maj@{n}: {rate:.4f} ({count}/{len(test_data)})")

    # Save outputs if requested
    if args.save_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.split("/")[-1]
        output_dir = f"outputs/{model_name}_{args.dataset}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all results
        output_file = os.path.join(output_dir, "all_results.json")
        with open(output_file, "w") as f:
            json.dump({
                "model": args.model,
                "dataset": args.dataset,
                "k": args.k,
                "num_samples": len(test_data),
                "acc_at_k": acc_at_k,
                "pass_at_k": pass_at_k,
                "pass_at_n": {f"pass@{n}": rate for n, rate, _ in pass_at_n_values},
                "maj_at_n": {f"maj@{n}": rate for n, rate, _ in maj_at_n_values},
                "correct_per_run": correct_per_run,
                "all_results": all_results  # already in the right format: [sample_idx][k_idx]
            }, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
