import os
import json
from utils import *
from glob import glob
from datetime import datetime
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pandas as pd
import argparse
import datasets

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct")
    parser.add_argument("--model_name", type=str, default="Qwen4B")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--gpus", type=str, required=True, help="which GPUs to use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    num_gpus = len(args.gpus.split(","))

    output_dir = f"./data"
    os.makedirs(output_dir, exist_ok=True)
    state_file = f"{output_dir}/rollout_state_k_{args.k}.json"
    final_output_file = f"{output_dir}/BMH_pass_at_{args.k}_equals_0.json"

    if os.path.exists(state_file):
        print(f"Resuming from checkpoint: {state_file}")
        state = read_json(state_file)
        train_samples = state['samples']
        current_round = state['current_round']
        print(f"Resuming from round {current_round}, {len([s for s in train_samples if s['num_pass'] == 0])} unsolved samples remaining")
    else:
        print("Starting new experiment...")
        data = datasets.load_dataset("open-r1/Big-Math-RL-Verified-Processed", "level_5")
        data = data["train"].to_list()

        train_samples = []
        for idx, i in enumerate(data):
            tmp = {}
            tmp['id'] = idx
            tmp['question'] = i['prompt']
            tmp['gold_answer'] = i['solution']
            tmp['domain'] = i['domain']
            tmp['llama8b_solve_rate'] = i['llama8b_solve_rate']
            tmp['num_pass'] = 0
            tmp['attempts'] = 0
            train_samples.append(tmp)
        current_round = 0

    MAX_MODEL_LEN = 32768
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    instruction_following = (f"\n\nThink step-by-step and put your final answer within \\boxed{{}}")

    llm = LLM(model = args.model,
            max_model_len=MAX_MODEL_LEN,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.95)

    sampling_params = SamplingParams(temperature=0.7,
                                    max_tokens=8192)

    for round_idx in range(current_round, args.k):
        remaining_samples = [s for s in train_samples if s['num_pass'] == 0]

        if len(remaining_samples) == 0:
            print("All samples solved at least once. Stopping early.")
            break

        print(f"\n========== Round {round_idx + 1}/{args.k} ==========")
        print(f"{len(remaining_samples)} samples remaining")

        prompts = []
        for sample in remaining_samples:
            msg = [
                {"role": "user", "content": sample['question'] + instruction_following}
            ]
            prompt = tokenizer.apply_chat_template(msg,
                                                tokenize=False,
                                                add_generation_prompt=True)
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)

        num_correct_this_round = 0
        for i, output in enumerate(outputs):
            reasoning = output.outputs[0].text
            remaining_samples[i]['attempts'] += 1

            boxed_answer = last_boxed_only_string(reasoning)
            if boxed_answer != "N/A":
                pred = remove_boxed(boxed_answer)
            else:
                pred = "N/A"

            try:
                if is_math_equiv(pred, remaining_samples[i]['gold_answer']):
                    remaining_samples[i]['num_pass'] += 1
                    num_correct_this_round += 1
            except:
                pass

        print(f"Correct this round: {num_correct_this_round}")
        print(f"Remaining for next round: {len(remaining_samples) - num_correct_this_round}")

        state = {
            'current_round': round_idx + 1,
            'samples': train_samples,
            'total_rounds': args.k
        }
        write_json(state, state_file)
        print(f"State saved to {state_file}")

    pass_at_k_equals_0 = [s for s in train_samples if s['num_pass'] == 0 and s['attempts'] >= args.k]

    print(f"\n========== Final Stats ==========")
    print(f"Total samples: {len(train_samples)}")
    print(f"pass@{args.k}=0 samples: {len(pass_at_k_equals_0)}")
    print(f"Ratio: {len(pass_at_k_equals_0) / len(train_samples) * 100:.2f}%")

    write_json(pass_at_k_equals_0, final_output_file)
    print(f"\npass@{args.k}=0 samples saved to: {final_output_file}")

    if len(pass_at_k_equals_0) > 0:
        print("\n===============")
        print("Sample:")
        print(f"Question: {pass_at_k_equals_0[0]['question'][:200]}...")
        print(f"Gold Answer: {pass_at_k_equals_0[0]['gold_answer']}")
        print("===============")

    if os.path.exists(state_file) and len([s for s in train_samples if s['num_pass'] == 0]) == 0:
        os.remove(state_file)
        print("\nAll samples solved. State file removed.")
