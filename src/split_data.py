import json
import random
from pathlib import Path

input_path = Path(__file__).parent / "data" / "BMH_full.json"
output_dir = input_path.parent

with open(input_path) as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)

split = int(len(data) * 0.8)
train, test = data[:split], data[split:]

with open(output_dir / "BMH_train.json", "w") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)

with open(output_dir / "BMH_test.json", "w") as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

print(f"Total: {len(data)} | Train: {len(train)} | Test: {len(test)}")
