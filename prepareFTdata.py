import json
import random
from pathlib import Path
from tqdm import tqdm

# ---------- CONFIG ----------
SEMANTIC_SPECS_PATH = "FinalData/semantic_specs.jsonl"
TEMPLATE_BANK_PATH = "FinalData/template_bank.json"
OUTPUT_TRAIN_PATH = "FinalData/train.jsonl"
OUTPUT_VAL_PATH = "FinalData/val.jsonl"
VAL_RATIO = 0.1  # 10% for validation
# ----------------------------

def load_jsonl(path):
    """Load a .jsonl file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    print("Loading files...")
    semantic_specs = load_jsonl(SEMANTIC_SPECS_PATH)
    with open(TEMPLATE_BANK_PATH, "r", encoding="utf-8") as f:
        template_bank = json.load(f)

    template_dict = {t["template_id"]: t for t in template_bank}

    data_pairs = []

    print("Building input/output pairs...")
    for item in tqdm(semantic_specs):
        template_id = item.get("template_id")
        template_entry = template_dict.get(template_id)
        if not template_entry:
            continue

        # Example: LINEAR:divide(ARG0,ARG1)|multiply(ARG0,ARG2)
        template_str = template_entry.get("template_str", item.get("template_str", ""))
        args = item.get("arg_to_number_heuristic", {})
        problem_text = item.get("problem_snippet", "").strip()

        # Convert ARG mappings to readable text
        args_text = ", ".join([f"{k}: {v}" for k, v in args.items()])

        # INPUT: structured description of the math equation + args
        input_text = (
            f"Template: {template_str}. "
            f"Arguments: {args_text}. "
            f"Task: Generate a realistic math word problem that matches this equation."
        )

        # TARGET: actual human problem
        target_text = problem_text

        data_pairs.append({"input_text": input_text, "target_text": target_text})

    print(f"Total pairs built: {len(data_pairs)}")

    # Shuffle and split
    random.shuffle(data_pairs)
    split_idx = int(len(data_pairs) * (1 - VAL_RATIO))
    train_data = data_pairs[:split_idx]
    val_data = data_pairs[split_idx:]

    # Save JSONL
    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    save_jsonl(train_data, OUTPUT_TRAIN_PATH)
    save_jsonl(val_data, OUTPUT_VAL_PATH)

    print(f"✅ Saved {len(train_data)} training and {len(val_data)} validation examples.")
    print(f"Train: {OUTPUT_TRAIN_PATH}")
    print(f"Val:   {OUTPUT_VAL_PATH}")

if __name__ == "__main__":
    main()
