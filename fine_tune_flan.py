from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments
)
import torch

# 🔹 1️⃣ Load model + tokenizer
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 🔹 2️⃣ Move model to the right device (MPS or CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"✅ Using device: {device}")

# 🔹 3️⃣ Load dataset
dataset = load_dataset("json", data_files={
    "train": "FinalData/train.jsonl",
    "validation": "FinalData/val.jsonl"
})

# 🔹 4️⃣ Preprocess function
def preprocess_function(batch):
    inputs = tokenizer(batch["input_text"], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(batch["target_text"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 🔹 5️⃣ Training setup
training_args = TrainingArguments(
    output_dir="./results_flan_t5_small_mac",
    evaluation_strategy="epoch",
    learning_rate=3e-4,               # Slightly higher helps on CPU/MPS
    per_device_train_batch_size=2,    # Reduce for memory safety
    per_device_eval_batch_size=2,
    num_train_epochs=3,               # You can increase later
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=False,                       # ❌ disable fp16 on Mac
    bf16=False,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none"                  # disable wandb/tensorboard if not needed
)

# 🔹 6️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 🔹 7️⃣ Start training
trainer.train()

# 🔹 8️⃣ Save the fine-tuned model
trainer.save_model("./flan_t5_formulaVerse_mac")
print("✅ Model fine-tuned and saved to ./flan_t5_formulaVerse_mac")
