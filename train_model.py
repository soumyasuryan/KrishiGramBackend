# train_model.py

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# ===========================
# Model & Tokenizer
# ===========================
model_name = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fix padding token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # Important after adding pad token

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ===========================
# Load Dataset
# ===========================
dataset = load_dataset(
    "json", 
    data_files=r"C:\Users\User\Desktop\All projects\@SIH2025\SIH_project_website\flask-backend\data.json"
)
dataset = dataset["train"].train_test_split(test_size=0.1)  # 90% train, 10% test

max_length = 200

def tokenize_function(example):
    encodings = tokenizer(
        example["input"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    encodings["labels"] = encodings["input_ids"].copy()  # Important for loss computation
    return encodings


tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ===========================
# LoRA Fine-Tuning Setup
# ===========================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_proj","q_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ===========================
# Trainer Arguments
# ===========================
training_args = TrainingArguments(
    output_dir="./fine_tuned_dialo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    fp16=False,  # CPU mode
    save_total_limit=2,
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

# ===========================
# Train & Save Model
# ===========================
trainer.train()
trainer.save_model("./fine_tuned_dialo")

print("✅ Fine-tuning completed and model saved!")
