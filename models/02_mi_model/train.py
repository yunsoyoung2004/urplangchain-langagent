
import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
OUTPUT_DIR = "models/mi"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# LoRA 적용
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
base = prepare_model_for_kbit_training(base)
peft_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(base, peft_config)

# 데이터셋 통합 로드
dfs = []
for filename in ["MI Dataset.csv", "AnnoMI-simple.csv", "AnnoMI-full.csv"]:
    path = os.path.join("datasets", filename)
    df = pd.read_csv(path)
    if "input" in df.columns and "response" in df.columns:
        df = df.rename(columns={"input": "instruction", "response": "output"})
    elif "utterance" in df.columns and "reply" in df.columns:
        df = df.rename(columns={"utterance": "instruction", "reply": "output"})
    elif "client" in df.columns and "counselor" in df.columns:
        df = df.rename(columns={"client": "instruction", "counselor": "output"})
    dfs.append(df[["instruction", "output"]])

df = pd.concat(dfs).dropna()
df["text"] = df.apply(lambda r: f"<s>[INST] {r['instruction']} [/INST] {r['output']}</s>", axis=1)
dataset = Dataset.from_pandas(df)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, per_device_train_batch_size=2,
    gradient_accumulation_steps=4, num_train_epochs=3,
    learning_rate=2e-5, logging_steps=10,
    save_strategy="epoch", evaluation_strategy="no",
    fp16=True, save_total_limit=2, report_to="none"
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=tokenized_dataset,
    tokenizer=tokenizer, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ MI 파인튜닝 완료")
