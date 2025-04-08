
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
OUTPUT_DIR = "models/ppi"
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

# test + dev 합치기
df_test = pd.read_csv("datasets/test.csv")
df_dev = pd.read_csv("datasets/dev.csv")
df = pd.concat([df_test, df_dev])
df = df.rename(columns={"input": "instruction", "response": "output"})
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
print("✅ PPI 파인튜닝 완료")
