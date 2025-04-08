
import os
import pandas as pd
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
OUTPUT_DIR = "models/empathy"
DATASET_NAME = "datasets/empathy.csv"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# 모델 로딩
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
base_model = prepare_model_for_kbit_training(base_model)

# LoRA 구성
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)

# 데이터 로드 및 전처리
if "empathy" == "cbt":
    dataset = load_dataset("Psychotherapy-LLM/CBT-Bench", split="train")
    def format_example(example):
        messages = [
            {"role": "system", "content": "당신은 CBT 전문가입니다."},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    dataset = dataset.map(lambda x: {"text": format_example(x)}, remove_columns=dataset.column_names)
else:
    df = pd.read_csv(DATASET_NAME)
    df = df.rename(columns={"input": "instruction", "response": "output"})
    df["text"] = df.apply(lambda row: f"<s>[INST] {row['instruction']} [/INST] {row['output']}</s>", axis=1)
    dataset = Dataset.from_pandas(df[["text"]])

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# 학습 설정
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    fp16=True,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ 파인튜닝 완료:", OUTPUT_DIR)
