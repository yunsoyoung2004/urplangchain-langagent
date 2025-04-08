
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

model_path = "models/mi"
base_model = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

def run(user_input):
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    prompt = f"사용자: {user_input}\n[MI 상담가]:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9)
    return tokenizer.decode(output[0], skip_special_tokens=True).split(":")[-1].strip()
