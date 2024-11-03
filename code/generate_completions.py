import json
import os
from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

from dotenv import load_dotenv

load_dotenv()


def load_model(model_name: str = 'codellama/CodeLlama-7b-hf', auth_token: str = None):
    if auth_token:
        login(auth_token)
        print("Logged in to Hugging Face Hub")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=auth_token,
        load_in_4bit=True,
        device_map="auto"
    )

    return tokenizer, model

def calculate_max_new_tokens(prefix: str) -> int:
    if "import" in prefix:
        return 20
    elif "class" in prefix or "def" in prefix:
        return 100
    else:
        return 75

def generate_completion(prefix: str, tokenizer, model) -> str:
    try:
        inputs = tokenizer(prefix, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        max_new_tokens = calculate_max_new_tokens(prefix)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated_text[len(prefix):].strip()
        return completion
    except Exception as e:
        print(f"Error generating completion: {str(e)}")
        return ""


def apply_model_to_dataset(dataset: List[Dict[str, str]], tokenizer, model) -> List[Dict[str, str]]:
    total = len(dataset)
    for i, example in enumerate(dataset):
        if (i + 1) % 5 == 0:
            print(f"Processing example {i + 1}/{total}")

        prefix = example['prefix']
        completion = generate_completion(prefix, tokenizer, model)
        dataset[i]['model_completion'] = completion

    return dataset


if __name__ == "__main__":
    auth_token = os.getenv("HF_AUTH_TOKEN")

    with open("../data/code_completion_dataset.json", 'r') as f:
        dataset = json.load(f)

    tokenizer, model = load_model(auth_token=auth_token)

    dataset_with_completions = apply_model_to_dataset(dataset, tokenizer, model)

    output_path = "../data/dataset_with_completions.json"
    with open(output_path, 'w') as f:
        json.dump(dataset_with_completions, f, indent=4)
