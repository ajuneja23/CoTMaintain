import modal
import numpy as np
import torch
from modal import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datetime import datetime
from datasets import load_dataset
from load_mathchat import parse_followup


app = modal.App("naive-wipe-cot-eval")

image = Image.debian_slim().pip_install("torch", "transformers", "numpy", "datasets")
volume = modal.Volume.from_name("benchmark-responses")
dataVolume = modal.Volume.from_name("data")


@app.function(
    gpu="A100", image=image, volumes={"/responses": volume, "/data": dataVolume}
)
def run_qwen():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M").to(
        device
    )

    def chainOfThought(question):
        input_text = f"{question}\nLet's think step by step:\n"
        orig_parts = len(input_text.split("\n"))
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        steps = response.split("\n")
        kept_steps = steps[:orig_parts]
        kept_steps.append(steps[-1])
        resultant_context = "\n".join(kept_steps)
        return kept_steps, resultant_context

    parsed_data = parse_followup("/data/BenchMark/follow_up.jsonl")
    for i in range(5):
        model_answers = []
        res1, context = chainOfThought(parsed_data[i]["question"])
        model_answers.append(res1[-1])
        for a_question in parsed_data[i]["a_dialogue"]:
            res1, context = chainOfThought(context + "\n" + a_question)
            model_answers.append(res1[-1])
        print(model_answers)
        print("=" * 100)
