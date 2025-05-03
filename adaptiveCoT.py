import torch
import numpy as np
import modal
from modal import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datetime import datetime
from datasets import load_dataset


# Define the Modal stub
app = modal.App("qwen-CoT-inference")

# Create a custom image with the required dependencies
image = Image.debian_slim().pip_install(
    "torch", "transformers", "numpy", "datasets"
)  # Added ctransformers
volume = modal.Volume.from_name("benchmark-responses")


@app.function(gpu="A100", image=image, volumes={"/responses": volume})
def run_qwen_model():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M").to(
        device
    )

    def chainOfThought(question):
        input_text = f"Question: {question}\nLet's think step by step:\n"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        steps = response.split("\n")[2:]
        perplexities = []

        for step in steps:
            if step.strip():
                step_inputs = tokenizer(step, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**step_inputs, labels=step_inputs["input_ids"])
                    log_probs = outputs.logits.log_softmax(dim=-1)
                    selected_token_log_probs = log_probs.gather(
                        -1, step_inputs["input_ids"].unsqueeze(-1)
                    ).squeeze(-1)
                    total_log_prob = selected_token_log_probs.sum().item()
                    perplexities.append(total_log_prob)
        cot_steps = steps
        step_perplexities = perplexities
        valid_indices = [
            i
            for i, (step, perplexity) in enumerate(zip(cot_steps, step_perplexities))
            if not (step.strip() == "" or perplexity != perplexity)  # Check for NaN
        ]
        cot_steps = [cot_steps[i] for i in valid_indices]
        step_perplexities = [step_perplexities[i] for i in valid_indices]
        # Find the step with the lowest perplexity among the non-final steps
        if len(step_perplexities) > 1:
            min_perplexity_index = min(
                range(len(step_perplexities) - 1), key=lambda i: step_perplexities[i]
            )
        else:
            min_perplexity_index = 0

        # Prepare the final list of steps
        final_steps = []
        # Add the first two steps from the original response that were not included in steps
        original_response_steps = response.split("\n")
        non_step_indices = [
            i for i, step in enumerate(original_response_steps) if step not in cot_steps
        ]
        final_steps.extend([original_response_steps[i] for i in non_step_indices[:2]])
        # Add the step with the lowest perplexity
        final_steps.append(cot_steps[min_perplexity_index])
        # Add the last step
        final_steps.append(cot_steps[-1])

        return final_steps

    def run_gsm():
        os.chdir("/data")
        ds = load_dataset("csv", data_files={"train": "gsm8k_train.csv"})

        questions = ds["train"]["question"][:5]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(timestamp, exist_ok=True)
        for i, question in enumerate(questions):
            with open(
                os.path.join("/benchmark-responses", timestamp, f"question_{i+1}.txt"),
                "w",
            ) as f:
                f.write(
                    question + "\n===========\n" + chainOfThought(question).join("\n")
                )

    run_gsm()


@app.local_entrypoint()
def main():
    run_qwen_model.remote()
