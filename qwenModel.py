import torch
import numpy as np
import modal
from modal import Image
from transformers import AutoTokenizer, AutoModelForCausalLM


# Define the Modal stub
app = modal.App("qwen-CoT-inference")

# Create a custom image with the required dependencies
image = Image.debian_slim().pip_install(
    "torch", "transformers", "numpy"
)  # Added ctransformers


@app.function(gpu="A100", image=image)
def run_qwen_model():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M").to(
        device
    )

    # Example math question
    question = "If a train travels 300 miles in 5 hours, what is its average speed?"

    # Generate CoT steps
    input_text = f"Question: {question}\nLet's think step by step:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=500,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
    )

    # Get the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Split into steps and calculate perplexity
    steps = response.split("\n")[2:]  # Skip the question and "Let's think step by step"
    perplexities = []

    for step in steps:
        if step.strip():
            # Calculate perplexity for each step
            step_inputs = tokenizer(step, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**step_inputs, labels=step_inputs["input_ids"])
                log_probs = outputs.logits.log_softmax(dim=-1)
                selected_token_log_probs = log_probs.gather(
                    -1, step_inputs["input_ids"].unsqueeze(-1)
                ).squeeze(-1)
                total_log_prob = selected_token_log_probs.sum().item()
                # perplexity = torch.exp(-total_log_prob / step_inputs["input_ids"].size(1))
                perplexities.append(
                    total_log_prob
                )  # sum of log probs is log of token probability

    # Store results
    cot_steps = steps
    step_perplexities = perplexities
    # Filter out steps with NaN perplexities or blank steps
    valid_indices = [
        i
        for i, (step, perplexity) in enumerate(zip(cot_steps, step_perplexities))
        if not (step.strip() == "" or perplexity != perplexity)  # Check for NaN
    ]

    # Update cot_steps and step_perplexities to only include valid entries
    cot_steps = [cot_steps[i] for i in valid_indices]
    step_perplexities = [step_perplexities[i] for i in valid_indices]

    print("Chain of Thought Steps:")
    for i, (step, perplexity) in enumerate(zip(cot_steps, step_perplexities)):
        print(f"Step {i+1}: {step}")
        print(f"Perplexity: {perplexity:.2f}")


@app.local_entrypoint()
def main():
    run_qwen_model.remote()
