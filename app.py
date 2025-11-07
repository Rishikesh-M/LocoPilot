# train_codegen_webui.py
import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
import gradio as gr


# ============================================================
# STEP 1: LOAD MULTIPLE DATASETS
# ============================================================
print("📦 Loading datasets...")

# Load Python code dataset
flytech_ds = load_dataset("flytech/python-codes-25k")

# Load OpenAI GSM8K datasets (main and socratic)
gsm8k_main = load_dataset("openai/gsm8k", "main")
gsm8k_socratic = load_dataset("openai/gsm8k", "socratic")


# Unify all datasets into prompt/code format
def unify_gsm8k(example):
    prompt = example["question"]
    code = "# Answer:\n" + example["answer"]
    return {"prompt": prompt, "code": code}

def unify_flytech(example):
    # Combine instruction + input as the prompt, output as the code
    prompt = example["instruction"]
    if example.get("input"):  # include extra context if available
        prompt += "\n" + example["input"]
    return {"prompt": prompt, "code": example["output"]}
# Process datasets
if "validation" in flytech_ds:
    flytech_train = flytech_ds["train"].map(unify_flytech)
    flytech_valid = flytech_ds["validation"].map(unify_flytech)
else:
    # create validation split manually
    split = flytech_ds["train"].train_test_split(test_size=0.1, seed=42)
    flytech_train = split["train"].map(unify_flytech)
    flytech_valid = split["test"].map(unify_flytech)

gsm8k_main = gsm8k_main["train"].map(unify_gsm8k)
gsm8k_socratic = gsm8k_socratic["train"].map(unify_gsm8k)

# Combine all datasets
train_data = concatenate_datasets([flytech_train, gsm8k_main, gsm8k_socratic])
valid_data = flytech_valid

dataset = {"train": train_data, "validation": valid_data}

print(f"✅ Combined dataset size: Train={len(train_data)}, Validation={len(valid_data)}")

# ============================================================
# STEP 2: LOAD MODEL & TOKENIZER
# ============================================================
model_name = "deepseek-ai/deepseek-coder-1.3b-base"
print("🔄 Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ============================================================
# STEP 3: PREPROCESS FUNCTION
# ============================================================
def preprocess_function(examples):
    inputs = [f"{p}\n{c}" for p, c in zip(examples["prompt"], examples["code"])]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

print("🧹 Tokenizing dataset (this may take a while)...")
tokenized_datasets = {
    "train": dataset["train"].map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names),
    "validation": dataset["validation"].map(preprocess_function, batched=True, remove_columns=dataset["validation"].column_names),
}

# ============================================================
# STEP 4: TRAINING CONFIG
# ============================================================
training_args = TrainingArguments(
    output_dir="./fine_tuned_codegen_model",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy="epoch",  # keep only supported args
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(2000)),  # subset for demo
    eval_dataset=tokenized_datasets["validation"].shuffle(seed=42).select(range(200)),
)

# ============================================================
# STEP 5: TRAINING FUNCTION (FOR WEBUI)
# ============================================================
def train_model():
    try:
        trainer.train()
        model.save_pretrained("./my_custom_codegen_model")
        tokenizer.save_pretrained("./my_custom_codegen_model")
        return "✅ Fine-tuning complete! Model saved to './my_custom_codegen_model'"
    except Exception as e:
        return f"❌ Training Error: {e}"

# ============================================================
# STEP 6: GENERATION FUNCTION
# ============================================================
def generate_code(prompt, max_len, temperature):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"❌ Generation Error: {e}"

# ============================================================
# STEP 7: GRADIO WEBUI
# ============================================================
def build_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as ui:
        gr.Markdown("# 🤖 CodeGen WebUI — Multi-Dataset Fine-Tuning & Generation")

        with gr.Tab("🚀 Train Model"):
            gr.Markdown("Train the DeepSeek Coder model on Flytech + GSM8K datasets.")
            train_btn = gr.Button("Start Fine-tuning")
            output_train = gr.Textbox(label="Training Log", lines=8)
            train_btn.click(train_model, outputs=output_train)

        with gr.Tab("💡 Generate Code / Answer"):
            prompt = gr.Textbox(label="Enter your instruction", placeholder="Write a Python function to check prime numbers.")
            max_len = gr.Slider(50, 512, value=200, step=10, label="Max Length")
            temp = gr.Slider(0.1, 1.0, value=0.3, step=0.1, label="Creativity (Temperature)")
            gen_btn = gr.Button("Generate Output")
            output_code = gr.Textbox(label="Generated Result", lines=15)

            gen_btn.click(generate_code, inputs=[prompt, max_len, temp], outputs=output_code)

        gr.Markdown("Made with ❤️ using Hugging Face Transformers + Gradio")

    return ui


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)