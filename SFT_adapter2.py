import transformers
import torch
import pandas as pd
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from datasets import Dataset

from huggingface_hub.hf_api import HfFolder; HfFolder.save_token("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")

device_map={"":0}
model_name="meta-llama/Meta-Llama-3.1-8B"
#new_model = "llama-3.1-8b-math"

compute_dtype=getattr(torch, "float16")

bnb_config= BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

if compute_dtype==torch.float16:
    major,_=torch.cuda.get_device_capability()
    if major>=8:
        print(80)

model=AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
)

model.config.use_cache=False
model.config.pretraining_tp=1

tokenizer=AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset=pd.read_csv("questions_and_answers.csv")
dataset=Dataset.from_pandas(dataset)

dataset=dataset.train_test_split(test_size=0.1)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['Question'])):
        text = f"### Question: {example['Question'][i]}\n ### Answer: {example['Answer'][i]}"
        output_texts.append(text)
    return output_texts

#instruction_template="### Question:"
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)


peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    output_dir="/tmp",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    save_steps=10,
    save_total_limit=2,
    logging_steps=10,
    logging_dir="/tmp",
    overwrite_output_dir=True
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset["train"],
    #eval_dataset=dataset["test"]
    args=SFTConfig(output_dir="/tmp"),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    peft_config=peft_config,
    tokenizer=tokenizer,
    max_seq_length=128,
)

trainer.train()

#trainer.model.save_pretrained(new_model)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device) # Move the model to the device

#input_text = "### Question: What is the generating function for the actuarial polynomials a(β)n(x)?  ### Answer:"
#inputs = tokenizer(input_text, return_tensors="pt")
#outputs = model.generate(**inputs, max_length=100)

#generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#print(f"Generated text: {generated_text}")


eval_dataset=dataset["test"]

for question in eval_dataset["Question"]:
    input_text=f"### Question: {question} ### Answer:"
    inputs=tokenizer(input_text, return_tensor="pt")
    outputs=model.generate(**inputs, max_length=128)

    generated_text= tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Generated text: {generated_text}")

