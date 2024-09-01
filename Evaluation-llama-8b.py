import transformers
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub.hf_api import HfFolder; HfFolder.save_token("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")

model_name="meta-llama/Meta-Llama-3.1-8B"
model=AutoModelForCausalLM.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

eval_dataset=pd.read_csv("eval_dataset.csv")
eval_dataset=Dataset.from_pandas(eval_dataset)

for question in eval_dataset["Question"]:
  input_text=f"### Question: {question} ### Answer:"
  
  



