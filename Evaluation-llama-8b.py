import transformers
import pandas as pd
import torch
from transformers import AutoTokenizer

from huggingface_hub.hf_api import HfFolder; HfFolder.save_token("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")

model_name="meta-llama/Meta-Llama-3.1-8B"
tokenizer=AutoTokenizer(model_name)
eval_dataset=pd.read_csv("eval_dataset.csv")

for question in eval_dataset["Question"]:
  input_text=f"### Question: {question} ### Answer:"
  



