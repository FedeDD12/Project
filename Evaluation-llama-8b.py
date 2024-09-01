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

results=[]
for question in eval_dataset["Question"]:
  input_text=f"### Question: {question} ### Answer:"
  inputs=tokenizer(input_text, return_tensors="pt")
  outputs=model.generate(**inputs, max_length=256)

  generated_text= tokenizer.decode(outputs[0], skip_special_tokens=True)

  if "### Answer:" in generated_text:
    answer = generated_text.split("### Answer:")[1].strip()
  else:
    answer = generated_text.strip()
    
  results.append({'Question': question, 'Generated Text': answer})
  print(f"Generated text: {generated_text}")

df = pd.DataFrame(results)
df.to_csv('results_llama-8b.csv', index=False)
  
  



