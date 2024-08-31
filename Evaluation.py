
import pandas as pd
from adapters import AutoAdapterModel, 
from transformers import AutoTokenizer


model_path="llama-3.1-8b-math"
model = AutoAdapterModel.from_pretrained(model_path)

tokenizer=AutoTokenizer.from_pretrained(model_path)

eval_dataset=pd.read_csv("eval_dataset.csv")
print(eval_dataset.head())

for question in eval_dataset["Question"]:
    input_text=f"### Question: {question} ### Answer:"
    inputs=tokenizer(input_text, return_tensors="pt")
    outputs=model.generate(**inputs, max_length=128)

    generated_text= tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Generated text: {generated_text}")
