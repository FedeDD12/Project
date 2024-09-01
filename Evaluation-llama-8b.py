
import pandas as pd
import torch
import ollama

eval_dataset=pd.read_csv("eval_dataset.csv")

qa_pairs=[]
for question in eval_dataset["Question"]:
    input_text=f"### Question: {question} ### Answer:"
    response = ollama.chat(model='llama3.1:8b-instruct-q8_0', messages=[
    {
      'role': 'user',
      'content': input_text,
    },
    ])
    print(response)
    qa_text = response['message']['content']

    try:
        question, answer = qa_text.split('### Answer:')
        question=question.replace("## Question:", "")
    except Exception as e:
        print("Error {} encountered...skipping...".format(e))
        continue
    

    qa_pairs.append((question.strip(), answer.strip()))

    # Write the question-answer pairs to a CSV file
    with open('questions_and_answers.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(qa_pairs[-1]) 

