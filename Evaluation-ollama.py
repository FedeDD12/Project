import ollama
import pandas as pd
import csv

dataset=pd.read_csv("eval_dataset.csv")

qa_pairs=[]

with open('results_noSFT.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Question', 'Answer'])  # Write the header row


for question in dataset["Question"]:

  response = ollama.chat(model='llama3.1:8b', messages=[
    {
      'role': 'user',
      'content': ' ### Question: "{}" ### Answer:'.format(question)
    },
  ])

  print(response)
  # Extract the question and answer from the response
  qa_text = response['message']['content']

  try:
    question, answer = qa_text.split('### Answer:')

    question=question.replace("### Question:", "")
  except Exception as e:
    print("Error {} encountered...skipping...".format(e))
    continue
    
  # Append the question and answer to the list as a tuple

  qa_pairs.append((question.strip(), answer.strip()))

  # Write the question-answer pairs to a CSV file
  with open('questions_and_answers.csv', 'a', newline='', encoding='utf-8') as file:
      writer = csv.writer(file)
