import ollama
import pandas as pd
import csv

dataset=pd.read_csv("eval_dataset.csv")

results=[]

with open('results_noSFT.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Question', 'Generated Text'])  # Write the header row


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

  #try:
    #question, answer = qa_text.split('### Answer:')

    #question=question.replace("### Question:", "")
  #except Exception as e:
    #print("Error {} encountered...skipping...".format(e))
    #continue

  results.append((question.strip(), qa_text.strip())

  with open('results_noSFT.csv', 'a', newline='', encoding='utf-8') as file:
      writer = csv.writer(file)
      writer.writerow(results[-1])
