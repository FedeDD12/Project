
import pandas as pd
import torch
import ollama

from huggingface_hub.hf_api import HfFolder; HfFolder.save_token("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")

eval_dataset=pd.read_csv("eval_dataset.csv")

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
        question, answer = qa_text.split('**Answer:**')

for i, text in enumerate(dataset["text"].values()):
  # print(text)
  # print('Generate me a question and an answer about the content of the following article "{}" in the following format:\
      # Question:\
      # Answer:'.format(text))
  response = ollama.chat(model='llama3.1:8b-instruct-q8_0', messages=[
    {
      'role': 'user',
      'content': 'Generate me only one question and answer pair about the content of the following article "{}".\
      The response must be in the following format:\
      "**Question:** <question text>\
      **Answer:** <answer text>"\
      Do not add an additional newline between the question and the answer.\
      Do not add any additional text other than the question and answer, do not offer other information.\
      The question must not be about mathematicians\' life.\
      If the article is about a mathematicians\' life do not generate anything.\
      The question must be a mathematical fact.\
      The answer must be as complete as possible and correlated by an explaination.\
      Do not give questions and answers about books. If the article is only about a book do not write anything.\
      Do not give answers that are too simple.\
      In the answer do not add any newline.\
      Write the answer in only one line.\
      Do not generate any list. \
      Assume that the person reading the question and answer has no access to the article and has no prior knowledge of the topic. Provide the necessary context.\
      '.format(text),
    },
  ])

  print(i, response)
  # Extract the question and answer from the response
  qa_text = response['message']['content']

  try:
    question, answer = qa_text.split('**Answer:**')
