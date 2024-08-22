import ollama
from datasets import load_from_disk

dataset=load_from_disk("filtered_wikipedia_dataset")
print(dataset)
print(dataset[1])


for line in dataset:
  text = line["text"]
  # print('Generate me a question and an answer about the content of the following article "{}" in the following format:\
      # Question:\
      # Answer:'.format(text))
  response = ollama.chat(model='llama3.1:8b-instruct-q8_0', messages=[
    {
      'role': 'user',
      'content': 'Generate me only one question and answer pair about the content of the following article "{}" in the following format:\
      **Question:**\
      **Answer:**\
      Do not add any additional text other than the question and answer, do not offer other information.\
      '.format(text),
    },
  ])
  print(response['message']['content'])
