from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset("rajpurkar/squad")

with open('data/qa_train.txt', 'w', encoding='utf-8') as file:
    for data in tqdm(ds['train']):
        context = data['context']
        question = data['question']
        answer = data['answers']['text'][0]

        # Write the formatted data to the file
        file.write(f"\\CONTEXT: {context}\n\\QUESTION: {question}\n\\ANSWER: {answer}\n\n")

with open('data/qa_eval.txt', 'w', encoding='utf-8') as file:
    for data in tqdm(ds['validation']):
        context = data['context']
        question = data['question']
        answer = data['answers']['text'][0]

        # Write the formatted data to the file
        file.write(f"\\CONTEXT: {context}\n\\QUESTION: {question}\n\\ANSWER: {answer}\n\n")
