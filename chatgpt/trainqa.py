import torch 
from model.bigram_model import BigramLanguageModel
from model.tokenizer import Tokenizer
import os
import config

"""
This file will be used for training the model for question answering
    - Input data will have context and question
    - Target data will have answer in form of text
    - There will be a max token limit of block_size for the input context and question
    - Only train question answering model after model has been trained for text generation
    - Trained on SQUAD dataset with context, question and answer
"""

def encode_question_answer():
    pass

torch.manual_seed(1337)

qadata = {}

# read training dataset
with open(config.qa_data_path, 'r', encoding='utf-8') as f:
    pass


# Load the pre-trained model
model = BigramLanguageModel()
model.load_state_dict(torch.load(config.model_path, weights_only=True))
model.to(config.device)


