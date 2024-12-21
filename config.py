import torch

# model
block_size = 128 # what is the maximum context length for prediction
n_embd = 256
n_head = 8
n_layer = 8

# training
batch_size = 64 # how many independent sequences will be processed in parallel
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu' # if training on a mac 
eval_iters = 200
dropout = 0.2

# vocabulary
vocab_size = 5256

#file paths
train_text_path = 'input.txt'
model_path = 'load/gpt_model.pth'
qa_data_path = 'data/qa_train.txt'