import torch
import argparse
from model.tokenizer import Tokenizer
from model.bigram_model import BigramLanguageModel
import config
import os
import subprocess

def train_model(train_text_path):
    print("Training Bigram Language Model...")
    with open(train_text_path, 'r', encoding='utf-8') as f:
            text = f.read()

    with open(config.train_text_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    subprocess.run(['python', 'train.py'], check=True)
    print("Model training complete and saved.")

def delete_files_in_folder(folder_path):
    # Ensure the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Loop through all files in the directory
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Check if it is a file and then delete it
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    else:
        print(f"The folder {folder_path} does not exist or is not a directory.")


def main():
    # Parsing command-line arguments
    parser = argparse.ArgumentParser(description="Text Generation Inference")
    parser.add_argument("--model_path", type=str, default="load/gpt_model.pth", help="Path to the trained model checkpoint")
    parser.add_argument("--inference_text", type=str, default="\n", help="Input text for inference")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--train_tokenizer", type=bool, default=False, help="Boolean indicator, if you would like to train the tokenizer")
    parser.add_argument("--training_text_path", type=str, default="input.txt", help="Path of training text in disk")
    parser.add_argument("--train_model", type=bool, default=False, help="Boolean indicator, if you would like to train the language model")
    
    args = parser.parse_args()
    
    # Load the tokenizer
    tokenizer = Tokenizer(vocab_size=config.vocab_size)

    if args.train_tokenizer:
        delete_files_in_folder('load')

    if args.train_model and os.path.isfile(args.model_path):
        os.remove(args.model_path)

    # Check if tokenizer has been trained, otherwise train tokenizer
    if not os.path.exists('load/vocab.pkl') or not os.path.exists('load/merges.pkl'):
        with open(args.training_text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print("---Training tokenizer---")
        tokenizer.train_tokenizer(text)

    # Check if model exists, otherwise train the language model
    if not os.path.exists(args.model_path):
        train_model(args.training_text_path)
    
    # Load the model
    model = BigramLanguageModel()  
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.to(config.device)
    model.eval()  # Set the model to evaluation mode
    
    # Prepare input text (encode and convert to tensor)
    input_text = args.inference_text
    encoded_input = tokenizer.encode(input_text, disable_tqdm=True)  # Tokenize the input text
    input_tensor = torch.tensor(encoded_input, dtype=torch.long, device=config.device).unsqueeze(0)  # Add batch dimension
    
    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(input_tensor, max_new_tokens=args.max_new_tokens)
    
    # Decode the generated tokens back into text
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    # Output the generated text
    print(generated_text)

if __name__ == "__main__":
    main()
