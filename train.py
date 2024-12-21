import torch
from model.bigram_model import BigramLanguageModel
from model.tokenizer import Tokenizer
import os
import pickle
from config import (
    block_size, batch_size, device, eval_iters, learning_rate,
    max_iters, eval_interval, vocab_size, train_text_path
)

# random seed for reproducibility
torch.manual_seed(1337)

class DataHandler:
    """Handles dataset encoding, saving, and loading."""
    def __init__(self, text_path, vocab_size, encoded_file):
        self.text_path = text_path
        self.tokenizer = Tokenizer(vocab_size)
        self.encoded_file = encoded_file
        self.text = self._load_text()
        self.encoded_data = self._load_or_encode_data()

    def _load_text(self):
        """Loads the training text from the file."""
        with open(self.text_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_encoded_data(self):
        """Loads encoded data from the disk if available."""
        if os.path.exists(self.encoded_file):
            with open(self.encoded_file, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_encoded_data(self, encoded_data):
        """Saves encoded data to the disk."""
        with open(self.encoded_file, 'wb') as f:
            pickle.dump(encoded_data, f)

    def _load_or_encode_data(self):
        """Loads encoded data or encodes the text if not already encoded."""
        encoded_data = self._load_encoded_data()
        if encoded_data is None:
            self.tokenizer.train_tokenizer(self.text)
            encoded_data = self.tokenizer.encode(self.text)
            self._save_encoded_data(encoded_data)
        else:
            print("---Loaded encoded data from disk---")
        return encoded_data

    def get_data_splits(self):
        """Returns train and validation data splits as tensors."""
        data = torch.tensor(self.encoded_data, dtype=torch.long)
        n = int(0.9 * len(data))
        return data[:n], data[n:]
    
class BatchLoader:
    """Handles batch generation for training and validation."""
    def __init__(self, train_data, val_data, block_size, batch_size, device):
        self.train_data = train_data
        self.val_data = val_data
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch(self, split):
        """Generates a batch of data inputs and targets."""
        data = self.train_data if split == 'train' else self.val_data
        indices = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in indices])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in indices])
        return x.to(self.device), y.to(self.device)



class ModelHandler:
    """Handles training, evaluation, and saving/loading of the model."""
    def __init__(self, model_class, device, learning_rate):
        self.model = model_class().to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def train(self, train_data, val_data, batch_loader, max_iters, eval_interval, eval_iters):
        """Trains the model."""
        self.model.train()
        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = self.estimate_loss(batch_loader, eval_iters)
                print(f"Step {iter}: Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")

            # Train on a batch
            x_batch, y_batch = batch_loader.get_batch('train')
            _, loss = self.model(x_batch, y_batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def estimate_loss(self, batch_loader, eval_iters):
        """Estimates the loss for train and validation data."""
        losses = {split: torch.zeros(eval_iters) for split in ['train', 'val']}
        self.model.eval()
        for split in ['train', 'val']:
            for k in range(eval_iters):
                x_batch, y_batch = batch_loader.get_batch(split)
                _, loss = self.model(x_batch, y_batch)
                losses[split][k] = loss.item()
        self.model.train()
        return {split: losses[split].mean().item() for split in losses}

    def save_model(self, save_path):
        """Saves the model's state_dict to a file."""
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, model_class, save_path):
        """Loads the model's state_dict from a file."""
        self.model = model_class().to(device)
        self.model.load_state_dict(torch.load(save_path))
        self.model.eval()
        print(f"Model loaded from {save_path}")
        return self.model


def main():
    # Initialize data handler and batch loader
    data_handler = DataHandler(
        text_path=train_text_path, 
        vocab_size=vocab_size, 
        encoded_file='load/encoded_data.pkl'
    )
    train_data, val_data = data_handler.get_data_splits()
    batch_loader = BatchLoader(
        train_data=train_data, 
        val_data=val_data, 
        block_size=block_size, 
        batch_size=batch_size, 
        device=device
    )

    # Initialize and train the model
    model_handler = ModelHandler(BigramLanguageModel, device, learning_rate)
    print(f"Number of model parameters: {sum(p.numel() for p in model_handler.model.parameters()) / 1e6} M parameters")
    model_handler.train(
        train_data=train_data, 
        val_data=val_data, 
        batch_loader=batch_loader, 
        max_iters=max_iters, 
        eval_interval=eval_interval, 
        eval_iters=eval_iters
    )

    # Save the trained model
    model_handler.save_model('load/gpt_model.pth')


if __name__ == "__main__":
    main()

