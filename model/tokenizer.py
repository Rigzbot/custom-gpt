import regex as re
from collections import Counter
from tqdm import tqdm
import os
import pickle


class Tokenizer(object):
    def __init__(self, vocab_size, vocab_file='load/vocab.pkl', merges_file='load/merges.pkl'):
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256
        self.tokens = []
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)} # starting 256 indexes
        self.chunk_size = 10000

        # check if vocab and merges already exist
        self.vocab_file = vocab_file
        self.merges_file = merges_file

        self.model_weights_exist()

    def model_weights_exist(self):
        """ Load pre-trained vocab and merges if available """
        if os.path.exists(self.vocab_file) and os.path.exists(self.merges_file):
            self.load_vocab()
            self.load_merges()
            return True
        else:
            return False
        
    def split_input_text(self, text):
        """ Splits input text into list using regex and returns the list """
        gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        return re.findall(gpt2pat, text)

    def create_utf_encodings(self, text):
        """ Convert text to the UTF-8 encoded bytes and then into integer values """
        subwords = self.split_input_text(text)

        for word in subwords:
            self.tokens.extend(list(map(int, word.encode("utf-8"))))

    def get_stats(self, ids):
        """ Create a dictionary of consecutive pair frequencies """
        return Counter(zip(ids, ids[1:]))
    
    def merge(self, ids, pair, idx):
        """ in the list of ints (ids), replace all consectuive occurences of pair with the new token idx """
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def train_tokenizer(self, text):
        if self.model_weights_exist():
            return
        
        self.create_utf_encodings(text)
        ids = list(self.tokens) # copy so we don't destroy original list
        for i in tqdm(range(self.num_merges), desc="Training tokenizer progress"):
            stats = self.get_stats(ids)
            if not stats:
                break
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, top_pair, idx)
            self.merges[top_pair] = idx

        # Update vocabulary with merged pairs
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        # Save the trained vocab and merges
        self.save_vocab()
        self.save_merges()
    
    def decode(self, ids):
        """ Given a list of integrers, return the corresponding decoded string """

        if not all(isinstance(idx, int) for idx in ids):
            raise ValueError("Input IDs must be a list of integers.")
        
        tokens = b"".join(self.vocab.get(idx, b"?") for idx in ids)  # Replace missing tokens with '?'
        text = tokens.decode("utf-8", errors="replace")
        return text

    
    def encode(self, text, disable_tqdm=False):
        """
            Given a string, return the list of integers (tokenization).
            Processes the input in chunks for efficiency.
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        
        if self.model_weights_exist() == False:
            raise ValueError("Tokenizer needs to be trained before inference.")
        
        tokens = []
        token_list = self.split_input_text(text)
        with tqdm(total=len(token_list), desc="Encoding Progress", disable=disable_tqdm) as pbar:
            for i in range(0, len(token_list), self.chunk_size):
                chunk = token_list[i:i + self.chunk_size]
                # Encode each token in the chunk and merge them
                chunk_tokens = [list(token.encode("utf-8")) for token in chunk]

                # Perform merging for each token's chunk
                for chunk_token in chunk_tokens:
                    while len(chunk_token) >= 2:
                        stats = self.get_stats(chunk_token)
                        pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))  # Find lowest-index mergeable pair
                        if pair not in self.merges:
                            break  # Nothing else can be merged
                        idx = self.merges[pair]
                        chunk_token = self.merge(chunk_token, pair, idx)

                    tokens.extend(chunk_token)
                pbar.update(len(chunk))
        return tokens
    
    def save_vocab(self):
        """Save the vocabulary to a file."""
        try:
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self.vocab, f)
        except Exception as e:
            print(f"Error saving vocabulary: {e}")

    def save_merges(self):
        """Save the merges to a file."""
        try:
            with open(self.merges_file, 'wb') as f:
                pickle.dump(self.merges, f)
        except Exception as e:
            print(f"Error saving merges: {e}")

    def load_vocab(self):
        """Load the vocabulary from a file."""
        try:
            with open(self.vocab_file, 'rb') as f:
                self.vocab = pickle.load(f)
        except Exception as e:
            print(f"Error loading vocabulary: {e}")

    def load_merges(self):
        """Load the merges from a file."""
        try:
            with open(self.merges_file, 'rb') as f:
                self.merges = pickle.load(f)
        except Exception as e:
            print(f"Error loading merges: {e}")
