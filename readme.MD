# Custom GPT Model

Welcome to the **Custom GPT Model** repository! This project is a decoder-based generative transformer that includes a custom implementation of the GPT architecture. It features a Byte Pair Encoding (BPE) tokenizer and supports training from scratch or fine-tuning on your own datasets. This repository is designed to provide flexibility and scalability for various natural language generation tasks.

---

## Features

- **Decoder-Based Transformer**: Implements the GPT model architecture focused on efficient and powerful text generation.
- **Custom Tokenizer**: Byte Pair Encoding (BPE) tokenizer for efficient text preprocessing and vocabulary management.
- **Training**: Train the model from scratch on your custom dataset.
- **Hyperparameter Tuning**: Extensive support for configuring hyperparameters to suit various training requirements.
- **Modular Codebase**: Clean and modular structure for easy customization and scalability.

---

## Getting Started

### Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8+
- Required Python libraries (install via `requirements.txt`)

```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/custom-gpt.git
cd custom-gpt
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training the Model

To train the model and tokenizer from scratch on your dataset:

```bash
python main.py --training_text_path <path-to-your-dataset> --train_tokenizer True
```

To train only the model, not the tokenizer:

```bash
python main.py --training_text_path <path-to-your-dataset> --train_model True
```

#### Example:

```bash
python main.py --training_text_path input.txt --train_tokenizer True
```

### Inference

To use the model for inference, you can start with any text and the model will complete it.

```bash
python main.py --inference_text <input text in string> --max_new_tokens 250
```

#### Example:

```bash
python main.py --inference_text "Once upon a time" --max_new_tokens 250
```

---

---

## Results

- **Training Logs**: Training and validation loss are logged for performance tracking.
- **Inference results**
Yet here more, the butchery of weeps,
Enlish-morrow of these precisehoods,
Sounds found leave a beast, and two
With the ways to the highest are speak.

CAMILLO:
A luke of Pady, and they do may:
Will is thee not as my speed, what is i' the pine durman
And in thy bows haste no, which commell Angelo?
This life is both we receive up. The lustow
complain I spoke hath eurer than my papering score
Soing to my goo late with eason Lords,
Your queen's enough to did I cry return:
The buy the bound of God, will serve you you


---

## Hyperparameter Tuning

The following hyperparameters can be adjusted in `config.py`:

- **Model Parameters**: Number of layers, hidden size, number of heads, block_size.
- **Training Parameters**: Batch size, learning rate, number of epochs, dropout.
- **Tokenizer Parameters**: Vocabulary size, merge operations.

Example snippet from `config.py`:

```python
# model
block_size = 64 
n_embd = 128
n_head = 4
n_layer = 4

# training
batch_size = 64 
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu' 
eval_iters = 200
dropout = 0.2

# vocabulary
vocab_size = 2256
train_text_path = 'input.txt'
```

## File Structure

```plaintext
custom-gpt/
├── data/                # Dataset files
├── model/               # Model architecture and tokenizer code
├── requirements.txt     # Python dependencies
├── config.py            # Hyperparameter configuration
└── README.md            # Project documentation
```

---

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

---

## Contact

For questions or feedback, feel free to reach out:

- **Name**: Rishik Gupta
- **Email**: rishikgupta@tamu.edu

