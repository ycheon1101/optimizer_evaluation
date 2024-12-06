from datasets import load_dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch

def load_tokenizer(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        tokenizer.pad_token_id = tokenizer.vocab_size - 1
    return tokenizer

def tokenize_data(texts, tokenizer):
    tokenized = [torch.tensor(tokenizer.encode(text)) for text in texts if text.strip()]
    return pad_sequence(tokenized, batch_first=True, padding_value=tokenizer.pad_token_id)

def batchify(data, batch_size):
    n_batch = data.size(0) // batch_size
    data = data[:n_batch * batch_size] 
    data = data.view(batch_size, -1).contiguous()  # Shape: (batch_size, seq_len)
    return data

def load_data(tokenizer, batch_size):
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    train_texts = dataset["train"]["text"]
    valid_texts = dataset["validation"]["text"]
    test_texts = dataset["test"]["text"]

    train_data = tokenize_data(train_texts, tokenizer)
    valid_data = tokenize_data(valid_texts, tokenizer)
    test_data = tokenize_data(test_texts, tokenizer)

    train_data = batchify(train_data, batch_size)
    valid_data = batchify(valid_data, batch_size)
    test_data = batchify(test_data, batch_size)

    return train_data, valid_data, test_data

def calculate_num_training_steps(train_data, seq_len, num_epochs):

    total_tokens = train_data.size(1)  # Total number of tokens in training data
    steps_per_epoch = total_tokens // seq_len  # Tokens processed in one epoch
    total_training_steps = steps_per_epoch * num_epochs
    return total_training_steps