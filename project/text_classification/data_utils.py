from datasets import load_dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader
import math

def load_tokenizer(model_name="huawei-noah/TinyBERT_General_4L_312D"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_data(examples, tokenizer, max_length=512):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

def load_data(tokenizer, batch_size=4):
    dataset = load_dataset("ag_news")

    # Tokenize dataset
    tokenized_datasets = dataset.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Create DataLoaders
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)
    return train_loader, test_loader

def calculate_num_training_steps(train_loader, num_epochs):
    """
    Calculate the total number of training steps.
    Args:
        train_loader: DataLoader for the training set.
        num_epochs: Number of epochs for training.
    Returns:
        Total number of training steps.
    """
    # Get the size of the training dataset
    train_dataset_size = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    # Calculate steps per epoch and total training steps
    steps_per_epoch = math.ceil(train_dataset_size / batch_size)
    total_training_steps = steps_per_epoch * num_epochs
    return total_training_steps