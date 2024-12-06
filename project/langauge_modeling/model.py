import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMLanguageModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Save model hyperparameter
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, x, hidden):
        x = self.embed(x)  # Embedding layer
        out, hidden = self.lstm(x, hidden)  # LSTM layer
        out = self.fc(out)  # Fully connected layer
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()) 
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_size))
