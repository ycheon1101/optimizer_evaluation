import math
import torch
from tqdm import tqdm
from train import get_batch

def load_model(model_class, model_path, device, vocab_size, embed_size, hidden_size, num_layers):
    model = model_class(vocab_size, embed_size, hidden_size, num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

device = "cuda" if torch.cuda.is_available() else "cpu" 

def evaluate_with_perplexity(model, data_source, criterion, batch_size, seq_len, vocab_size):
    model.to(device)
    model.eval()
    total_loss = 0
    batch_count = 0
    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(1) - 1, seq_len):
            data, targets = get_batch(data_source, i, seq_len)
            data, targets = data.to(device), targets.to(device)
            
            output, hidden = model(data, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())

            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
            batch_count += 1
    
    avg_loss = total_loss / batch_count
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity
