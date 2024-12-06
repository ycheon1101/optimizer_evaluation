import os
import torch
from tqdm import tqdm
import pickle

def get_batch(data, i, seq_len=35):
    seq_len = min(seq_len, data.size(1) - 1 - i)
    data_seq = data[:, i:i+seq_len]
    target_seq = data[:, i+1:i+1+seq_len]
    return data_seq, target_seq

device = "cuda" if torch.cuda.is_available() else "cpu"  # Device 설정

def train_model(model, train_data, optimizer, criterion, scheduler, num_epochs, save_path, batch_size, seq_len, vocab_size):
    
    os.makedirs(save_path, exist_ok=True)
    epoch_losses = []   # Average loss per epoch
    batch_avg_losses = []  # Loss per batch
    iteration_losses = []  # Loss per iteration
    best_loss = float('inf') 

    for epoch in range(num_epochs):
        model.train()
        hidden = model.init_hidden(batch_size)
        total_loss = 0
        epoch_loss = 0  
        batch_count = 0  
        batch_loss_accum = 0  
        
        progress_bar = tqdm(range(0, train_data.size(1) - 1, 35), desc=f"Epoch {epoch+1}")
        for i in progress_bar:
            data, targets = get_batch(train_data, i)
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            output, hidden = model(data, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())  # Detach hidden states
            loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record losses
            batch_loss = loss.item()
            total_loss += batch_loss
            epoch_loss += batch_loss
            iteration_losses.append(batch_loss)  # Iteration 단위 손실 기록
            batch_loss_accum += batch_loss  # 배치 손실 누적
            batch_count += 1
            
            if batch_count % 20 == 0:
                batch_avg_loss = batch_loss_accum / 20
                batch_avg_losses.append(batch_avg_loss)
                batch_loss_accum = 0  
            
        # Update scheduler
        if scheduler:
            scheduler.step()
            
        # Calculate and save average epoch loss
        avg_epoch_loss = epoch_loss / batch_count
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f}")
        
        # Save loss data
        losses_save_path = os.path.join(save_path, "losses.pkl")
        with open(losses_save_path, "wb") as f:
            pickle.dump({
                "epoch_losses": epoch_losses, 
                "batch_avg_losses": batch_avg_losses, 
                "iteration_losses": iteration_losses
            }, f)
        print(f"Losses saved at {losses_save_path}")
        
        # Save the best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model_save_path = os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved with loss {best_loss:.4f} at {model_save_path}")
    
    return epoch_losses, batch_avg_losses, iteration_losses