import os
import torch
import pickle
from tqdm import tqdm
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model, train_loader, optimizer, criterion, scheduler, num_epochs, save_path):
    model.to(device)  
    best_loss = float('inf')
    epoch_losses = [] 
    batch_losses = []  
    iteration_losses = [] 

    os.makedirs(save_path, exist_ok=True) 
    losses_save_path = os.path.join(save_path, "losses.pkl")  

    for epoch in range(num_epochs):
        model.train()  
        epoch_loss = 0 

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}") 
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()  
            outputs = model(input_ids, attention_mask)  
            loss = criterion(outputs.logits, labels)  
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            batch_losses.append(batch_loss)  
            iteration_losses.append(batch_loss)  
            epoch_loss += batch_loss

        if scheduler:
            scheduler.step()

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)

        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}")

        with open(losses_save_path, "wb") as f:
            pickle.dump({
                "epoch_losses": epoch_losses,
                "batch_losses": batch_losses,
                "iteration_losses": iteration_losses
            }, f)
        print(f"Losses saved at {losses_save_path}")
        
        if best_loss > avg_epoch_loss:
            best_loss = avg_epoch_loss
            model_save_path = os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")
    
    return epoch_losses, batch_losses, iteration_losses
