import os
import torch
import pickle
from tqdm import tqdm
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model, train_loader, optimizer, criterion, scheduler, num_epochs, save_path):
    model.to(device)  # Move model to the appropriate device
    best_loss = float('inf')
    epoch_losses = []  # Average loss per epoch
    batch_losses = []  # Loss per batch
    iteration_losses = []  # Loss per iteration

    os.makedirs(save_path, exist_ok=True)  # Create directory for saving models and losses
    losses_save_path = os.path.join(save_path, "losses.pkl")  # Path to save loss data

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0  # Cumulative loss for the current epoch

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")  # Progress bar

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()  # Clear gradients
            outputs = model(images)  # Compute model outputs
            loss = criterion(outputs, labels)  # Calculate loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Record losses
            batch_loss = loss.item()
            batch_losses.append(batch_loss)  # Save batch loss
            iteration_losses.append(batch_loss)  # Save iteration loss
            epoch_loss += batch_loss

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Calculate and save average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)

        # Print epoch metrics
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}")

        # Save loss data
        with open(losses_save_path, "wb") as f:
            pickle.dump({
                "epoch_losses": epoch_losses,
                "batch_losses": batch_losses,
                "iteration_losses": iteration_losses
            }, f)
        print(f"Losses saved at {losses_save_path}")

        # Save the best model
        if best_loss > avg_epoch_loss:
            best_loss = avg_epoch_loss
            model_save_path = os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

    return epoch_losses, batch_losses, iteration_losses
