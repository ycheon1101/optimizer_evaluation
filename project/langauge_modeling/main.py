import argparse
import torch
from model import LSTMLanguageModel
from data_utils import load_tokenizer, load_data, calculate_num_training_steps
from train import train_model
from eval import evaluate_with_perplexity, load_model
import torch.optim.lr_scheduler as lr_scheduler
import os

def main(optimizer_name, use_scheduler=False, use_warmup=False, num_warmup_steps=1500, power=1.0, min_lr=0.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 공통 device 설정
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.vocab_size  # 공통 vocab_size 설정
    
    train_data, valid_data, test_data = load_data(tokenizer, batch_size=20)
    
    model = LSTMLanguageModel(vocab_size, embed_size=128, hidden_size=256, num_layers=2)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Optimizer configurations
    optimizer_settings = {
        "adam": {"optimizer": torch.optim.Adam, "lr": 5e-5, "weight_decay": 0.0001},
        "sgd": {"optimizer": torch.optim.SGD, "lr": 1e-2, "weight_decay": 0.0001},
        "rmsprop": {"optimizer": torch.optim.RMSprop, "lr": 1e-3, "weight_decay": 0.0001},
        "adagrad": {"optimizer": torch.optim.Adagrad, "lr": 1e-2, "weight_decay": 0.0001},
        "adamw": {"optimizer": torch.optim.AdamW, "lr": 1e-3, "weight_decay": 0.01}    
    }
    
    # Check for a valid optimizer
    if optimizer_name.lower() not in optimizer_settings:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported. Choose from: {list(optimizer_settings.keys())}")
    
    optimizer_config = optimizer_settings[optimizer_name.lower()]
    optimizer = optimizer_config["optimizer"](
        model.parameters(),
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"]
    )

    # Calculate total training steps
    total_iters = calculate_num_training_steps(train_data, seq_len=35, num_epochs=3)

    # Scheduler configuration
    scheduler = None
    if use_scheduler:
        if use_warmup:
            # Define warmup scheduler
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    # Warmup phase
                    return float(current_step) / float(max(1, num_warmup_steps))
                else:
                    # Poly Decay phase
                    remaining_iters = total_iters - num_warmup_steps
                    poly_decay = (1 - (current_step - num_warmup_steps) / remaining_iters) ** power
                    return max(poly_decay, min_lr / optimizer_config["lr"])
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Save path setup
    save_path = f"./model_checkpoints_{optimizer_name}"
    if use_scheduler:
        save_path += "_scheduler"
    if use_warmup:
        save_path += "_warmup"

    # Train the model
    train_model(model, train_data, optimizer, criterion, scheduler, num_epochs=3,
                save_path=save_path, batch_size=20, seq_len=35, vocab_size=vocab_size)
    
    # Evaluate the model
    model = load_model(LSTMLanguageModel, os.path.join(save_path, "best_model.pth"), device, vocab_size, embed_size=128, hidden_size=256, num_layers=2)
    model.to(device)
    test_loss, test_perplexity = evaluate_with_perplexity(model, test_data, criterion, batch_size=20, seq_len=35, vocab_size=vocab_size)
    print(f"Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, required=True, help="Optimizer to use: adam, sgd, rmsprop, adagrad, adamw")
    parser.add_argument("--use_scheduler", action="store_true", help="Use LR scheduler if specified")
    parser.add_argument("--use_warmup", action="store_true", help="Use LR warmup if specified")
    parser.add_argument("--num_warmup_steps", type=int, default=1500, help="Number of warmup steps")
    parser.add_argument("--power", type=float, default=1.0, help="Power for poly decay")
    parser.add_argument("--min_lr", type=float, default=0.0, help="Minimum learning rate for poly decay")
    args = parser.parse_args()
    main(
        args.optimizer,
        use_scheduler=args.use_scheduler,
        use_warmup=args.use_warmup,
        num_warmup_steps=args.num_warmup_steps,
        power=args.power,
        min_lr=args.min_lr
    )