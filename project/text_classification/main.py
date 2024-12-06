from transformers import AutoModelForSequenceClassification
from data_utils import load_tokenizer, load_data, calculate_num_training_steps
from train import train_model
from eval import evaluate_model
from torch.optim import lr_scheduler
import torch
import os

def main(optimizer_name, use_scheduler=False, use_warmup=False, num_warmup_steps=1500, power=1.0, min_lr=0.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    tokenizer = load_tokenizer()
    num_epochs = 3
    
    train_loader, test_loader = load_data(tokenizer, batch_size=32)

    model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=4)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer_settings = {
        "adam": {"optimizer": torch.optim.Adam, "lr": 5e-5, "weight_decay": 0.0001},
        "sgd": {"optimizer": torch.optim.SGD, "lr": 1e-2, "weight_decay": 0.0001},
        "rmsprop": {"optimizer": torch.optim.RMSprop, "lr": 1e-3, "weight_decay": 0.0001},
        "adagrad": {"optimizer": torch.optim.Adagrad, "lr": 1e-2, "weight_decay": 0.0001},
        "adamw": {"optimizer": torch.optim.AdamW, "lr": 5e-5, "weight_decay": 0.01}    
    }
    
    if optimizer_name.lower() not in optimizer_settings:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported. Choose from: {list(optimizer_settings.keys())}")
    
    optimizer_config = optimizer_settings[optimizer_name.lower()]
    optimizer = optimizer_config["optimizer"](
        model.parameters(),
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"]
    )
    
    total_iters = calculate_num_training_steps(train_loader, num_epochs)

    scheduler = None
    if use_scheduler:
        if use_warmup:
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                else:
                    remaining_iters = total_iters - num_warmup_steps
                    poly_decay = (1 - (current_step - num_warmup_steps) / remaining_iters) ** power
                    return max(poly_decay, min_lr / optimizer_config["lr"])
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    save_path = f"./model_checkpoints_{optimizer_name}"
    if use_scheduler:
        save_path += "_scheduler"
    if use_warmup:
        save_path += "_warmup"

    train_model(
        model, train_loader, optimizer, criterion, scheduler, num_epochs=num_epochs,
        save_path=save_path
    )
    
    model.load_state_dict(torch.load(os.path.join(save_path, "best_model.pth")))
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

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