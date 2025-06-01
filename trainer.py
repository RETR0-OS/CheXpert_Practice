import torch
from torch.nn import BCEWithLogitsLoss
from model import CheXpertSwinV2Model
from data_class import CheXpertDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from torchmetrics.classification import MultilabelLogAUC


def train_model(data_path, batch_size, epochs, pretrained_weights, save_model_dir, logs_dir, gpu, lr, optim, workers, val_epochs=5):

    best_auc = 0.0

    # Initialize dataset
    train_dataset = CheXpertDataset(data_path+"/train.csv")
    val_dataset = CheXpertDataset(data_path+"/valid.csv")


    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    # Initialize model
    model = CheXpertSwinV2Model(
        img_size=224,
        pretrained_weights=pretrained_weights,
        gpu=gpu,
        lr=lr,
        optim=optim
    )

    if optim == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif optim == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif optim == "sgd":
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optim}. Please use 'adamw', 'adam', or 'sgd'.")

    criterion = BCEWithLogitsLoss()  # Binary Cross-Entropy Loss with logits

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        loss = 0.0
        for batch_features, batch_labels in train_dataloader:
            # Move data to device
            batch_features = batch_features.to(model.device)
            batch_labels = batch_labels.to(model.device)

            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % val_epochs == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

            ## Run validation
            print("Initiating model validation...")
            val_loss, log_auc = validate_model(model, val_dataloader)
            with open(f"{logs_dir}/log.txt", "a") as log_file:
                log_file.write(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Log AUC: {log_auc:.4f}\n")

            if log_auc > best_auc:
                best_auc = log_auc
                print(f"New best Log AUC: {best_auc:.4f}. Saving model...")
                torch.save(model.state_dict(), f"{save_model_dir}/best_model.pth")

    with open(f"{logs_dir}/final_log.txt", "a") as log_file:
        log_file.write(f"Training completed. Best Log AUC: {best_auc:.4f}\n")

    ## Return the best model's state_dict
    with open(f"{save_model_dir}/best_model.pth", "rb") as f:
        best_model_state = torch.load(f)
        return best_model_state

def validate_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    criterion = BCEWithLogitsLoss()
    metric = MultilabelLogAUC(num_labels=14, average="macro", thresholds=None)
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(model.device)
            batch_labels = batch_labels.to(model.device)
            print(batch_labels)
            outputs = model(batch_features)
            log_auc = metric(outputs, batch_labels)
            loss = criterion(outputs, batch_labels.float())
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f} \t Log AUC: {log_auc:.4f}")
    return avg_loss, log_auc