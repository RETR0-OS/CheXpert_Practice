import torch
from torch.nn import BCEWithLogitsLoss
from model import CheXpertSwinV2Model
from data_class import CheXpertDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from torcheval.metrics import BinaryAUROC
import os

def train_model(data_path, batch_size, epochs, pretrained_weights, save_model_dir, logs_dir, gpu, lr, optim, workers, val_epochs=5):

    best_auc = 0.0

    # Create directories if they don't exist
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

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

    f = open(f"{logs_dir}/log.txt", "w")  # Clear old log contents, if exists
    f.close()
    f = open(f"{logs_dir}/final_log.txt", "w")  # clear logs, if exists.
    f.close()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0

        for batch_features, batch_labels in train_dataloader:
            model.train()
            # Move data to device
            batch_features = batch_features.to(model.device)
            batch_labels = batch_labels.to(model.device)

            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.float())
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % val_epochs == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_dataloader):.4f}")

            ## Run validation
            print("Initiating model validation...")
            val_loss, auc = validate_model(model, val_dataloader)

            os.makedirs(logs_dir, exist_ok=True)
            os.makedirs(save_model_dir, exist_ok=True)

            with open(f"{logs_dir}/log.txt", "a") as log_file:
                log_file.write(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, AUC: {auc:.4f}\n")

            if auc > best_auc:
                best_auc = auc
                print(f"New best AUC: {best_auc:.4f}. Saving model...")
                torch.save(model.state_dict(), f"{save_model_dir}/best_model.pth")
    with open(f"{logs_dir}/final_log.txt", "a") as log_file:
        log_file.write(f"Training completed. Best AUC: {best_auc:.4f}\n")

    ## Return the best model's state_dict
    with open(f"{save_model_dir}/best_model.pth", "rb") as f:
        best_model_state = torch.load(f)
        return best_model_state

def validate_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    criterion = BCEWithLogitsLoss()
    metric = BinaryAUROC(num_tasks=14)  # Set num_tasks to the number of classes
    metric.to(model.device)

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(model.device)
            batch_labels = batch_labels.to(model.device)
            outputs = model(batch_features)

            loss = criterion(outputs, batch_labels.float())
            total_loss += loss.item()

            all_outputs.append(outputs.transpose(0,1)) #transpose to (14, batch_size)
            all_labels.append(batch_labels.transpose(0,1)) #transpose to (14, batch_size)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metric.update(all_outputs, all_labels)
    auc = metric.compute()
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f} \t AUC: {auc.mean().item():.4f}")
    return avg_loss, auc.mean().item()


# # Test train function
# if __name__ == "__main__":
#     # Example usage
#     best_model_state = train_model(
#         data_path="./dataset",
#         batch_size=32,
#         epochs=4,
#         pretrained_weights="ImageNet_1k",
#         save_model_dir="./Outputs/models",
#         logs_dir="./Outputs/logs",
#         gpu=0,
#         lr=2e-3,
#         optim="adamw",
#         workers=2,
#         val_epochs=1
#     )
#     print("Training complete. Best model state saved.")