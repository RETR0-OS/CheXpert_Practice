import os
import argparse
from model import CheXpertSwinV2Model
from data_class import CheXpertDataset
from trainer import train_model


def arg_parser():
    parser = argparse.ArgumentParser(description="CheXpert Practice")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index to use")
    parser.add_argument("--img_size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="ImageNet_1k | Random | Path to weights")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--optim" , type=str, default="adamw", choices=["adamw", "adam", "sgd"], help="Optimizer to use")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--workers", type=int, default=2, help="Number of workers for data loading")
    parser.add_argument("--val_epochs", type=int, default=5, help="Validation frequency in epochs")
    return parser.parse_args()


def main(args: argparse.Namespace):
    print("*"*50)
    print(args)
    print()
    print("*"*50, end="\n\n")

    logs_dir = os.path.join(os.path.abspath(args.output_dir),"logs")
    models_dir = os.path.join(os.path.abspath(args.output_dir),"models")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    print(f"Logs will be saved to: {logs_dir}")
    print(f"Models will be saved to: {models_dir}", end="\n\n")

    best_model_state = train_model(data_path=args.dataset_path,
                batch_size=args.batch_size,
                epochs=args.epochs,
                pretrained_weights=args.pretrained_weights,
                save_model_dir=models_dir,
                logs_dir=logs_dir,
                gpu=args.gpu,
                lr=args.lr,
                optim=args.optim,
                workers=args.workers,
                val_epochs=args.val_epochs)

    print("Training complete.")

if __name__ == "__main__":
    args = arg_parser()
    main(args)