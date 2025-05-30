import os
import argparse


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
