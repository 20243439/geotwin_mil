import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "moco-v3"))

import argparse
import torch
from feature_extractor import MocoTrainer, MocoFeatureExtractor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoCo Training & Feature Extraction")
    
    # Required argument
    parser.add_argument("--mode", type=str, choices=["train", "extract"], required=True,
                        help="Choose 'train' for training, 'extract' for feature extraction")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory of the satellite images")

    # Optional arguments
    parser.add_argument("--map_dir", type=str,
                        help="Root directory of the topographical maps (only for training)")
    parser.add_argument("--model_path", type=str,
                        help="Path to trained (or pre-trained) MoCo model (.pth)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default=64)")

    # Add the missing arguments below
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default=10)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default=1e-4)")
    parser.add_argument("--rank", type=int, default=4,
                        help="Rank for LoRA adaptation (default=4)")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to a pretrained MoCo checkpoint (.pth.tar)")
    parser.add_argument("--save_path", type=str, default="moco_lora_model.pth",
                        help="Save path for the LoRA-finetuned model (default=moco_lora_model.pth)")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        # Pass your newly added arguments to the trainer
        trainer = MocoTrainer(
            root_dir=args.root_dir,
            map_dir=args.map_dir,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            rank=args.rank
        )
        trainer.train()
        trainer.save_model(path=args.save_path)

    elif args.mode == "extract":
        # For extraction, pass whichever arguments you need
        extractor = MocoFeatureExtractor(
            model_path=args.model_path,  # or args.save_path if you want the newly saved model
            root_dir=args.root_dir,
            device=device,
            batch_size=args.batch_size
        )
        extractor.save_features()
