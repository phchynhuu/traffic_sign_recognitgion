import argparse
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision.models import resnet18


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model for traffic sign recognition')
    parser.add_argument('--model', type=str, default='yolov9c.pt', help='Path to model (local file or URL)')
    parser.add_argument('--data', type=str, default='./datasets/data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., "cuda" or "cpu")')
    parser.add_argument('--output', type=str, default='runs/train', help='Output directory for results')
    parser.add_argument('--cache', action='store_true', help='Cache images for faster training')
    parser.add_argument('--amp', action='store_true', help='Use AMP mixed precision')
    parser.add_argument('--use_resnet', action='store_true', help='Replace YOLO head with ResNet')
    return parser.parse_args()


def replace_yolo_head_with_resnet(model):
    print("Replacing YOLO head with ResNet")
    resnet = resnet18(pretrained=True)
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, model.model[-1].nc)  # Match YOLO's number of classes

    # Replace the YOLO head layer with ResNet
    model.model[-1] = resnet
    return model


def train_yolo(args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False  # Set to False for consistent input sizes

    try:
        print(f"Loading model from: {args.model}")
        model = YOLO(args.model)

        if args.use_resnet:
            model = replace_yolo_head_with_resnet(model)

        print(f"Starting training with {args.epochs} epochs...")
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            cache=args.cache,
            amp=args.amp,
        )

        print(f"Training completed. Results saved in {args.output}")

    except RuntimeError as e:
        print(f"Runtime error: {str(e)}")
        torch.cuda.empty_cache()
        raise

    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    args = parse_args()
    train_yolo(args)
