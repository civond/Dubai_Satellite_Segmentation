import json
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

from utils.create_transforms import create_transforms
from utils.get_loader import get_loader
from utils.train import train_fn
from utils.checkpoints import *

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_WORKERS = 4
IMAGE_HEIGHT = 256  
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN = True

DATA_DIR = "data/"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    # Reset final layer to 6 classes
    model.aux_classifier[4] = nn.Conv2d(
        in_channels=model.aux_classifier[4].in_channels,  # 10
        out_channels=6,                                    # new number of classes
        kernel_size=1
    )
    model.to(DEVICE)

    transforms = create_transforms(IMAGE_HEIGHT, IMAGE_WIDTH)
    loader = get_loader(DATA_DIR, 
                        BATCH_SIZE, 
                        transforms,
                        NUM_WORKERS, 
                        TRAIN, 
                        PIN_MEMORY)
        
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch: {epoch}")

        train_fn(device, loader, model, optimizer, loss_fn, scaler)

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)


if __name__ == "__main__":
    main()