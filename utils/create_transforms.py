import torch
from torchvision.transforms import v2
#from utils.get_loader import get_loader


# Create transforms object
def create_transforms(IMAGE_HEIGHT, IMAGE_WIDTH, TRAIN):
    # Train
    if TRAIN == True:
        transform = v2.Compose([
                #v2.ToTensor(),
                v2.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.1),
                v2.RandomRotation(degrees=35),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] # variance is std**2
                )
                ])
    
    # Inference
    if TRAIN == False:
        transform = v2.Compose([
                #v2.ToTensor(),
                v2.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] # variance is std**2
                )
                ])
        
    return transform