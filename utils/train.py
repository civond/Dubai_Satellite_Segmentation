import torch
from tqdm import tqdm

def train_fn(device, loader, model, optimizer, loss_fn, scaler):
    model.train()  # make sure model is in training mode
    loop = tqdm(loader)
    
    total_loss = 0  
    for batch_idx, (data, labels) in enumerate(loop):
        data = data.to(device)
        labels = labels.to(device)

        # forward pass
        with torch.amp.autocast('cuda'):
            predictions = model(data)['out']  # <-- extract tensor
            loss = loss_fn(predictions, labels)

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()  
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader) 

    return avg_loss