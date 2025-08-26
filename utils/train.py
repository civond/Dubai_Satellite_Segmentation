import torch
from tqdm import tqdm



def train_fn(device, loader, model, optimizer, loss_fn, scaler):
    model.train()  # make sure model is in training mode
    loop = tqdm(loader)
    
    for batch_idx, (data, labels) in enumerate(loop):
        data = data.to(device)
        labels = labels.to(device)

        # forward pass
        with torch.cuda.amp.autocast():
            predictions = model(data)['out']  # <-- extract tensor
            loss = loss_fn(predictions, labels)

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

"""def train_fn(device, loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    
    for batch_idx, (data, labels) in enumerate(loop):
        data = data.to(device=device)
        labels = labels.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, labels)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())"""