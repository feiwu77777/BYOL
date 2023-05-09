import torch
from byol_pytorch import BYOL
from torchvision import models
from dataHandlers import DataHandlerAurisSeg
from torch.utils.data import DataLoader
from routes import AURIS_SEG_PATH, PRINT_PATH
from dataset_utils import divide_data_split_auris
import numpy as np
from utils import set_random
import os

if __name__ == '__main__':
    if os.path.isfile(PRINT_PATH):
        os.remove(PRINT_PATH)
    
    IMG_SIZE = 220
    BATCH_SIZE = 128
    EPOCHS = 200
    SEED = 0
    SIMSIAM = True

    resnet = models.resnet50(pretrained=True)

    learner = BYOL(
        resnet,
        image_size = IMG_SIZE,
        hidden_layer = 'avgpool',
        use_momentum = not SIMSIAM,
    )

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    set_random(SEED)
    train_data, val_data, _ = divide_data_split_auris(AURIS_SEG_PATH, AURIS_SEG_PATH, num_val=0)
    
    dataset = DataHandlerAurisSeg(data_path=train_data, label_path=AURIS_SEG_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    with open(PRINT_PATH, "a") as f:
        f.write(f'folders: {sorted(train_data.keys())}, {len(train_data.keys())}\n')
        f.write(f'train dataset length {len(dataset)}\n')
        f.write(f'first dataset sample: {dataset.data_pool[0]}\n')
        f.write(f'train dataloader length: {len(dataloader)}, bs: {BATCH_SIZE}\n')

    best_loss = np.inf
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for images, labels, names in dataloader:
            loss = learner(images)
            epoch_loss += loss.item()    
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder
        
        epoch_loss /= len(dataloader)
        with open(PRINT_PATH, "a") as f:
            f.write(f'--- Epoch: {epoch}, loss: {epoch_loss}\n')
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            with open(PRINT_PATH, "a") as f:
                f.write(f'best loss saved: {best_loss}\n')
            # save your improved network
            torch.save(
                    {'epoch': epoch,
                    'state_dict': learner.online_encoder.state_dict()}
                , './results/checkpoints/online_encoder.pt')

