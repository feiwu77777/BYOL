import torch
from BYOL import BYOL
# from byol_pytorch import BYOL

from torchvision import models
from routes import PRINT_PATH
from dataset_utils import prepare_dataset
import numpy as np
from utils import set_random
import os

def load_simSiam_ImageNet(model):
    weight_path = '../pretrained_models/simSiam_ImageNet/checkpoint_0099.pth.tar'
    checkpoint = torch.load(weight_path)
    weight = checkpoint['state_dict']
    new_weight = {}
    for k, v in weight.items():
        new_weight[k.replace('module.encoder.', '')] = v

    logs = model.load_state_dict(new_weight, strict=False)
    return model, logs

if __name__ == '__main__':
    if os.path.isfile(PRINT_PATH):
        os.remove(PRINT_PATH)
    
    IMG_SIZE = 512
    BATCH_SIZE = 32 # mak it 32 for cityscapes
    workers = 4 # nb of cpus
    EPOCHS = 200
    SEED = 0
    SIMSIAM = True

    resnet = models.resnet50(pretrained=True)
    ### load simSiam model ###
    resnet, logs = load_simSiam_ImageNet(resnet)
    with open(PRINT_PATH, "a") as f:
        f.write(f'simSiam model loaded - logs: {logs}\n')

    learner = BYOL(
        resnet,
        image_size = IMG_SIZE,
        hidden_layer = 'avgpool',
        use_momentum = not SIMSIAM,
    )

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    set_random(SEED)
    dataset_name = 'intuitive'
    dataloader, dataset_sampler = prepare_dataset(dataset_name, BATCH_SIZE, workers, distributed=False, SEED=SEED)

    _, _, names = next(iter(dataloader))
    with open(PRINT_PATH, "a") as f:
        # f.write(f'folders: {sorted(train_data.keys())}, {len(train_data.keys())}\n')
        f.write(f'train dataset length {len(dataloader.dataset)}\n')
        f.write(f'first dataset sample: {names[0]}\n')
        f.write(f'train dataloader length: {len(dataloader)}, bs: {BATCH_SIZE}\n')

    best_loss = np.inf
    for epoch in range(EPOCHS):
        # with open(PRINT_PATH, "a") as f:
        #     f.write(f'--- Epoch: {epoch}\n')
        epoch_loss = 0
        for images, labels, names in dataloader:
            # with open(PRINT_PATH, "a") as f:
            #     f.write(f'batch: {names}\n')
            #     f.write(f'batch shape: {images.shape}\n')
            loss = learner(images)
            epoch_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

            if not SIMSIAM:
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
                    'online_encoder': learner.online_encoder.state_dict(),
                    'online_predictor': learner.online_predictor.state_dict()}
                , './results/checkpoints/best_model.pt')

        torch.save(
                {'epoch': epoch,
                'online_encoder': learner.online_encoder.state_dict(),
                'online_predictor': learner.online_predictor.state_dict()}
            , './results/checkpoints/latest_model.pt')
