import torch
from BYOL import BYOL

from torchvision import models
from routes import PRINT_PATH
import numpy as np
from utils import set_random
from dataset_utils import prepare_dataset
import os

import torch.distributed as dist
import torch.multiprocessing as mp

def init_distributed_dataparallel():
    ngpus_per_node = torch.cuda.device_count()
    with open(PRINT_PATH, "a") as f:
        f.write(f'ngpus per node: {ngpus_per_node}')

    world_size = 1
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    world_size = ngpus_per_node * world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, world_size))

def main(gpu, ngpus_per_node, world_size):
    init_rank = 0
    rank = init_rank * ngpus_per_node + gpu

    dist_backend = 'nccl'
    dist_url = 'tcp://127.0.0.1:33330'
    dist.init_process_group(backend=dist_backend, init_method=dist_url,
                            world_size=world_size, rank=rank)

    with open(PRINT_PATH, "a") as f:
        f.write(f'Rank {rank} - Initialized process group\n')
    
    if dist.get_rank() == 0:
        if os.path.exists(PRINT_PATH):
            os.remove(PRINT_PATH)
        if not os.path.exists('./results/checkpoints'):
            os.makedirs('./results/checkpoints')
    
    IMG_SIZE = 769
    BATCH_SIZE = 16
    EPOCHS = 200
    SEED = 0
    SIMSIAM = False

    resnet = models.resnet50(pretrained=True)

    ## distributed training code
    torch.cuda.set_device(gpu)
    resnet.cuda(gpu)
    BATCH_SIZE = int(BATCH_SIZE / ngpus_per_node)
    workers = int((4 + ngpus_per_node - 1) / ngpus_per_node)

    learner = BYOL(
        resnet,
        image_size = IMG_SIZE,
        hidden_layer = 'avgpool',
        use_momentum = not SIMSIAM,
    )

    learner = torch.nn.parallel.DistributedDataParallel(learner, device_ids=[gpu], find_unused_parameters=not SIMSIAM)
    device = torch.device('cuda:{}'.format(gpu))
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    ## prepare datasets ##
    set_random(SEED)
    dataset_name = 'cityscapes'
    dataloader, dataset_sampler = prepare_dataset(dataset_name, BATCH_SIZE, workers, distributed=True)

    _, _, names = next(iter(dataloader))
    with open(PRINT_PATH, "a") as f:
        f.write(f'--- Rank {rank} ---\n')
        # f.write(f'folders: {sorted(train_data.keys())}, {len(train_data.keys())}\n')
        f.write(f'train dataset length {len(dataloader.dataset)}\n')
        f.write(f'first dataset sample: {names[0]}\n')
        f.write(f'train dataloader length: {len(dataloader)}, bs: {BATCH_SIZE}\n')

    best_loss = np.inf
    print_once = True
    for epoch in range(EPOCHS):
        dataset_sampler.set_epoch(epoch)
        
        with open(PRINT_PATH, "a") as f:
            f.write(f'Rank {rank} - Epoch: {epoch}\n')
        epoch_loss = 0
        for images, labels, names in dataloader:
            if print_once:
                with open(PRINT_PATH, "a") as f:
                    f.write(f'Rank {rank} - images: {images.shape}, labels: {labels.shape}\n')
                print_once = False
            images = images.to(device, non_blocking=True)
            loss = learner(images)
            epoch_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

            if not SIMSIAM:
                learner.module.update_moving_average() # update moving average of target encoder
        
        epoch_loss /= len(dataloader)
        with open(PRINT_PATH, "a") as f:
            f.write(f'--- Rank {rank} - Epoch: {epoch}, loss: {epoch_loss}\n')
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            with open(PRINT_PATH, "a") as f:
                f.write(f'Rank {rank} - best loss saved: {best_loss}\n')
            # save your improved network
            torch.save(
                    {'epoch': epoch,
                    'online_encoder': learner.module.online_encoder.state_dict(),
                    'online_predictor': learner.module.online_predictor.state_dict()}
                , './results/checkpoints/best_model.pt')

        torch.save(
                {'epoch': epoch,
                'online_encoder': learner.module.online_encoder.state_dict(),
                'online_predictor': learner.module.online_predictor.state_dict()}
            , './results/checkpoints/latest_model.pt')
        
if __name__ == '__main__':
    init_distributed_dataparallel()
