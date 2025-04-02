import torch
from BYOL import BYOL
# from byol_pytorch import BYOL

from torchvision import models
from routes import PRINT_PATH
from dataset_utils import prepare_dataset
import numpy as np
from utils import set_random
import os

dataset_img_size = {
    'pascal_VOC': 512,
    'auris': 220,
    'intuitive': 224,
    'cityscapes': 769,
}

if __name__ == '__main__':
    # set GPU number to 1 in the bash file ##
    if os.path.isfile(PRINT_PATH):
        os.remove(PRINT_PATH)

    dataset_name = 'auris'
    IMG_SIZE = dataset_img_size[dataset_name]
    BATCH_SIZE = 16
    workers = 4 # nb of cpus
    SIMSIAM = False

    resnet = models.resnet50(pretrained=True)

    learner = BYOL(
        resnet,
        image_size = IMG_SIZE,
        hidden_layer = 'avgpool',
        use_momentum = not SIMSIAM,
    )

    dataloader, dataset_sampler = prepare_dataset(dataset_name, BATCH_SIZE, workers, distributed=False)

    _, _, names = next(iter(dataloader))
    with open(PRINT_PATH, "a") as f:
        # f.write(f'folders: {sorted(train_data.keys())}, {len(train_data.keys())}\n')
        f.write(f'train dataset length {len(dataloader.dataset)}\n')
        f.write(f'first dataset sample: {names[0]}\n')
        f.write(f'train dataloader length: {len(dataloader)}, bs: {BATCH_SIZE}\n')

    pretrained_folder = 'simSiam_ImageNet_finetune' # BYOL, simSiam_ImageNet_finetune
    save_root = f'/storage/workspaces/artorg_aimi/ws_00000/fei/{pretrained_folder}/{dataset_name}'
    checkpoint_path = f'{save_root}/checkpoints/best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(0) if torch.cuda.is_available() else storage.cpu())
    learner.online_encoder.load_state_dict(checkpoint['online_encoder'])
    learner.cuda()
    learner.eval()

    if not os.path.exists(f'{save_root}/features'):
        os.makedirs(f'{save_root}/features', exist_ok=True)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    learner.online_encoder.net.layer4.register_forward_hook(get_activation("features"))

    with torch.no_grad():
        for images, y, names in dataloader:
            images = images.cuda()
            _ = learner.online_encoder(images) # this is the feature after the projection head
            for i, name in enumerate(names):
                seq, frame_nb = name.split('/')
                if not os.path.exists(f'{save_root}/features/{seq}'):
                    os.makedirs(f'{save_root}/features/{seq}', exist_ok=True)
                feature = activation['features'][i].cpu()
                
                torch.save(feature, f'{save_root}/features/{name}.pth')
       
