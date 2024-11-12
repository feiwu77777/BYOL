from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import os
# import albumentations as A
import cv2
import glob
import torch

from utils import set_random
from routes import SKATEBOARD_PATH, AURIS_SEG_PATH

class DataHandlerIntuitive(Dataset):
    def __init__(self,
                 basepath: str,
                 sequences,
                 transform=None,
                 validation=False,
                 filter=None):
        super(DataHandlerIntuitive).__init__()
        self.img_path = []
        self.labels = []
        self.filter = filter
        self.validation = validation

        self.labels_path = basepath + 'labels/'
        images_path = basepath + 'images/'
        # load labels and frames
        ind_keyword = len(self.labels_path + 'seq_xx/')
        for seq in sequences:
            files = sorted(
                glob.glob(os.path.join(self.labels_path, seq, '*.png')))
            img_files = os.listdir(os.path.join(images_path, seq))
            assert len(files) > 0
            for f in files:
                if f[ind_keyword:] not in img_files:
                    continue
                if self.filter is None:
                    self.labels.append(f)
                    self.img_path.append(f.replace('labels', 'images'))

        self.transform = transform

    def open_imgs(self, left_path):
        img_left = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_RGB2BGR)
        if self.transform is not None:
            x1, x2 = self.transform(Image.fromarray(img_left))

        return x1, x2

    def __getitem__(self, item):
        left_path = self.img_path[item]
        if self.validation:
            set_random(item)
        x1, x2 = self.open_imgs(left_path)
        return [x1, x2], np.zeros((10, 10))

    def __len__(self):
        return len(self.img_path)


class DataHandlerYoutube(Dataset):
    def __init__(self,
                 data_path,
                 transform=None,
                 validation=False,
                 label_path=SKATEBOARD_PATH):
        self.data_path = data_path
        self.data_pool = np.concatenate(list(data_path.values()), axis=0)
        self.transform = transform
        self.validation = validation
        self.label_path = label_path

    def __getitem__(self, index):
        x = Image.open(self.data_pool[index])
        name = self.data_pool[index][len(self.label_path):]
        # name = self.data_pool[index].split('/')[-1]
        # name = name.split('.')[0]
        if self.validation:
            set_random(index)
        if self.transform is not None:
            x1, x2 = self.transform(x)

        return [x1, x2], np.zeros((10, 10)), [name, name]

    def __len__(self):
        return len(self.data_pool)


class DataHandlerAurisSeg(Dataset):
    def __init__(self, 
                 data_path, 
                 transform=None, 
                 label_path=AURIS_SEG_PATH, 
                 eval=False):
        self.data_path = data_path
        self.data_pool = np.concatenate(list(data_path.values()), axis=0)
        self.transform = transform
        self.label_path = label_path
        self.eval = eval

        self.normalize_op = transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )
        self.tensor_op = transforms.ToTensor()

    def __getitem__(self, index):
        img_path, lab_path = self.data_pool[index]
        x = Image.open(img_path)
        x = self.tensor_op(x)
        x = self.normalize_op(x)

        name = lab_path[len(self.label_path):]
        name = name.split('/')[0] + '/' + name.split('/')[1][len('frame'):-len('.png')]

        return x, np.zeros((10, 10)), name

    def __len__(self):
        return len(self.data_pool)


class DataHandlerUAVID(Dataset):
    def __init__(self,
                 basepath: str,
                 sequences,
                 transform=None,
                 validation=False,
                 filter=None):
        self.imgs = []
        self.filter = filter
        self.validation = validation

        self.img_path = basepath
        # load labels and frames
        for seq in sequences:
            # glob is for getting the global path from root to the file
            img_files =  sorted(os.listdir(self.img_path + seq))
            assert len(img_files) > 0
            for f in img_files:
                if f not in img_files:
                    continue
                if self.filter is None:
                    self.imgs.append(self.img_path + seq + '/' + f)

        self.transform = transform

    def open_imgs(self, img_path):
        img = Image.open(img_path)
        if self.transform is not None:
            x1, x2 = self.transform(img)
        
        return x1, x2

    def __getitem__(self, item):
        img_path = self.imgs[item]
        if self.validation:
            set_random(item)
        x1, x2 = self.open_imgs(img_path)
        return [x1, x2], np.zeros((10, 10))

    def __len__(self):
        return len(self.imgs)


class DataHandlerPascal(Dataset):
    def __init__(
        self,
        img_path,
        sequences,
        transform=None,
    ):
        super(DataHandlerPascal).__init__()
        self.img_path = img_path
        self.data_pool = []

        for frame_path in sequences:
            self.data_pool.append(
                (frame_path, frame_path.replace("images", "labels").replace("jpg", "png"))
            )
            
        self.data_pool = np.array(self.data_pool)
        self.transform = transform

        self.normalize_op = transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )
        self.tensor_op = transforms.ToTensor()

    def __getitem__(self, item):
        img_path, label_path = self.data_pool[item]
        x = Image.open(img_path)
        x = self.tensor_op(x)
        # x = self.normalize_op(x)

        name = img_path.split("/")[-2] + "/" + img_path.split("/")[-1].split(".")[0]
        return x, np.zeros((10, 10)), name

    def __len__(self):
        return len(self.data_pool)


class DataHandlerCityscapes(Dataset):
    def __init__(
        self,
        img_path,
        sequences,
    ):
        super(DataHandlerCityscapes).__init__()
        self.img_path = img_path
        self.data_pool = np.array(sequences)

        self.normalize_op = transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )
        self.tensor_op = transforms.ToTensor()

    def __getitem__(self, item):
        # ToDo check if dataloading and label decoding is a bottleneck, if yes perform pre-loading
        # set seed for consistent DA
        img_path = self.data_pool[item]
        name = img_path.split("/")[-2] + "/" + img_path.split("/")[-1].split(".")[0]

        img = Image.open(img_path)
        img = self.tensor_op(img)
        return img, np.zeros((10, 10)), name

    def __len__(self):
        return len(self.data_pool)


class DataHandlerIntuitive(Dataset):
    def __init__(self,
                 basepath: str,
                 sequences,):
        super(DataHandlerIntuitive).__init__()
        self.img_path = []
        self.labels = []

        self.labels_path = basepath + 'labels/'
        images_path = basepath + 'images/'
        # load labels and frames
        ind_keyword = len(self.labels_path + 'seq_xx/')
        for seq in sequences:
            files = sorted(
                glob.glob(os.path.join(self.labels_path, seq, '*.png')))
            img_files = os.listdir(os.path.join(images_path, seq))
            assert len(files) > 0
            for f in files:
                if f[ind_keyword:] not in img_files:
                    continue
                self.labels.append(f)
                self.img_path.append(f.replace('labels', 'images'))

        self.normalize_op = transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )
        self.tensor_op = transforms.ToTensor()

    def __getitem__(self, item):
        img_path = self.img_path[item]
        x = Image.open(img_path).resize((512, 512))
        x = self.tensor_op(x)

        name = img_path.split("/")[-2] + "/" + img_path.split("/")[-1].split(".")[0]
        return x, np.zeros((10, 10)), name

    def __len__(self):
        return len(self.img_path)