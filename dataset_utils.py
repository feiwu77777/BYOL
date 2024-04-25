import numpy as np
import os
import torch
from routes import CLASS_ID_CUT, CLASS_ID_TYPE, AURIS_SEG_PATH, PASCAL_PATH
from dataHandlers import DataHandlerAurisSeg, DataHandlerPascal


def prepare_dataset(dataset_name, batch_size, workers, distributed=False):
    ## the data transformation is carried out in the learner class

    if dataset_name == 'auris':
        train_data, val_data, _ = divide_data_split_auris(AURIS_SEG_PATH, AURIS_SEG_PATH, num_val=0)
        dataset = DataHandlerAurisSeg(data_path=train_data, label_path=AURIS_SEG_PATH)
    
    elif dataset_name == 'pascal_VOC':
        with open(PASCAL_PATH + 'train.txt', 'r') as file:
            TRAIN_SEQ = file.read()
        img_path = PASCAL_PATH + 'images/'
        TRAIN_SEQ = TRAIN_SEQ.split('\n')
        TRAIN_SEQ = [img_path + t + '.jpg' for t in TRAIN_SEQ if len(t) > 0]
        dataset = DataHandlerPascal(img_path=img_path, sequences=TRAIN_SEQ)

    if distributed:
        dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=(dataset_sampler is None),
            num_workers=workers, pin_memory=True, sampler=dataset_sampler)
    else:
        dataset_sampler = None
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)
    return dataloader, dataset_sampler


def divide_data_split_auris(img_path, lab_path, num_val=15):
    data_split = {
        '69-00': '',
        '73-00': '',
        '73-01': '',
        '43-00': '',
        '73-02': '',
        '75-01': '',
        '42-00': '',
        '42-01': '',
        '77-02': '',
        '75-00': '',
        '48-01': '',
        '47-00': '',
        '45-00': '',
        '77-00': '',
        '48-00': '',

        '01-00': 'train',
        '01-01': 'train',
        '01-02': 'train',
        '01-03': 'train',
        '01-04': 'train',
        '01-05': '',
        '01-06': 'train',
        '01-07': 'train',

        '02-00': 'train',
        '02-01': 'train',

        '03-00': 'train',
        '03-01': 'train',
        '03-02': 'train',
        '03-03': 'train',
        '03-04': 'train',
        '03-05': 'train',
        '03-06': 'train',
        '03-07': 'train',

        '04-00': '',
        '04-01': '',
        '04-02': '',
        '04-03': '',
        '04-04': 'train',
        '04-05': '',
        '04-06': 'train',
        '04-07': 'train',

        '12-00': '',
        '12-01': '',
        '12-02': '',
        '12-03': 'train',
        '12-04': '',

        '13-00': 'train',
        '13-01': '',
        '13-02': 'train',
        '13-03': 'train',
        '13-04': 'train',

        '15-00': 'train',
        '15-01': 'train',
        '15-02': 'train',
        '15-03': '',
        '15-04': 'train',
        '15-05': 'train',

        '09-00': 'test',
        '09-02': 'test',
        '09-01': 'test',
        '08-03': 'test',
        '10-02': 'test',
        '11-04': 'test',
        '08-01': 'test',
        '08-06': 'test',
        '11-03': 'test',
        '10-03': 'test',
        '10-00': 'test',
        '11-00': 'test',
        '08-04': 'test',
        '08-05': 'test',
        '10-06': 'test',
        '10-01': 'test',
        '08-02': 'test',
        '10-05': 'test',
        '11-01': 'test',
        '08-00': 'test',
        '10-04': 'test',
        '08-08': 'test',
        '10-07': 'test',
        '11-02': 'test',
        '08-07': 'test',
        '08-09': 'test'
    }

    imgs_path = sorted(os.listdir(img_path))
    labels_path = sorted(os.listdir(lab_path))

    all_imgs_train = []
    all_imgs_test = []
    for name in imgs_path:
        if name[0] == '.':
            continue

        if name in data_split and data_split[name] != '':
            img_files = []
            files = [f for f in os.listdir(img_path + name) if f[0] != '.']
            files = sorted(files, key=lambda x: int(x[5:-4]))
            for file in files:
                if file[0] == '.':
                    continue
                img_files.append(img_path + name + '/' + file)

            if data_split[name] == 'train':
                all_imgs_train.append(img_files)
            elif data_split[name] == 'test':
                all_imgs_test.append(img_files)

    all_labels_train = []
    all_labels_test = []
    for name in labels_path:
        if name[0] == '.':
            continue
        if name in data_split and data_split[name] != '':
            label_files = []
            files = [f for f in os.listdir(lab_path + name) if f[0] != '.']
            files = sorted(files, key=lambda x: int(x[5:-4]))
            for file in files:
                if file[0] == '.':
                    continue
                label_files.append(lab_path + name + '/' + file)

            if data_split[name] == 'train':
                all_labels_train.append(label_files)
            elif data_split[name] == 'test':
                all_labels_test.append(label_files)

    assert len(all_labels_train) == len(all_imgs_train)
    assert len(all_labels_test) == len(all_imgs_test)

    indexes = np.arange(len(all_labels_train))
    np.random.shuffle(indexes)

    train_data = {}
    val_data = {}
    ind = len(lab_path + CLASS_ID_TYPE)
    for i, n in enumerate(indexes):
        if i < num_val:
            L = []
            for j, frame in enumerate(all_imgs_train[n]):
                L.append((frame, all_labels_train[n][j]))
            val_data[frame[ind - len(CLASS_ID_TYPE):ind -
                        len(CLASS_ID_CUT)]] = np.array(L)
        else:
            L = []
            for j, frame in enumerate(all_imgs_train[n]):
                L.append((frame, all_labels_train[n][j]))
            train_data[frame[ind - len(CLASS_ID_TYPE):ind -
                        len(CLASS_ID_CUT)]] = np.array(L)

    test_data = {}
    for i, frames in enumerate(all_imgs_test):
        L = []
        for j, frame in enumerate(frames):
            L.append((frame, all_labels_test[i][j]))
        test_data[frame[ind - len(CLASS_ID_TYPE):ind -
                        len(CLASS_ID_CUT)]] = np.array(L)
    return train_data, val_data, test_data
