import numpy as np
import os
from routes import CLASS_ID_CUT, CLASS_ID_TYPE

# def get_data(path):
#     img_paths = []
#     tool_labels = []
#     episod_labels = []

#     # iterate over all patient
#     for folder in sorted(os.listdir(path)):
#         if folder.isnumeric():
#             #print('folder: ', folder)
#             folder_path = os.path.join(path, folder)
#             tool_dict = get_label(path, folder)
#             # a dict whose key is the frame index and value is the label

#             # image paths and labels per patient
#             frames = []
#             tool_label = []
#             # iterate over all the frames
#             for frame_i in sorted(os.listdir(folder_path)):
#                 nb = int(frame_i[5:frame_i.find('.')])  # get frame index
#                 if nb in tool_dict:  # if this frame has a label
#                     img_path = os.path.join(folder_path, frame_i)
#                     frames.append(img_path)
#                     tool_label.append(tool_dict[nb])

#             img_paths.append(frames)
#             tool_labels.append(tool_label)

#     return img_paths, tool_labels, episod_labels

# def get_label(path, folder):
#     tool_label = {}
#     episod_label = {}
#     list_nb = 0
#     with open(os.path.join(path, f'{folder}.txt'), 'r') as file:
#         label = file.read()
#     i = 0
#     while i < len(label):
#         if label[i].isnumeric():
#             j = i
#             while label[j].isnumeric():
#                 j += 1
#             res = int(label[i:j])
#             i = j
#             if list_nb % 3 == 0:
#                 key = res
#             elif list_nb % 3 == 1:
#                 if key in tool_label and tool_label[key] != res:
#                     print('error')
#                 tool_label[key] = res
#             elif list_nb % 3 == 2:
#                 if key in episod_label and episod_label[key] != res:
#                     print('error')
#                 episod_label[key] = res
#             list_nb += 1
#         else:
#             i += 1

#     return tool_label

# def divide_data(path, transform):
#     img_paths, tool_labels, episod_labels = get_data(path)

#     ############# split into training and test set ##############
#     train_ind = [1, 2, 3, 4]

#     train_imgs = [x for i, x in enumerate(img_paths) if i in train_ind]
#     train_imgs = np.concatenate(train_imgs, axis=0)
#     train_tool = [x for i, x in enumerate(tool_labels) if i in train_ind]
#     train_tool = np.concatenate(train_tool, axis=0)

#     ########### balance the dataset to have nb_tool = nb_background #############
#     tool_idxs = np.arange(len(train_tool))[train_tool != 0]
#     back_idxs = np.arange(len(train_tool))[train_tool == 0]

#     np.random.shuffle(back_idxs)
#     np.random.shuffle(tool_idxs)
#     print('first background ind is: ', back_idxs[0])
#     print('first tool ind is: ', tool_idxs[0])

#     back_idxs = back_idxs[:len(tool_idxs)]
#     new_inds = np.concatenate((tool_idxs, back_idxs))
#     np.random.shuffle(new_inds)
#     print('first element of new_inds: ', new_inds[0])

#     train_tool = train_tool[new_inds]
#     train_imgs = train_imgs[new_inds]

#     print('dataset length: ', len(train_tool))
#     return DataHandler(path=train_imgs, label=train_tool, transform=transform)

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
