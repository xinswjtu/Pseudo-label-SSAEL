import argparse
import numpy as np
import torch
import random
from torch import nn


def set_seed(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def initialize_weights(m):
    for module in m.modules():
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(module.weight, 0.0, 0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def model_test(model, data_dl, devcie):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_dl:
            data, target = data.to(devcie), target.to(devcie)
            _, s_output =  model(data)
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target).item()
    acc = correct / len(data_dl.dataset)
    return acc


def get_label_data(train_x, train_y, n_class, n_label_data, is_shuffle=True):
    '''
    :param train_x: data
    :param train_y: label
    :param n_class: category
    :param n_label_data: number of every class
    :param is_shuffle:
    :return:
    '''
    selected_indices = []

    for i in range(n_class):
        indices = np.where(train_y == i)[0]
        selected_for_label = np.random.choice(indices, n_label_data, replace=False)
        selected_indices.extend(selected_for_label)

    all_indices = np.arange(train_x.shape[0])
    remaining_indices = np.setdiff1d(all_indices, selected_indices)

    # labeled datasets
    label_x = train_x[selected_indices]
    label_y = train_y[selected_indices]

    # training datasets
    train_x = train_x[remaining_indices]
    train_y = train_y[remaining_indices]

    # shuffle
    if is_shuffle:
        shuffled_indices = np.random.permutation(len(label_y))
        label_x = label_x[shuffled_indices]
        label_y = label_y[shuffled_indices]

        shuffled_indices = np.random.permutation(len(train_y))
        train_x = train_x[shuffled_indices]
        train_y = train_y[shuffled_indices]
    return train_x, train_y, label_x, label_y


class Metric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')