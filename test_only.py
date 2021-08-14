#! /usr/bin/env python3

import random
import time
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from model import CNNModel

import json

from test import test
from plotting import save_loss_curve

import steves_utils.ORACLE.torch as ORACLE_Torch
from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    ALL_RUNS,
    serial_number_to_id
)


torch.set_default_dtype(torch.float64)

BEST_MODEL_PATH = sys.argv[1]
cuda = True
cudnn.benchmark = True






experiment_name = "(in)sanity"
lr = 0.0001
n_epoch = 25
batch_size = 50
source_distance = [2]
target_distance = source_distance
desired_serial_numbers = ALL_SERIAL_NUMBERS
alpha = 0.001
num_additional_extractor_fc_layers=1
patience = 10
seed = 1337
num_examples_per_device=10
window_stride=1
window_length=256 #Will break if not 256 due to model hyperparameters
desired_runs=ALL_RUNS


random.seed(seed)
torch.manual_seed(seed)

source_ds = ORACLE_Torch.ORACLE_Torch_Dataset(
                desired_serial_numbers=desired_serial_numbers,
                desired_distances=source_distance,
                desired_runs=desired_runs,
                window_length=window_length,
                window_stride=window_stride,
                num_examples_per_device=num_examples_per_device,
                seed=seed,  
                max_cache_size=0,
                transform_func=lambda x: (x["iq"], serial_number_to_id(x["serial_number"]), x["distance_ft"])
)

target_ds = ORACLE_Torch.ORACLE_Torch_Dataset(
                desired_serial_numbers=desired_serial_numbers,
                desired_distances=target_distance,
                desired_runs=desired_runs,
                window_length=window_length,
                window_stride=window_stride,
                num_examples_per_device=num_examples_per_device,
                seed=seed,  
                max_cache_size=0,
                transform_func=lambda x: (x["iq"], serial_number_to_id(x["serial_number"]), x["distance_ft"])
)

def wrap_datasets_in_dataloaders(datasets, **kwargs):
    dataloaders = []
    for ds in datasets:
        dataloaders.append(
            torch.utils.data.DataLoader(
                ds,
                **kwargs
            )
        )
    
    return dataloaders

source_train_ds, source_val_ds, source_test_ds = ORACLE_Torch.split_dataset_by_percentage(0.7, 0.15, 0.15, source_ds, seed)
target_train_ds, target_val_ds, target_test_ds = ORACLE_Torch.split_dataset_by_percentage(0.7, 0.15, 0.15, target_ds, seed)

source_train_dl, source_val_dl, source_test_dl = wrap_datasets_in_dataloaders(
    (source_train_ds, source_val_ds, source_test_ds),
    batch_size=batch_size,
    shuffle=True,
    num_workers=5,
    persistent_workers=True,
    prefetch_factor=10,
    pin_memory=True
)
target_train_dl, target_val_dl, target_test_dl = wrap_datasets_in_dataloaders(
    (target_train_ds, target_val_ds, target_test_ds),
    batch_size=batch_size,
    shuffle=True,
    num_workers=5,
    persistent_workers=True,
    prefetch_factor=10,
    pin_memory=True
)

my_net = CNNModel(num_additional_extractor_fc_layers)

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.CrossEntropyLoss()
loss_domain = torch.nn.L1Loss()
# loss_domain = torch.nn.MSELoss()


if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True


print("Loading best model {}".format(BEST_MODEL_PATH))
my_net = torch.load(BEST_MODEL_PATH)

source_test_label_accuracy, source_test_label_loss, source_test_domain_loss = \
    test(my_net, loss_class, loss_domain, source_test_dl)

target_test_label_accuracy, target_test_label_loss, target_test_domain_loss = \
    test(my_net, loss_class, loss_domain, target_test_dl)



out = ""
out += "Experiment name: {}\n".format(
    experiment_name
)

out += "Source Test Label Acc: {test_label_acc}, Source Test Label Loss: {test_label_loss}, Source Test Domain Loss: {test_domain_loss}\n".format(
    test_label_acc=source_test_label_accuracy,
    test_label_loss=source_test_label_loss,
    test_domain_loss=source_test_domain_loss,
)

out += "Target Test Label Acc: {test_label_acc}, Target Test Label Loss: {test_label_loss}, Target Test Domain Loss: {test_domain_loss}\n".format(
    test_label_acc=target_test_label_accuracy,
    test_label_loss=target_test_label_loss,
    test_domain_loss=target_test_domain_loss,
)

print(out)