#! /usr/bin/env python3

import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from model import CNNModel
from test import test

cuda = True
cudnn.benchmark = True
lr = 1e-4
batch_size = 1024
n_epoch = 100
model_root = "./model"

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

from steves_utils import utils
from torch_dataset_accessor.torch_windowed_shuffled_dataset_accessor import get_torch_windowed_shuffled_datasets

# class DummyDataset(torch.utils.data.Dataset):
#     def __init__(self, digit, c):
#         # self.t = torch.ones(2, 224).to(device)
#         # self.t = torch.ones(3, 28, 28)
#         self.t = torch.ones(2, 128)
#         self.t = self.t * digit
#         self.c = c

#     def __getitem__(self, index):
#         return (
#             self.t,
#             self.c
#         )

#     def __len__(self):
#         return 100000

#     @property
#     def num_classes(self):
#         # raise Exception("Really?")
#         return 20

source_distance = 50
target_distance = 14

source_ds_path = "{datasets_base_path}/automated_windower/windowed_EachDevice-200k_batch-100_stride-20_distances-{distance}".format(
    datasets_base_path=utils.get_datasets_base_path(), distance=source_distance
)

target_ds_path = "{datasets_base_path}/automated_windower/windowed_EachDevice-200k_batch-100_stride-20_distances-{distance}".format(
    datasets_base_path=utils.get_datasets_base_path(), distance=target_distance
)

datasets_source = get_torch_windowed_shuffled_datasets(source_ds_path, take=102400)
datasets_target = get_torch_windowed_shuffled_datasets(target_ds_path, take=102400)

train_ds_source = datasets_source["train_ds"]
train_ds_target = datasets_target["train_ds"]

test_ds_source = datasets_source["test_ds"]
test_ds_target = datasets_target["test_ds"]

dataloader_source = torch.utils.data.DataLoader(
    dataset=train_ds_source,
    batch_size=batch_size,
    # shuffle=True,
    # num_workers=8
)

dataloader_target = torch.utils.data.DataLoader(
    dataset=train_ds_target,
    batch_size=batch_size,
    # shuffle=True,
    # num_workers=8
)

my_net = CNNModel()

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training
best_accu_t = 0.0
for epoch in range(n_epoch):

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    for i in range(len_dataloader):

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # print("Alpha", alpha)

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label = data_source

        my_net.zero_grad()
        batch_size = len(s_label)

        domain_label = torch.zeros(batch_size).long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            domain_label = domain_label.cuda()


        class_output, domain_output = my_net(input_data=s_img, alpha=alpha)


        err_s_label = loss_class(class_output, s_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # training model using target data
        data_target = data_target_iter.next()
        t_img, _ = data_target

        batch_size = len(t_img)

        domain_label = torch.ones(batch_size).long()

        if cuda:
            t_img = t_img.cuda()
            domain_label = domain_label.cuda()

        _, domain_output = my_net(input_data=t_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
        sys.stdout.flush()
        torch.save(my_net, '{0}/mnist_mnistm_model_epoch_current.pth'.format(model_root))

    print('\n')
    accu_s = test(test_ds_source)
    print('Accuracy of the %s dataset: %f' % ('Source', accu_s))
    accu_t = test(test_ds_target)
    print('Accuracy of the %s dataset: %f\n' % ('Target', accu_t))
    if accu_t > best_accu_t:
        best_accu_s = accu_s
        best_accu_t = accu_t
        torch.save(my_net, '{0}/mnist_mnistm_model_epoch_best.pth'.format(model_root))

print('============ Summary ============= \n')
print('Accuracy of the %s dataset: %f' % ('mnist', best_accu_s))
print('Accuracy of the %s dataset: %f' % ('mnist_m', best_accu_t))
print('Corresponding model was save in ' + model_root + '/mnist_mnistm_model_epoch_best.pth')