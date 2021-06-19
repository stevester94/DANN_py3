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

torch.set_default_dtype(torch.float64)


cuda = True
cudnn.benchmark = True
lr = 0.0001
n_epoch = 100
model_root = "./model"

manual_seed = 1337
random.seed(manual_seed)
torch.manual_seed(manual_seed)

from steves_utils import utils
from steves_utils.ORACLE.windowed_shuffled_dataset_accessor import Windowed_Shuffled_Dataset_Factory
import tensorflow as tf

batch_size = 64
ORIGINAL_BATCH_SIZE = 100

source_distance = 50
target_distance = 14

def apply_dataset_pipeline(datasets):
    """
    Apply the appropriate dataset pipeline to the datasets returned from the Windowed_Shuffled_Dataset_Factory
    """
    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]

    # train_ds = train_ds.map(
    #     lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
    #     num_parallel_calls=tf.data.AUTOTUNE,
    #     deterministic=True
    # )

    # val_ds = val_ds.map(
    #     lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
    #     num_parallel_calls=tf.data.AUTOTUNE,
    #     deterministic=True
    # )

    # test_ds = test_ds.map(
    #     lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
    #     num_parallel_calls=tf.data.AUTOTUNE,
    #     deterministic=True
    # )

    train_ds = train_ds.map(
        lambda x: (x["IQ"], x["serial_number_id"]),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    val_ds = val_ds.map(
        lambda x: (x["IQ"], x["serial_number_id"]),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda x: (x["IQ"], x["serial_number_id"]),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    train_ds = train_ds.unbatch()
    val_ds = val_ds.unbatch()
    test_ds = test_ds.unbatch()

    train_ds = train_ds.shuffle(100 * ORIGINAL_BATCH_SIZE, reshuffle_each_iteration=True)
    
    train_ds = train_ds.batch(batch_size)
    val_ds  = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    train_ds = train_ds.prefetch(100)
    val_ds   = val_ds.prefetch(100)
    test_ds  = test_ds.prefetch(100)

    return train_ds, val_ds, test_ds


def get_shuffled_and_windowed_from_pregen_ds(path):
    datasets = Windowed_Shuffled_Dataset_Factory(path)

    return apply_dataset_pipeline(datasets)



source_ds_path = "{datasets_base_path}/automated_windower/windowed_EachDevice-200k_batch-100_stride-20_distances-{distance}".format(
    datasets_base_path=utils.get_datasets_base_path(), distance=source_distance
)

target_ds_path = "{datasets_base_path}/automated_windower/windowed_EachDevice-200k_batch-100_stride-20_distances-{distance}".format(
    datasets_base_path=utils.get_datasets_base_path(), distance=target_distance
)



train_ds_source, val_ds_source, test_ds_source = get_shuffled_and_windowed_from_pregen_ds(source_ds_path)
train_ds_target, val_ds_target, test_ds_target = get_shuffled_and_windowed_from_pregen_ds(target_ds_path)



print("Unfortunately have to calculate the length of the source dataset by iterating over it. Standby...")
num_batches_in_train_ds_source = 0
for i in train_ds_source:
    num_batches_in_train_ds_source += 1
print("Done. Source Train DS Length:", num_batches_in_train_ds_source)

print("Unfortunately have to calculate the length of the source dataset by iterating over it. Standby...")
num_batches_in_train_ds_target = 0
for i in train_ds_target:
    num_batches_in_train_ds_target += 1
print("Done. Target Train DS Length:", num_batches_in_train_ds_target)

# print("We are hardcoding DS length!")
# num_batches_in_train_ds_source = 6250
# num_batches_in_train_ds_target = 6250

my_net = CNNModel()

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.CrossEntropyLoss()
loss_domain = torch.nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training
best_accu_t = 0.0
for epoch in range(n_epoch):

    len_dataloader = min(num_batches_in_train_ds_source, num_batches_in_train_ds_target)
    data_source_iter = train_ds_source.as_numpy_iterator()
    data_target_iter = train_ds_target.as_numpy_iterator()

    for i in range(len_dataloader):

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # print("Alpha", alpha)

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label = data_source

        s_img = torch.from_numpy(s_img)
        s_label = torch.from_numpy(s_label).long()

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

        ####################################################
        # training model using target data
        ####################################################
        data_target = data_target_iter.next()
        t_img, t_label = data_target
        t_img = torch.from_numpy(t_img)
        t_label = torch.from_numpy(t_label).long()

        batch_size = len(t_img)

        domain_label = torch.ones(batch_size).long()

        if cuda:
            t_img = t_img.cuda()
            domain_label = domain_label.cuda()
            t_label = t_label.cuda()

        class_output, domain_output = my_net(input_data=t_img, alpha=alpha)
        err_t_label = loss_class(class_output, t_label)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_t_label + err_s_domain + err_s_label
        # err = err_t_label + err_s_label
        err.backward()
        optimizer.step()

        if i % 20 == 0:
            sys.stdout.write(
                "epoch: {epoch}, [iter: {batch} / all {total_batches}], err_s_label: {err_s_label}, err_s_domain: {err_s_domain}, err_t_domain: {err_t_domain}, err_t_label: {err_t_label}\n".format(
                    epoch=epoch,
                    batch=i,
                    total_batches=len_dataloader,
                    err_s_label=err_s_label.cpu().item(),
                    err_s_domain=err_s_domain.cpu().item(),
                    err_t_domain=err_t_domain.cpu().item(),
                    err_t_label=err_t_label.cpu().item(),
                )
            )

            sys.stdout.flush()

    # print('\n')
    # accu_s = test(test_ds_source)
    # print('Accuracy of the %s dataset: %f' % ('Source', accu_s))
    # accu_t = test(test_ds_target)
    # print('Accuracy of the %s dataset: %f\n' % ('Target', accu_t))
    # if accu_t > best_accu_t:
    #     best_accu_s = accu_s
    #     best_accu_t = accu_t
    #     torch.save(my_net, '{0}/mnist_mnistm_model_epoch_best.pth'.format(model_root))

print('============ Summary ============= \n')
print('Accuracy of the %s dataset: %f' % ('mnist', best_accu_s))
print('Accuracy of the %s dataset: %f' % ('mnist_m', best_accu_t))
print('Corresponding model was save in ' + model_root + '/mnist_mnistm_model_epoch_best.pth')