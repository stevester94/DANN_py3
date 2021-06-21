#! /usr/bin/env python3

import random
import time
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

from tf_dataset_getter  import get_shuffled_and_windowed_from_pregen_ds

torch.set_default_dtype(torch.float64)


import matplotlib.pyplot as plt
def _do_loss_curve(history):
    plt.figure()
    plt.title('Losses')
    plt.plot(history["indices"], history['val_label_loss'], label='Validation Label Loss')
    plt.plot(history["indices"], history['val_domain_loss'], label='Validation Domain Loss')
    plt.plot(history["indices"], history['train_label_loss'], label='Train Label Loss')
    plt.plot(history["indices"], history['train_domain_loss'], label='Train Domain Loss')
    plt.legend()
    plt.xlabel('Epoch')


def plot_loss_curve(history):
    _do_loss_curve(history)
    plt.show()


BATCH_LOGGING_DECIMATION_FACTOR = 20
NUM_LOGS_PER_EPOCH = 10

cuda = True
cudnn.benchmark = True
lr = 0.0001
n_epoch = 2
model_root = "./model"

manual_seed = 1337
random.seed(manual_seed)
torch.manual_seed(manual_seed)

from steves_utils import utils

batch_size = 128
# batch_size = 1
ORIGINAL_BATCH_SIZE = 100

source_distance = "2.8.14.20.26"
target_distance = 32

source_ds_path = "{datasets_base_path}/automated_windower/windowed_EachDevice-200k_batch-100_stride-20_distances-{distance}".format(
    datasets_base_path=utils.get_datasets_base_path(), distance=source_distance
)

target_ds_path = "{datasets_base_path}/automated_windower/windowed_EachDevice-200k_batch-100_stride-20_distances-{distance}".format(
    datasets_base_path=utils.get_datasets_base_path(), distance=target_distance
)



train_ds_source, val_ds_source, test_ds_source = get_shuffled_and_windowed_from_pregen_ds(source_ds_path, ORIGINAL_BATCH_SIZE, batch_size)
train_ds_target, val_ds_target, test_ds_target = get_shuffled_and_windowed_from_pregen_ds(target_ds_path, ORIGINAL_BATCH_SIZE, batch_size)



# print("Unfortunately have to calculate the length of the source dataset by iterating over it. Standby...")
# num_batches_in_train_ds_source = 0
# for i in train_ds_source:
#     num_batches_in_train_ds_source += 1
# print("Done. Source Train DS Length:", num_batches_in_train_ds_source)

# print("Unfortunately have to calculate the length of the source dataset by iterating over it. Standby...")
# num_batches_in_train_ds_target = 0
# for i in train_ds_target:
#     num_batches_in_train_ds_target += 1
# print("Done. Target Train DS Length:", num_batches_in_train_ds_target)

# print("We are hardcoding DS length!")
num_batches_in_train_ds_source = 25000
num_batches_in_train_ds_target = 25000
# num_batches_in_train_ds_source = 50
# num_batches_in_train_ds_target = 50

my_net = CNNModel()

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.CrossEntropyLoss()
# loss_domain = torch.nn.CrossEntropyLoss()
loss_domain = torch.nn.L1Loss()
# loss_domain = torch.nn.MSELoss()


if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training
best_accu_t = 0.0
last_time = time.time()


history = {}
history["indices"] = []
history["val_label_loss"] = []
history["val_domain_loss"] = []
history["train_label_loss"] = []
history["train_domain_loss"] = []

for epoch in range(n_epoch):

    data_source_iter = train_ds_source.as_numpy_iterator()
    # data_target_iter = train_ds_target.as_numpy_iterator()

    batches_to_log = np.linspace(1, num_batches_in_train_ds_source, NUM_LOGS_PER_EPOCH, dtype=int)
    err_s_label_epoch = 0
    err_s_domain_epoch = 0

    for i in range(num_batches_in_train_ds_source):
        log_this_batch = i in batches_to_log

        p = float(i + epoch * num_batches_in_train_ds_source) / n_epoch / num_batches_in_train_ds_source
        gamma = 10
        alpha = 2. / (1. + np.exp(-gamma * p)) - 1

        alpha = 0.1

        # alpha = 0
        # print(p)

        # print("Alpha", alpha)

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label, s_domain = data_source

        s_img = torch.from_numpy(s_img)
        s_label = torch.from_numpy(s_label).long()
        s_domain = torch.from_numpy(s_domain).long()

        my_net.zero_grad()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            s_domain = s_domain.cuda()


        class_output, domain_output = my_net(input_data=s_img, t=s_domain, alpha=alpha)
        domain_output = torch.flatten(domain_output)


        err_s_label = loss_class(class_output, s_label)
        err_s_domain = loss_domain(domain_output, s_domain)

        err_s_label_epoch += err_s_label.cpu().item()
        err_s_domain_epoch += err_s_domain.cpu().item()

        err = err_s_domain + err_s_label


        err.backward()
        optimizer.step()

        if i % BATCH_LOGGING_DECIMATION_FACTOR == 0:
            cur_time = time.time()
            batches_per_second = BATCH_LOGGING_DECIMATION_FACTOR / (cur_time - last_time)
            last_time = cur_time
            sys.stdout.write(
                (
                    "epoch: {epoch}, [iter: {batch} / all {total_batches}], "
                    "batches_per_second: {batches_per_second}, "
                    "err_s_label: {err_s_label}, "
                    "err_s_domain: {err_s_domain},"
                    "alpha: {alpha}\n"
                ).format(
                        batches_per_second=batches_per_second,
                        epoch=epoch+1,
                        batch=i,
                        total_batches=num_batches_in_train_ds_source,
                        err_s_label=err_s_label.cpu().item(),
                        err_s_domain=err_s_domain.cpu().item(),
                        alpha=alpha
                    )
            )

            sys.stdout.flush()

        if log_this_batch:
            print("Logging this batch")
            val_label_accuracy, val_label_loss, val_domain_loss = \
                test(my_net, loss_class, loss_domain, val_ds_source.as_numpy_iterator())
            
            print(
                val_label_loss,
                val_domain_loss,
                err_s_label_epoch / i,
                err_s_domain_epoch / i,
            )

            history["indices"].append(epoch + i/num_batches_in_train_ds_source)
            history["val_label_loss"].append(val_label_loss)
            history["val_domain_loss"].append(val_domain_loss)
            history["train_label_loss"].append(err_s_label_epoch / i)
            history["train_domain_loss"].append(err_s_domain_epoch / i)

            print("Val label accuracy:{}, Val label loss:{}, Val domain loss: {}".format(
                val_label_accuracy, val_label_loss, val_domain_loss))

plot_loss_curve(history)
    # accu_t = test(test_ds_target)
    # print('Accuracy of the %s dataset: %f\n' % ('Target', accu_t))
    # if accu_t > best_accu_t:
    #     best_accu_s = accu_s
    #     best_accu_t = accu_t
    #     torch.save(my_net, '{0}/mnist_mnistm_model_epoch_best.pth'.format(model_root))

# print('============ Summary ============= \n')
# print('Accuracy of the %s dataset: %f' % ('mnist', best_accu_s))
# print('Accuracy of the %s dataset: %f' % ('mnist_m', best_accu_t))
# print('Corresponding model was save in ' + model_root + '/mnist_mnistm_model_epoch_best.pth')
