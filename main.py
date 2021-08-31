#! /usr/bin/env python3

import random
import time
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import json
import os

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


BATCH_LOGGING_DECIMATION_FACTOR = 20
RESULTS_PATH = "./results/"


BEST_MODEL_PATH = RESULTS_PATH+"./best_model.pth"
cuda = True
cudnn.benchmark = True



# MODEL_TYPE = "CIDA"
MODEL_TYPE = "CNN"
MAX_CACHE_SIZE = 200000*len(ALL_SERIAL_NUMBERS)*1000


if __name__ == "__main__"  and len(sys.argv) == 1:
    j = json.loads(sys.stdin.read())
elif __name__ == "__main__"  and len(sys.argv) > 1:
    fake_args = {}
    fake_args["experiment_name"] = "Fill Me"
    fake_args["lr"] = 0.0001
    fake_args["n_epoch"] = 10
    fake_args["batch_size"] = 256
    fake_args["source_distance"] = ALL_DISTANCES_FEET
    fake_args["target_distance"] = ALL_DISTANCES_FEET
    fake_args["desired_serial_numbers"] = ALL_SERIAL_NUMBERS
    fake_args["alpha"] = 0.001
    fake_args["num_additional_extractor_fc_layers"]=1
    fake_args["patience"] = 10
    fake_args["seed"] = 1337
    fake_args["num_examples_per_device"]=20000
    fake_args["window_stride"]=50
    fake_args["window_length"]=256 #Will break if not 256 due to model hyperparameters
    fake_args["desired_runs"]=[1]
    j = fake_args


lr = j["lr"]
n_epoch = j["n_epoch"]
batch_size = j["batch_size"]
source_distance = j["source_distance"]
target_distance = j["target_distance"]
desired_serial_numbers = j["desired_serial_numbers"]
alpha = j["alpha"]
num_additional_extractor_fc_layers = j["num_additional_extractor_fc_layers"]
experiment_name = j["experiment_name"]
patience = j["patience"]
seed = j["seed"]
num_examples_per_device = j["num_examples_per_device"]
window_stride = j["window_stride"]
window_length = j["window_length"]
desired_runs = j["desired_runs"]

print(j)

experiment = {}
experiment["args"] = j

random.seed(seed)
torch.manual_seed(seed)

start_time_secs = time.time()

source_ds = ORACLE_Torch.ORACLE_Torch_Dataset(
                desired_serial_numbers=desired_serial_numbers,
                desired_distances=source_distance,
                desired_runs=desired_runs,
                window_length=window_length,
                window_stride=window_stride,
                num_examples_per_device=num_examples_per_device,
                seed=seed,  
                max_cache_size=MAX_CACHE_SIZE,
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
                max_cache_size=MAX_CACHE_SIZE,
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
    num_workers=1,
    persistent_workers=True,
    prefetch_factor=50,
    pin_memory=True
)
_, target_val_dl, target_test_dl = wrap_datasets_in_dataloaders(
    (target_train_ds, target_val_ds, target_test_ds),
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    persistent_workers=True,
    prefetch_factor=50,
    pin_memory=True
)

# print("Priming train_dl")
# data_source_iter = iter(source_train_dl)
# for i in range(len(source_train_dl)):
#     data_source = data_source_iter.next()

# print("Priming train_dl")
# data_source_iter = iter(source_train_dl)
# for i in range(len(source_train_dl)):
#     data_source = data_source_iter.next()

# print("Priming train_dl")
# data_source_iter = iter(source_train_dl)
# for i in range(len(source_train_dl)):
#     data_source = data_source_iter.next()

if MODEL_TYPE == "CNN":
    from cnn_model import CNN_Model
    my_net = CNN_Model(num_additional_extractor_fc_layers)
elif MODEL_TYPE == "CIDA":
    from cida_model import CIDA_Model
    my_net = CIDA_Model(num_additional_extractor_fc_layers)

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

# loss_class = torch.nn.CrossEntropyLoss()
loss_class = torch.nn.NLLLoss()
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
history["source_val_label_loss"] = []
history["source_val_domain_loss"] = []
history["target_val_label_loss"] = []
history["target_val_domain_loss"] = []
history["source_train_label_loss"] = []
history["source_train_domain_loss"] = []
history["source_val_label_accuracy"] = []
history["target_val_label_accuracy"] = []

best_epoch_index_and_combined_val_label_loss = [0, float("inf")]
for epoch in range(1,n_epoch+1):

    data_source_iter = iter(source_train_dl)
    # data_target_iter = train_ds_target.as_numpy_iterator()

    err_s_label_epoch = 0
    err_s_domain_epoch = 0

    for i in range(len(source_train_dl)):

        if alpha is None:
            p = float(i + epoch * source_train_dl) / n_epoch / source_train_dl
            gamma = 10
            alpha = 2. / (1. + np.exp(-gamma * p)) - 1

        alpha = 0
        # print(p)

        # print("Alpha", alpha)

        # training model using source data
        data_source = data_source_iter.next()
        # print(data_source)
        s_img, s_label, s_domain = data_source

        # s_img = data_source["iq"]
        # s_label = data_source["serial_number"]
        # s_domain = data_source["distance_ft"]

        # s_img = torch.from_numpy(s_img)
        # s_label = torch.from_numpy(s_label).long()
        # s_domain = torch.from_numpy(s_domain).long()

        my_net.zero_grad()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            s_domain = s_domain.cuda()


        class_output, domain_output = my_net(x=s_img, t=s_domain, alpha=alpha)
        domain_output = torch.flatten(domain_output)


        err_s_label = loss_class(class_output, s_label)
        err_s_domain = loss_domain(domain_output, s_domain)

        err_s_label_epoch += err_s_label.cpu().item()
        err_s_domain_epoch += err_s_domain.cpu().item()

        err = err_s_domain + err_s_label


        err.backward()
        optimizer.step()

        if i % BATCH_LOGGING_DECIMATION_FACTOR == 0:
            # import psutil
            # print("Nump open files", len(psutil.Process().open_files()))

            # print("Len Cache: ", source_train_ds.os.cache.get_len_cache())
            # print("Cache Misses: ", source_train_ds.os.cache.get_cache_misses())
            # print("Cache Hits: ", source_train_ds.os.cache.get_cache_hits())

            cur_time = time.time()
            batches_per_second = BATCH_LOGGING_DECIMATION_FACTOR / (cur_time - last_time)
            last_time = cur_time
            sys.stdout.write(
                (
                    "epoch: {epoch}, [iter: {batch} / all {total_batches}], "
                    "batches_per_second: {batches_per_second:.4f}, "
                    "err_s_label: {err_s_label:.4f}, "
                    "err_s_domain: {err_s_domain:.4f}, "
                    "alpha: {alpha:.4f}\n"
                ).format(
                        batches_per_second=batches_per_second,
                        epoch=epoch,
                        batch=i,
                        total_batches=len(source_train_dl),
                        err_s_label=err_s_label.cpu().item(),
                        err_s_domain=err_s_domain.cpu().item(),
                        alpha=alpha
                    )
            )

            sys.stdout.flush()

    source_val_label_accuracy, source_val_label_loss, source_val_domain_loss = \
        test(my_net, loss_class, loss_domain, source_val_dl)
    
    target_val_label_accuracy, target_val_label_loss, target_val_domain_loss = \
        test(my_net, loss_class, loss_domain, target_val_dl)

    history["indices"].append(epoch)
    history["source_val_label_loss"].append(source_val_label_loss)
    history["source_val_domain_loss"].append(source_val_domain_loss)
    history["target_val_label_loss"].append(target_val_label_loss)
    history["target_val_domain_loss"].append(target_val_domain_loss)
    history["source_train_label_loss"].append(err_s_label_epoch / i)
    history["source_train_domain_loss"].append(err_s_domain_epoch / i)
    history["source_val_label_accuracy"].append(source_val_label_accuracy)
    history["target_val_label_accuracy"].append(target_val_label_accuracy)

    sys.stdout.write(
        (
            "=============================================================\n"
            "epoch: {epoch}, "
            "acc_src_val_label: {source_val_label_accuracy:.4f}, "
            "err_src_val_label: {source_val_label_loss:.4f}, "
            "err_src_val_domain: {source_val_domain_loss:.4f}, "
            "acc_trgt_val_label: {target_val_label_accuracy:.4f}, "
            "err_trgt_val_label: {target_val_label_loss:.4f}, "
            "err_trgt_val_domain: {target_val_domain_loss:.4f}"
            "\n"
            "=============================================================\n"
        ).format(
                epoch=epoch,
                source_val_label_accuracy=source_val_label_accuracy,
                source_val_label_loss=source_val_label_loss,
                source_val_domain_loss=source_val_domain_loss,
                target_val_label_accuracy=target_val_label_accuracy,
                target_val_label_loss=target_val_label_loss,
                target_val_domain_loss=target_val_domain_loss,
            )
    )

    sys.stdout.flush()

    combined_val_label_loss = source_val_label_loss + target_val_label_loss
    if best_epoch_index_and_combined_val_label_loss[1] > combined_val_label_loss:
        print("New best")
        best_epoch_index_and_combined_val_label_loss[0] = epoch
        best_epoch_index_and_combined_val_label_loss[1] = combined_val_label_loss
        torch.save(my_net, BEST_MODEL_PATH)
    
    elif epoch - best_epoch_index_and_combined_val_label_loss[0] > patience:
        print("Patience ({}) exhausted".format(patience))
        break


print("Loading best model from epoch {} with combined loss of {}".format(*best_epoch_index_and_combined_val_label_loss))
my_net = torch.load(BEST_MODEL_PATH)

save_loss_curve(history, RESULTS_PATH+"loss_curve.png")

source_test_label_accuracy, source_test_label_loss, source_test_domain_loss = \
    test(my_net, loss_class, loss_domain, source_test_dl)

target_test_label_accuracy, target_test_label_loss, target_test_domain_loss = \
    test(my_net, loss_class, loss_domain, target_test_dl)

source_val_label_accuracy, source_val_label_loss, source_val_domain_loss = \
    test(my_net, loss_class, loss_domain, source_val_dl)

target_val_label_accuracy, target_val_label_loss, target_val_domain_loss = \
    test(my_net, loss_class, loss_domain, target_val_dl)


stop_time_secs = time.time()
total_time_secs = stop_time_secs - start_time_secs

experiment["results"] = {}
experiment["results"]["source_test_label_accuracy"] = source_test_label_accuracy
experiment["results"]["source_test_label_loss"] = source_test_label_loss
experiment["results"]["source_test_domain_loss"] = source_test_domain_loss

experiment["results"]["target_test_label_accuracy"] = target_test_label_accuracy
experiment["results"]["target_test_label_loss"] = target_test_label_loss
experiment["results"]["target_test_domain_loss"] = target_test_domain_loss

experiment["results"]["source_val_label_accuracy"] = source_val_label_accuracy
experiment["results"]["source_val_label_loss"] = source_val_label_loss
experiment["results"]["source_val_domain_loss"] = source_val_domain_loss

experiment["results"]["target_val_label_accuracy"] = target_val_label_accuracy
experiment["results"]["target_val_label_loss"] = target_val_label_loss
experiment["results"]["target_val_domain_loss"] = target_val_domain_loss

experiment["results"]["total_time_seconds"] = total_time_secs

experiment["history"] = history

with open(os.path.join(RESULTS_PATH,"experiment.json"), "w") as f:
    json.dump(experiment, f, sort_keys=True, indent=4)