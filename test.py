import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, digit, c):
        # self.t = torch.ones(2, 224).to(device)
        self.t = torch.ones(2, 128) * digit
        self.c = c

    def __getitem__(self, index):
        return (
            self.t,
            self.c
        )

    def __len__(self):
        return 100000

def test(dataset_name):
    assert dataset_name in ['MNIST', 'mnist_m']

    model_root = 'models'
    image_root = os.path.join('dataset', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""

    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    if dataset_name == 'mnist_m':
        test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')

        dataset = GetLoader(
            data_root=os.path.join(image_root, 'mnist_m_test'),
            data_list=test_list,
            transform=img_transform_target
        )

        # SM: Faking out the mnist dataset, should get great accuracy
        dataset = DummyDataset(1,5)
    else:
        dataset = datasets.MNIST(
            root='dataset',
            train=False,
            transform=img_transform_source,
        )

        # SM: Faking out the mnist dataset, should get horrible accuracy
        dataset = DummyDataset(1,7)



    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ test """

    my_net = torch.load(os.path.join(
        model_root, 'mnist_mnistm_model_epoch_current.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        class_output, _ = my_net(input_data=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
