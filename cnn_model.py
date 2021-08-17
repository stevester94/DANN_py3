import torch
import torch.nn as nn
from functions import ReverseLayerF

NUM_CLASSES=16

class CNN_Model(nn.Module):

    def __init__(self, num_additional_extractor_fc_layers):
        super(CNN_Model, self).__init__()

        # My shit
        # self.feature.add_module("f_flatten", nn.Flatten())
        # self.feature.add_module('c_fc1', nn.Linear(2*128, 800))

        
        # self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        # self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        # self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu1', nn.ReLU(True))
        # self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        # self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        # self.feature.add_module('f_drop1', nn.Dropout2d())
        # self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu2', nn.ReLU(True))
        # The final shape of the feature extractor is None,50,4,4. The subsequent components just flatten this though

        # x = torch.ones(10, 3, 28,28)
        # print(self.feature(x).shape)

        # First conv layer
        # I assume out channels is number of filters
        # self.feature.add_module('f_conv1', nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1))
        # self.feature.add_module('f_bn1', nn.BatchNorm1d(50))
        # self.feature.add_module('f_pool1', nn.MaxPool1d(2))
        # self.feature.add_module('f_relu1', nn.ReLU(True))

        # # Second conv layer
        # self.feature.add_module('f_conv2', nn.Conv1d(in_channels=50, out_channels=50, kernel_size=29, stride=1))
        # self.feature.add_module('f_bn2', nn.BatchNorm1d(50))
        # self.feature.add_module('f_drop1', nn.Dropout())
        # self.feature.add_module('f_pool2', nn.MaxPool1d(2))
        # self.feature.add_module('f_relu2', nn.ReLU(True))

        self.feature = nn.Sequential()

        # Unique naming matters
        self.feature.add_module('dyuh_1', nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1))
        self.feature.add_module('dyuh_2', nn.ReLU(False)) # Optionally do the operation in place
        self.feature.add_module('dyuh_3', nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, stride=2))
        self.feature.add_module('dyuh_4', nn.ReLU(False))
        self.feature.add_module('dyuh_5', nn.Dropout())

        self.feature.add_module("dyuh_6", nn.Flatten())

        self.feature.add_module('dyuh_7', nn.Linear(50 * 58, 80)) # Input shape, output shape
        self.feature.add_module('dyuh_8', nn.ReLU(False))
        self.feature.add_module('dyuh_9', nn.Dropout())

        self.feature.add_module('dyuh_10', nn.Linear(80, NUM_CLASSES))

        # self.feature.add_module('dyuh_10', nn.Linear(256, 256))
        # self.feature.add_module('dyuh_11', nn.ReLU(False))
        # self.feature.add_module('dyuh_12', nn.Dropout())

        self.feature.add_module('dyuh_11', nn.LogSoftmax(dim=1))

        """
        Original
        """
        # self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc1', nn.Linear(50 * 58, 100))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout())
        # self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # self.class_classifier.add_module('c_fc3', nn.Linear(100, 16))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        # self.domain_classifier = nn.Sequential()
        # self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 58, 100))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        # self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        # self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))


        # self.class_classifier = nn.Sequential()

        # # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        # self.domain_classifier = nn.Sequential()
        # self.domain_classifier.add_module('d_fc1', nn.Linear(256, 100))
        # # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        # self.domain_classifier.add_module('d_relu1', nn.ReLU(False))
        # self.domain_classifier.add_module('d_fc2', nn.Linear(100, 1))
        # # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x, t, alpha):
        y_hat = self.feature(x)

        # Fake out the domain_output
        t_hat = [[-1.0]] * x.shape[0]
        # l = [[-10.0,-10.0]] * 512
        # domain_output =  np.asarray(l)
        t_hat =  torch.as_tensor(t_hat).cuda()

        return y_hat, t_hat
