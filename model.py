import numpy
from tensorflow.python.ops.gen_array_ops import shape
import torch
import torch.nn as nn
from functions import ReverseLayerF


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature_2 = nn.Sequential()
        import torch

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


        self.feature.add_module('f_conv1', nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1))
        self.feature.add_module('f_relu2', nn.ReLU(False))
        self.feature.add_module('f_conv2', nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, stride=2))
        self.feature.add_module('f_relu2', nn.ReLU(False))
        self.feature.add_module('f_drop1', nn.Dropout())


        self.feature_2.add_module('f_fc1', nn.Linear(50 * 58 + 1, 256)) # t is also fed into the feature extractor
        # self.feature_2.add_module('f_fc2', nn.Linear(256, 256))
        # self.feature_2.add_module('f_fc3', nn.Linear(256, 256))

        # x = torch.ones(10, 2, 128)
        # print(self.feature(x).shape)
        # import sys
        # sys.exit(1)

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


        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(256, 256))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(False))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(256, 80))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(False))
        self.class_classifier.add_module('c_fc3', nn.Linear(80, 16))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(256, 100))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(False))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 1))
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, t, alpha):
        # print("input_data:", input_data.shape)

        # Doesn't change anything
        # input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        # print("feature:", feature.shape)

        feature = feature.view(-1, 50 * 58)
        
        t = torch.reshape(t, shape=(t.shape[0], 1))

        feature_2 = self.feature_2(torch.cat((feature,t), dim=1))
        # print("Feature View:", feature.shape)

        reverse_feature = ReverseLayerF.apply(feature_2, alpha)
        # print("Reverse Feature:", feature.shape)
        
        class_output = self.class_classifier(feature_2)
        domain_output = self.domain_classifier(reverse_feature)

        domain_output = torch.clamp(domain_output, 0, 30)

        # print(domain_output)


        # Fake out the domain_output
        # l = [[1.0,0.0]] * 1024
        # l = [[-10.0,-10.0]] * 512
        # domain_output =  numpy.asarray(l)
        # domain_output =  torch.as_tensor(domain_output).cuda()



        # import sys
        # sys.exit(0)

        # print("====================")
        return class_output, domain_output
