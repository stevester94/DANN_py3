import torch
import torch.nn as nn
from functions import ReverseLayerF
import numpy as np

NUM_CLASSES=16

class CNN_Tipoff_Model(nn.Module):

    def __init__(self):
        super(CNN_Tipoff_Model, self).__init__()
        
        print("Using Tipoff CNN")
        self.conv = nn.Sequential()
        self.dense = nn.Sequential()

        # Unique naming matters
        
        # This first layer does depthwise convolution; each channel gets (out_channels/groups) number of filters. These are applied, and
        # then simply stacked in the output
        #self.feature.add_module('dyuh_1', nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1, groups=2))
        self.conv.add_module('dyuh_1', nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1))
        self.conv.add_module('dyuh_2', nn.ReLU(False)) # Optionally do the operation in place
        self.conv.add_module('dyuh_3', nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, stride=2))
        self.conv.add_module('dyuh_4', nn.ReLU(False))
        self.conv.add_module('dyuh_5', nn.Dropout())
        self.conv.add_module("dyuh_6", nn.Flatten())

        self.dense.add_module('dyuh_7', nn.Linear(50 * 58 + 1, 80)) # Input shape, output shape
        self.dense.add_module('dyuh_8', nn.ReLU(False))
        self.dense.add_module('dyuh_9', nn.Dropout())
        self.dense.add_module('dyuh_10', nn.Linear(80, NUM_CLASSES))
        self.dense.add_module('dyuh_11', nn.LogSoftmax(dim=1))

    def forward(self, x, t, alpha):
        conv_result = self.conv(x)

        t = torch.reshape(t, shape=(t.shape[0], 1))

        t = np.zeros(x.shape[0])
        t = torch.from_numpy(t).cuda()
        t = t.reshape(t.shape + (1,))

        concat = torch.cat((conv_result,t), dim=1)

        y_hat = self.dense(concat)

        # Fake out the domain_output
        t_hat = [[-1.0]] * x.shape[0]
        # l = [[-10.0,-10.0]] * 512
        # domain_output =  np.asarray(l)
        t_hat =  torch.as_tensor(t_hat).cuda()

        return y_hat, t_hat