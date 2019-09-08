from torch import nn
import torch
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.full_conn1 = nn.Linear(1000 * 64, 1000)
        self.full_conn2 = nn.Linear(1000, 30)
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(1000 * 64, 1000)
        # self.fc2 = nn.Linear(1000, 30)
        # self.meow = nn.BatchNorm2d(32)
        # self.meow2 = nn.BatchNorm2d(64)
        # self.bn1 = nn.BatchNorm1d(1000)
        # self.bn2 = nn.BatchNorm1d(30)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.full_conn1(out)
        out = self.full_conn2(out)
        return out
        # x = F.relu(self.meow(self.conv1(x)))
        # x = self.pooling(x)
        #
        # x = F.relu(self.meow2(self.conv2(x)))
        # x = self.pooling(x)
        # x = x.reshape(x.size(0), -1)
        #
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.dropout(x, p=0.5, training=self.training)
        # # x = F.relu(self.fc1(x))
        # x = F.relu(self.bn2(self.fc2(x)))
        # x = F.dropout(x, p=0.25, training=self.training)
        # # x = self.fc2(x)
        # return x

