import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)



 
import torch
import torch.nn as nn


class EEGNet2d(nn.Module):
    """
    four block:
    1. conv2d
    2. depthwiseconv2d
    3. separableconv2d
    4. classify
    """
    def __init__(self, batch_size=4, num_class=2):
        super(EEGNet2d, self).__init__()
        self.batch_size = batch_size
        # 1. conv2d
        self.block1 = nn.Sequential()
        self.block1_conv = nn.Conv2d(in_channels=1,
                                     out_channels=8,
                                     kernel_size=(1, 64),
                                     padding=(0, 32),
                                     bias=False
                                     )
        self.block1.add_module('conv1', self.block1_conv)
        self.block1.add_module('norm1', nn.BatchNorm2d(8))

        # 2. depthwiseconv2d
        self.block2 = nn.Sequential()
        # [N, 8, 64, 128] -> [N, 16, 1, 128]
        self.block2.add_module('conv2', nn.Conv2d(in_channels=8,
                                                  out_channels=16,
                                                  kernel_size=(271, 1),
                                                  groups=2,
                                                  bias=False))
        self.block2.add_module('norm3', nn.BatchNorm2d(16))
        self.block2.add_module('act1', nn.ELU())
        # [N, 16, 1, 128] -> [N, 16, 1, 32]
        self.block2.add_module('pool1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.block2.add_module('drop1', nn.Dropout(p=0.5))

        # 3. separableconv2d
        self.block3 = nn.Sequential()
        self.block3.add_module('conv3', nn.Conv2d(in_channels=16,
                                                  out_channels=32,
                                                  kernel_size=(1, 16),
                                                  padding=(0, 8),
                                                  groups=16,
                                                  bias=False
                                                  ))
        self.block3.add_module('conv4', nn.Conv2d(in_channels=32,
                                                  out_channels=64,
                                                  kernel_size=(1, 1),
                                                  bias=False))
        self.block3.add_module('norm2', nn.BatchNorm2d(64))
        self.block3.add_module('act2', nn.ELU())
        self.block3.add_module('pool2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.block3.add_module('drop2', nn.Dropout(p=0.5))

        # 4. classify
        self.classify = nn.Sequential(nn.Linear(512, num_class))

        # self.classify = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, num_class)
        #     )

    def forward(self, x):
        # [B, 64, 128] -> [B, 1, 64, 128]
        if len(x.shape) == 3:
            x = x.unsqueeze(1) 
        # x = torch.reshape(x, (self.batch_size, 1, 271, 281))

        # [B, 1, 64, 128] -> [B, 1, 64, 127]
        # because pytorch's padding does not have the same option,
        # remove one column before convolution
        # x = x[:, :, :, range(127)]

        # [B, 1, 64, 128] -> [B, 8, 64, 128]
        x = self.block1(x)

        # [B, 8, 64, 128] -> [B, 16, 1, 128] -> [B, 16, 1, 32]
        x = self.block2(x)

        # [B, 16, 1, 32] -> [B, 16, 1, 31]
        # because pytorch's padding does not have the same option,
        # remove one column before convolution
        # x = x[:, :, :, range(31)]

        # [B, 16, 1, 31] -> [B, 16, 1, 4]
        x = self.block3(x)

        # [B, 16, 1, 4] -> [B, 64]
        x = x.view(x.size(0), -1)

        # [B, 64] -> [B, num_class]
        x = self.classify(x)

        # x = nn.functional.softmax(x, dim=1)

        return x

