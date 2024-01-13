import torch
from torch import nn
class PeCFE(nn.Module):
    def __init__(self,):
        super().__init__()
        self.pe=nn.Parameter(torch.randn(1,  1, 256,256))
        self.pre=nn.Sequential(
            nn.LayerNorm([256, 256]),
            nn.Conv2d(1, 16, 1, 1, 0),
            nn.GELU()
        )
        self.conv1=nn.Sequential(
            nn.LayerNorm([256, 256]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([256,256]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([256, 256]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([256, 256]),

        )
        self.conv1_1 = nn.Sequential(
            nn.LayerNorm([256, 256]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([256, 256]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([256, 256]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([256, 256]),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.LayerNorm([128, 128]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([128, 128]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([128, 128]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([128, 128]),

        )
        self.conv2_1 = nn.Sequential(
            nn.LayerNorm([128, 128]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([128, 128]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([128, 128]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([128, 128]),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.LayerNorm([64, 64]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([64, 64]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([64, 64]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([64, 64]),

        )
        self.conv3_1 = nn.Sequential(
            nn.LayerNorm([64, 64]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([64, 64]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([64, 64]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([64, 64]),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.LayerNorm([32, 32]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([32, 32]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([32, 32]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([32, 32]),

        )
        self.conv4_1 = nn.Sequential(
            nn.LayerNorm([32, 32]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([32, 32]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([32, 32]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([32, 32]),
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.LayerNorm([16, 16]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([16, 16]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([16, 16]),
            nn.Conv2d(16,16,3,1,1),
            nn.GELU(),
            nn.LayerNorm([16, 16]),

        )
        self.conv5_1 = nn.Sequential(
            nn.LayerNorm([16, 16]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([16, 16]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([16, 16]),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.LayerNorm([16, 16]),
            nn.MaxPool2d(2)
        )
        self.res1=nn.Sequential(
            nn.LayerNorm([256, 256]),
            nn.Conv2d(16, 16, 2, 2,0 ),
            nn.GELU()
        )
        self.res2 = nn.Sequential(
            nn.LayerNorm([128, 128]),
            nn.Conv2d(16, 16, 2, 2,0 ),
            nn.GELU()
        )
        self.res3 = nn.Sequential(
            nn.LayerNorm([64, 64]),
            nn.Conv2d(16, 16, 2, 2,0 ),
            nn.GELU()
        )
        self.res4 = nn.Sequential(
            nn.LayerNorm([32, 32]),
            nn.Conv2d(16, 16, 2, 2,0 ),
            nn.GELU()
        )
        self.res5 = nn.Sequential(
            nn.LayerNorm([16, 16]),
            nn.Conv2d(16, 16, 2, 2,0 ),
            nn.GELU()
        )
        self.highway1=nn.Sequential(
            nn.LayerNorm([256, 256]),
            nn.Conv2d(16, 16, 8, 8, 0),
            nn.GELU()
        )
        self.highway2=nn.Sequential(
            nn.LayerNorm([256, 256]),
            nn.Conv2d(16, 16, 32, 32, 0),
            nn.GELU()
        )
        self.head=nn.Sequential(
            nn.Conv2d(16,16,1,1,0),
            nn.Flatten(start_dim=1, end_dim=-1),
        )
        self.gelu=nn.GELU()
    def forward(self, x):
        x = self.pre(x)
        # x=self.pre(x+self.pe)
        temp=x
        h1=self.highway1(x)
        h2=self.highway2(x)
        x=self.conv1(x)
        x+=temp
        x=self.gelu(x)
        temp = self.res1(x)
        x=self.conv1_1(x)
        x += temp
        x = self.gelu(x)
        temp = x
        x=self.conv2(x)
        x+=temp
        x = self.gelu(x)
        temp=self.res2(x)
        x = self.conv2_1(x)
        x += temp
        x = self.gelu(x)
        temp = x
        x = self.conv3(x)
        x += temp
        x = self.gelu(x)
        temp = self.res3(x)
        x = self.conv3_1(x)
        x += temp
        # x += h1
        x = self.gelu(x)

        temp = x
        x = self.conv4(x)
        x += temp
        x = self.gelu(x)
        temp = self.res4(x)
        x = self.conv4_1(x)
        x += temp
        x = self.gelu(x)

        temp = x
        x = self.conv5(x)
        x += temp
        x = self.gelu(x)
        temp = self.res5(x)
        x = self.conv5_1(x)
        x += temp
        # x += h2
        x = self.gelu(x)

        x=self.head(x)
        return x
