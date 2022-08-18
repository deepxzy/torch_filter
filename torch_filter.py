import torch
import torch.nn as nn
import numpy as np
import cv2


class torch_filter(nn.Module):
    def __init__(self, filter_weight, is_grad=False):
        super(torch_filter, self).__init__()
        assert type(filter_weight) == np.ndarray
        k=filter_weight.shape[0]
        filter=torch.tensor(filter_weight).unsqueeze(dim=0).unsqueeze(dim=0)
        filters = torch.cat([filter, filter, filter], dim=0)

        self.conv = nn.Conv2d(3, 3, kernel_size=k, groups=3, bias=False, padding=int((k-1)/2))
        self.conv.weight.data.copy_(filters)
        self.conv.requires_grad_(is_grad)


    def forward(self,x):
        output = self.conv(x)
        output = torch.clip(output, 0, 1)
        return output

if __name__ == '__main__':
    weight = np.ones((5,5))
    net=torch_filter(weight,is_grad=False)
    img=torch.randn((9,3,256,256))
    img=net(img)
    print(img.shape)#torch.Size([9, 3, 256, 256])

