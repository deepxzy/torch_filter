import torch
import cv2
import numpy as np
from torch_filter import torch_filter
weight = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
net=torch_filter(weight,is_grad=False)
img=cv2.imread(r"images/img.png")
image = np.transpose((np.array(img, np.float64))/255, [2, 0, 1])
image = torch.from_numpy(image).type(torch.FloatTensor)
image = image.unsqueeze(dim=0)
image_sharp=net(image)
image_sharp=image_sharp.cpu().detach().numpy().copy().squeeze()
predictimag=np.transpose(image_sharp, [1, 2, 0])*255
cv2.imwrite(r'images/new_img.png',predictimag)