import numpy as np
import torch
import cv2
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import math
IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


def showImage(train_loader):
    # get some random training images
    train_images, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_images.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = torchvision.utils.make_grid(train_images) # pytorch format is CHW  ,torch.Size([4, 3, 256, 128])
    npimg = img.numpy() # convert to numpy on cpu
    npimg= (npimg * np.array(IMAGENET_STD).reshape(3, 1, 1) + np.array(IMAGENET_MEAN).reshape(3, 1, 1)) # unnormalize
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # plt.show format is HWC
    plt.savefig('result.png')

def classify_transforms(height, width):
    assert isinstance(height, int), f'ERROR: classify_transforms size {height} must be integer, not (list, tuple)'
    assert isinstance(width, int), f'ERROR: classify_transforms size {width} must be integer, not (list, tuple)'
    return T.Compose([LetterBox(height, width), ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

class LetterBox:
    # LetterBox class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, height=640, width=640, auto=False, stride=32):
        super().__init__()
        self.h, self.w = (height, width)
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top:top + h, left:left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out

class CenterCrop:
    # CenterCrop class for image preprocessing, i.e. T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, height=640, width=640):
        super().__init__()
        self.h, self.w = (height, width)

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top:top + m, left:left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:
    # ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        super().__init__()
        self.half = half

    def __call__(self, im):  # im = np.array HWC in BGR order
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
