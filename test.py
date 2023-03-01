
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.augmentations import IMAGENET_MEAN, IMAGENET_STD
from models.model import resnet50, Net

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='runs/train-cls/exp2/last.pt', help='model.pt path(s)')
    parser.add_argument('--image', type=str,
                        default='gender', help='image path')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--height', type=int, default=256,
                        help='train, val image size (pixels)')
    parser.add_argument('--width', type=int, default=128,
                        help='train, val image size (pixels)')
    parser.add_argument('--workers', type=int, default=4,
                        help='max dataloader workers (per RANK in DDP mode)')
    return parser.parse_known_args()[0]

def main(opt):
    classes = ('female', 'male')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    img0 = cv2.imread(opt.image, cv2.IMREAD_COLOR) # BGR HWC
    img = cv2.resize(img0, (opt.width, opt.height ))
    img = np.float32(img) / 255  # uint8 to float32, 0-255 to 0.0-1.0
    img -= IMAGENET_MEAN
    img /= IMAGENET_STD
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    image = torch.from_numpy(img).unsqueeze(0)
    image = image.to(device)

    model = resnet50(nc=2)
    model.load_state_dict(torch.load(opt.weight))
    model.eval()
    model.to(device)

    outputs = model(image)
    _, predictions = torch.max(outputs, 1)
    
    im_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    plt.imshow(im_rgb)# plt.imshow format is HWC
    plt.title(classes[predictions])
    plt.savefig('result.jpg')

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
