
import argparse
import torch
from pathlib import Path
from models.model import resnet50, Net
from utils.dataloaders import create_dataloader


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='runs/train-cls/exp2/last.pt', help='model.pt path(s)')
    parser.add_argument('--dataset', type=str,
                        default='gender', help='data path')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--height', type=int, default=256,
                        help='train, val image size (pixels)')
    parser.add_argument('--width', type=int, default=128,
                        help='train, val image size (pixels)')
    parser.add_argument('--workers', type=int, default=4,
                        help='max dataloader workers (per RANK in DDP mode)')
    return parser.parse_known_args()[0]


def main(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_loader = create_dataloader(
        path=Path(opt.dataset) / 'val', img_height=opt.height, img_width=opt.width, batch_size=opt.batch_size, workers=opt.workers)
    classes = list(test_loader.dataset.class_to_idx.keys())
    nc = len(classes)
    model = resnet50(nc=nc)
    # model = Net()
    print(opt.weight)
    model.load_state_dict(torch.load(opt.weight))
    model.to(device)

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # again no gradients needed
    with torch.no_grad():
        model.eval()
        for (imgs, labs) in test_loader:
            imgs, labs = imgs.to(
                device, non_blocking=True), labs.to(device)
            outputs = model(imgs)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labs, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
