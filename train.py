
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from utils.dataloaders import create_dataloader
from utils.augmentations import showImage
from models.model import resnet50, Net
from utils.general import increment_path
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torchinfo import summary

# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
print(LOCAL_RANK)
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='gender', help='data path')
    parser.add_argument('--epochs', type=int, default=100,
                        help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=72,
                        help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--height', type=int, default=256,
                        help='train, val image size (pixels)')
    parser.add_argument('--width', type=int, default=128,
                        help='train, val image size (pixels)')
    parser.add_argument('--freeze', action='store_true', help='Freeze layers: backbone')
    parser.add_argument('--workers', type=int, default=4,
                        help='max dataloader workers (per RANK in DDP mode)')
    return parser.parse_known_args()[0]


def main(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Parameters
    save_dir = increment_path(
        Path('runs/train-cls') / 'exp', exist_ok=None)  # increment run

    # Directories
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = save_dir / 'last.pt', save_dir / 'best.pt'

    train_loader = create_dataloader(
        path = Path(opt.dataset) / 'train', img_height = opt.height, img_width = opt.width, batch_size = opt.batch_size, workers = opt.workers)
    
    test_loader = create_dataloader(
        path = Path(opt.dataset) / 'val', img_height = opt.height, img_width = opt.width, batch_size = opt.batch_size, workers = opt.workers)
    

    classes = list(train_loader.dataset.class_to_idx.keys())
    nc = len(classes)    
    model = resnet50(nc=nc)
    #model = Net()
    
    # Freeze
    freeze = opt.freeze
    if freeze:
        for k, v in model.named_parameters():
            if 'body' in k:
                v.requires_grad = False
            print(f'freezing {k}  {v.requires_grad} ')

    model.to(device)
    print(summary(model, (1, 3, 256, 128)))

    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lrf = 0.001
    lf = lambda x: (1 - x / opt.epochs) * (1.0 - lrf+ lrf ) # linear

    # Scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    for epoch in range(opt.epochs):  # loop over the dataset multiple times
        tloss = 0.0 # train loss
        model.train()

        #pbar = enumerate(train_loader)
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        #for i, (images, labels) in enumerate(train_loader):
        for i, (images, labels) in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            
            # Test
            if i == len(pbar) - 1:  # last batch
                # prepare to count predictions for each class
                correct_pred = {classname: 0 for classname in classes}
                total_pred = {classname: 0 for classname in classes}
                # again no gradients needed
                with torch.no_grad():
                    model.eval()
                    for (imgs, labs) in test_loader:
                        imgs, labs = imgs.to(device, non_blocking=True), labs.to(device)
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
        print(('\n' + '%11s' * 4) % ('Epoch', 'GPU_mem', 'total_loss', 'lr'))
        print( f"{f'{epoch + 1}/{opt.epochs}':>10}{mem:>10}{tloss:>12.3g}  {scheduler.get_last_lr()}" + ' ' * 36)
        scheduler.step()
        torch.save(model.state_dict(), last)
    print('Finished Training')

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
