import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter

class resnet50(nn.Module):
    def __init__(self, nc=1000):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        return_layers = {'avgpool': 'feat'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        print(f'return_layers: {[name for name, _ in backbone.named_children()]}')
        self.fc1 = nn.Linear(2048, nc)

    def forward(self, input):
        output = self.body(input)
        output = list(output.values())
        output = torch.flatten(output[0], 1) # flatten all dimensions except batch
        output = self.fc1(output)
        return output

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(28304, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    model = resnet50(nc=2)
    out = model(torch.rand(1, 3, 256, 128))
    print(out.shape)

    '''
    m = torchvision.models.resnet18(pretrained=True)
    # extract layer1 and layer3, giving as names `feat1` and feat2`
    new_m = torchvision.models._utils.IntermediateLayerGetter(m,{'layer1': 'feat1', 'layer3': 'feat2'})
    out = new_m(torch.rand(1, 3, 224, 224))
    print([(k, v.shape) for k, v in out.items()])
    '''