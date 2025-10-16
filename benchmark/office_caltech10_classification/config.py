import flgo.benchmark
import torchvision
import torchvision.utils
import torch.utils.data
from torchvision.datasets.utils import download_and_extract_archive, download_url, extract_archive
from torchvision import transforms
import os
from PIL import Image
import torch.nn as nn
from collections import OrderedDict

class OCDomainDataset(torchvision.datasets.VisionDataset):
    classes = ['backpack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']
    def __init__(self, root, domain:str, transforms=None, transform=None, target_transform=None):
        super(OCDomainDataset, self).__init__(root, transforms, transform, target_transform)
        self.domain = domain
        # construct image paths
        self.domain_path = os.path.join(self.root, 'Office_Caltech_DA_Dataset-main',self.domain)
        self.images_path = []
        self.labels = []
        img_lists = [os.listdir(os.path.join(self.domain_path, c)) for c in self.classes]
        for i, (img_list, c) in enumerate(zip(img_lists, self.classes)):
            for img in img_list:
                self.images_path.append(os.path.join(self.domain_path, c, img))
                self.labels.append(i)

    def __getitem__(self, item):
        img_path =self.images_path[item]
        label = self.labels[item]
        image = Image.open(img_path)
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)
        if self.transforms is not None:
            image, label = self.transforms(image, label)
        if image.dtype==torch.uint8 or image.dtype==torch.int8:
            image = image/255.0
        return image, label

    def __len__(self):
        return len(self.images_path)

class OfficeCaltech10(torch.utils.data.ConcatDataset):
    domains = ('Caltech', 'amazon', 'dslr', 'webcam',)
    url = 'https://github.com/ChristophRaab/Office_Caltech_DA_Dataset/archive/refs/heads/main.zip'
    def __init__(self, root, download=True, transform=None, target_transform=None, transforms=None) -> None:
        datasets = []
        file_exist = [os.path.exists(os.path.join(root, 'Office_Caltech_DA_Dataset-main', d)) for d in self.domains]
        if not all(file_exist) and download:
            download_and_extract_archive(self.url, root, remove_finished=True)
        for i, domain in enumerate(self.domains):
            transform = transform[i] if transform is list and len(transform) > i else transform
            target_transform = target_transform[i] if target_transform is list and len(target_transform) > i else target_transform
            transforms = transforms[i] if transforms is list and len(transforms) > i else transforms
            datasets.append(OCDomainDataset(root, domain=domain, transform=transform, target_transform=target_transform, transforms=transforms))
        super().__init__(datasets)
        self.domain_ids = []
        g = 0
        for i in range(len(self.datasets)):
            if i>=self.cumulative_sizes[g]:
                g += 1
            self.domain_ids.append(g)
transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.PILToTensor(),
    # transforms.Lambda(lambda x: x.float())
])
path = os.path.join(flgo.benchmark.data_root, 'office_caltech10')
train_data = OfficeCaltech10(path, transform=transform)

class AlexNet(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn6(self.fc1(x))
        x = self.relu(x)
        x = self.bn7(self.fc2(x))
        x = self.relu(x)
        x = self.fc3(x)
        return x

def get_model():
    return AlexNet()