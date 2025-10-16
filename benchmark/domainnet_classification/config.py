import torchvision.utils
import torch.utils.data
from torchvision.datasets.utils import download_and_extract_archive, download_url, extract_archive
from torchvision import transforms
import os
from PIL import Image
import flgo.benchmark

domain_list = [
    'clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'
]
path = os.path.join(flgo.benchmark.data_root, 'domainnet')
classes = [
    'bird',
    'feather',
    'headphones',
    'ice_cream',
    'teapot',
    'tiger',
    'whale',
    'windmill',
    'wine_glass',
    'zebra'
]
class DomainDataset(torchvision.datasets.VisionDataset):
    url_temp = {
        "{}.zip": "http://csr.bu.edu/ftp/visda/2019/multi-source/{}.zip",
        "{}_train.txt": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/{}_train.txt",
        "{}_test.txt":"http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/{}_test.txt"
    }
    file_names = []
    def __init__(self, root, domain:str, split='train', classes = None, download:bool=True, transforms=None, transform=None, target_transform=None):
        super(DomainDataset, self).__init__(root, transforms, transform, target_transform)
        self.domain = domain
        if not os.path.exists(os.path.join(root, "{}_{}.txt".format(self.domain, split))):
            download_url("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/{}_{}.txt".format(self.domain, split), root=self.root)
        if not os.path.exists(os.path.join(root, self.domain)):
            if os.path.exists(os.path.join(root, "{}.zip".format(self.domain))):
                try:
                    extract_archive(os.path.join(self.root, "{}.zip".format(self.domain)), self.root, remove_finished=False)
                except Exception as e:
                    print(e)
                    raise FileExistsError('There exists error in download .zipfile')
            else:
                if download==True:
                    self.download_data()
                else:
                    raise FileExistsError('File not exists. Please set download=True the download the raw data of {}'.format(self.domain))
        with open(os.path.join(root, '{}_{}.txt'.format(self.domain, split)), 'r') as inf:
            self.all_images_path = inf.readlines()
        self.all_label_names = [p.split(os.path.sep)[1] for p in self.all_images_path]
        self.label_list = tuple(sorted(list(set(self.all_label_names))))
        self.set_classes(classes)

    def download_data(self):
        for k,v in self.url_temp.items():
            file_name = k.format(self.domain)
            url = v.format(self.domain) if self.domain in ['infograph', 'quickdraw', 'real', 'sketch'] else v.format(f"groundtruth/{self.domain}")
            if file_name.endswith('.zip'):
                download_and_extract_archive(url, self.root, remove_finished=True)
            else:
                download_url(url, self.root)

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

    def set_classes(self, classes):
        if classes is None: classes = self.label_list
        self.classes = classes
        tmp_images = []
        tmp_labels = []
        for i,(img, lb) in enumerate(zip(self.all_images_path, self.all_label_names)):
            if lb in self.classes:
                tmp_images.append(os.path.join(self.root, img.strip().split(' ')[0]))
                tmp_labels.append(lb)
        self.images_path = tmp_images
        self.labels = [self.classes.index(lb) for lb in tmp_labels]

class DomainNet(torch.utils.data.ConcatDataset):
    domains = ('clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch')
    def __init__(self, root, split:str='all', classes=None, download=True, transform=None, target_transform=None, transforms=None) -> None:
        datasets = []
        for i, domain in enumerate(self.domains):
            transform = transform[i] if transform is list and len(transform) > i else transform
            target_transform = target_transform[i] if target_transform is list and len(target_transform) > i else target_transform
            transforms = transforms[i] if transforms is list and len(transforms) > i else transforms
            if split!='all':
                datasets.append(DomainDataset(root, domain=domain, split=split, classes=classes, download=download, transform=transform, target_transform=target_transform, transforms=transforms))
            else:
                data_train = DomainDataset(root, domain=domain, split='train', classes=classes, download=download, transform=transform, target_transform=target_transform, transforms=transforms)
                data_test = DomainDataset(root, domain=domain, split='test', classes=classes, download=download, transform=transform, target_transform=target_transform, transforms=transforms)
                datasets.append(torch.utils.data.ConcatDataset([data_train, data_test]))
        super().__init__(datasets)
        self.domain_ids = []
        g = 0
        for i in range(len(self.datasets)):
            if i>=self.cumulative_sizes[g]:
                g += 1
            self.domain_ids.append(g)

    def set_classes(self, classes):
        for dataset in self.datasets: dataset.set_classes(classes)
        self.classes = classes
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.domain_ids = []
        g = 0
        for i in range(len(self.datasets)):
            if i>=self.cumulative_sizes[g]:
                g += 1
            self.domain_ids.append(g)

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.PILToTensor(),
])
train_data = DomainNet(path,
                       classes=classes, transform=transform)
test_data = None
import torch.nn as nn
from collections import OrderedDict

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
        self.fc2 = nn.Linear(1024,1024)
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
    # return torchvision.models.get_model('resnet18', num_classes = len(classes))
    return AlexNet(len(train_data.classes))