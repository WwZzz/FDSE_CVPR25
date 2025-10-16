import torchvision.utils
import torch.utils.data
from torchvision.datasets.utils import download_and_extract_archive
from torchvision import transforms
import os
from PIL import Image
import torch.nn as nn
from collections import OrderedDict
import flgo

class PACSDomainDataset(torchvision.datasets.VisionDataset):
    classes = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')
    def __init__(self, root, domain:str, transforms=None, transform=None, target_transform=None):
        super(PACSDomainDataset, self).__init__(root, transforms, transform, target_transform)
        self.domain = domain
        # construct image paths
        self.domain_path = os.path.join(self.root, 'Homework3-PACS-master','PACS',self.domain)
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

class PACS(torch.utils.data.ConcatDataset):
    domains = ('art_painting', 'cartoon', 'photo', 'sketch')
    url = 'https://github.com/MachineLearning2020/Homework3-PACS/archive/refs/heads/master.zip'
    def __init__(self, root, download=True, transform=None, target_transform=None, transforms=None) -> None:
        datasets = []
        file_exist = [os.path.exists(os.path.join(root, 'Homework3-PACS-master','PACS', d)) for d in self.domains]
        if not all(file_exist) and download:
            download_and_extract_archive(self.url, root, remove_finished=True)
        for i, domain in enumerate(self.domains):
            transform = transform[i] if transform is list and len(transform) > i else transform
            target_transform = target_transform[i] if target_transform is list and len(target_transform) > i else target_transform
            transforms = transforms[i] if transforms is list and len(transforms) > i else transforms
            datasets.append(PACSDomainDataset(root, domain=domain, transform=transform, target_transform=target_transform, transforms=transforms))
        super().__init__(datasets)

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.PILToTensor(),
])
path = os.path.join(flgo.benchmark.data_root, 'PACS')
train_data = PACS(path, transform=transform).datasets
val_data = None
test_data = None

loss_fn = nn.CrossEntropyLoss()

def data_to_device(batch_data, device):
    return batch_data[0].to(device), batch_data[1].to(device)

def eval(model, data_loader, device) -> dict:
    model.eval()
    model.to(device)
    total_loss = 0.0
    num_correct = 0
    num_samples = 0
    for batch_id, batch_data in enumerate(data_loader):
        batch_data = data_to_device(batch_data, device)
        outputs = model(batch_data[0])
        batch_mean_loss = loss_fn(outputs, batch_data[-1]).item()
        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct = y_pred.eq(batch_data[-1].data.view_as(y_pred)).long().cpu().sum()
        num_correct += correct.item()
        num_samples += len(batch_data[-1])
        total_loss += batch_mean_loss * len(batch_data[-1])
    return {'accuracy': 1.0 * num_correct / num_samples, 'loss': total_loss / num_samples}

def compute_loss(model, batch_data, device) -> dict:
    tdata = data_to_device(batch_data, device)
    outputs = model(tdata[0])
    loss = loss_fn(outputs, tdata[-1])
    return {'loss': loss}

class AlexNet(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=7):
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


if __name__=='__main__':
    domains = ('art_painting', 'cartoon', 'photo', 'sketch')
    classes = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    t2i = transforms.Compose([transforms.ToPILImage(), ])
    WIDTH = 2
    COLS = 2
    for k, dk in enumerate(train_data):
        fig, ax = plt.subplots(nrows=WIDTH, ncols=WIDTH, sharex='all', sharey='all')
        ax = ax.flatten()
        for i in range(WIDTH**2):
            dk_train = dk
            idx = np.random.choice(range(len(dk_train)))
            img = t2i(dk_train[idx][0])
            ax[i].set_title(classes[int(dk_train[idx][1])], loc='center',pad=0, fontsize=11, fontweight='bold')
            ax[i].imshow(img, interpolation='nearest')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.axis('off')
        # plt.tight_layout()
        # fig.suptitle(domains[k], fontsize=14, fontweight='bold')
        plt.savefig(f'tmp_{domains[k]}.png', dpi=300)
    imgs = [Image.open(f'tmp_{domain}.png') for k, domain in enumerate(domains)]
    n = len(imgs)
    ROWS = int(n/COLS)
    if n % COLS != 0: ROWS+=1
    fig, axs = plt.subplots(ROWS, COLS, figsize=(5,3),sharex='all', sharey='all')
    axs = axs.flatten()
    # 遍历图像并绘制
    for k, (ax, img) in enumerate(zip(axs, imgs)):
        ax.imshow(img, interpolation='nearest')
        ax.set_title(f"Client-{domains[k]}", loc='center', pad=0, fontsize=11, fontweight='bold')
        ax.text(0.5, -0.005, f'size={len(train_data[k])}', ha='center', va='top', transform=ax.transAxes)
        ax.axis('off')  # 不显示坐标轴
    # 显示拼接后的图形
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0., hspace=0.)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig('res.png', dpi=300)
    [os.remove(f'tmp_{domain}.png') for k, domain in enumerate(domains)]
    # plt.show()
    # img = data[0].datasets[0][0][0]
    # img = tensor2img(img)
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.show()
    print('ok')