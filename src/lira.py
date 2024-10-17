import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (Convert, Cutout, RandomHorizontalFlip,
                             RandomTranslate, ToDevice, ToTensor, ToTorchImage)
from ffcv.transforms.common import Squeeze
from PIL import Image
from scipy.stats import norm
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import VisionDataset
from tqdm import tqdm


def logitize(l):
    if l==0:
        f = np.exp(-l) - np.random.random()/100000000
    else:
        f = np.exp(-l)
    phi = np.log(f / (1-f))
    return phi

def get_label_pipeline():
    label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(torch.device('cuda')), Squeeze()]
    return label_pipeline

def get_image_pipeline(train=True, **args):
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
    if train:
        image_pipeline.extend([
            RandomHorizontalFlip(),
            RandomTranslate(padding=2),
            Cutout(8, tuple(map(int, args["cifar_mean"]))), # Note Cutout is done before normalization.
        ])
    image_pipeline.extend([
        ToTensor(),
        ToDevice(torch.device('cuda'), non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16),
        torchvision.transforms.Normalize(args["cifar_mean"], args["cifar_std"]),
    ])
    return image_pipeline

def get_shadow_loaders(path, batch_size, size_shadow_dataset, 
                       sample_idxs_in, sample_idxs_out, seed_in, seed_out, **args):
    label_pipeline = get_label_pipeline()
    image_pipeline = get_image_pipeline(cifar_mean=args['cifar_mean'], cifar_std=args['cifar_std'])
    pipelines={
        'label':label_pipeline,
        'image':image_pipeline
    }

    # IN loader
    loader_in = Loader(
        path,
        batch_size=batch_size,
        num_workers=8,
        drop_last=False,
        order=OrderOption.RANDOM,
        pipelines=pipelines,
        indices=sample_idxs_in,
        seed=seed_in
    )

    # OUT loader
    loader_out = Loader(
        path,
        batch_size=batch_size,
        num_workers=8,
        drop_last=False,
        order=OrderOption.RANDOM,
        pipelines=pipelines,
        indices=sample_idxs_out,
        seed=seed_out
    )

    return loader_in, loader_out
    


def train_MLP_classifier(X_train, y_train, dims=[8, 4], out_dim=2, epochs=200, lr=1e-3, seed=123):
    train_dataset = TensorDataset(torch.Tensor(X_train.to_numpy()), torch.Tensor(y_train.to_numpy()))
    train_loader = DataLoader(train_dataset, batch_size=64)

    #test_dataset = TensorDataset(torch.Tensor(X_val.to_numpy()), torch.Tensor(y_val.to_numpy()))
    #test_loader = DataLoader(test_dataset, batch_size=64)

    torch.manual_seed(seed)
    random.seed(seed)
    mlp = MLP([X_train.shape[1]] + dims, out_dim=out_dim)
    optimizer = optim.Adam(mlp.parameters(), lr=lr)

    if out_dim==1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for e in range(epochs):

        current_loss = 0
        val_loss = 0
        for i, data in enumerate(train_loader):
            inputs, targets = data

            optimizer.zero_grad()

            outputs = mlp(inputs)#.reshape(-1)
            targets = targets.type(torch.LongTensor)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            current_loss += loss.item()

    return mlp

class MLP(nn.Module):

    def __init__(self, dims, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.layers = nn.ModuleList([])

        for i in range(len(dims) - 1):
            self.layers.append(
                nn.Linear(
                    in_features=dims[i], out_features=dims[i + 1]
                )
            )
        self.layers.append(
            nn.Linear(in_features=dims[-1], out_features=out_dim)
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)
        x = self.layers[-1](x)
        if self.out_dim==1:
            x = torch.sigmoid(x)
        #x = x.softmax(dim=1)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_convnet():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net = Co

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    return 

class MyDataset(VisionDataset):
    def __init__(self, paths, transform=None, target_transform=None, sample_ids=None, 
                 target_record_id=None, shadow_dataset=False, in_dataset=False, add_target=False,
                add_data=None):
        data = []
        labels = []
        for path in paths:
            with open(path, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
                data.append(batch['data'])
                labels.append(batch['labels'])
        data = np.vstack(data).reshape(-1,3,32,32)
        data = data.transpose((0,2,3,1))
        labels = np.array(labels).flatten()

        if add_data is not None:
            with open(add_data[0], 'rb') as f:
                data_to_add = pickle.load(f)
            with open(add_data[1], 'rb') as f:
                labels_to_add = pickle.load(f)
            data = np.append(data, data_to_add, axis=0)
            labels = np.append(labels, labels_to_add)
            self.data = data
            self.labels = labels
        
        elif add_target:
            target_record = data[target_record_id].reshape(-1,3,32,32).transpose((0,2,3,1))
            data_shadow = np.append(target_record, data_shadow, axis=0)
            labels_shadow = np.append(labels[target_record_id], labels_shadow)
            self.data = data_shadow
            self.labels = labels_shadow

        # create shadow datasets
        elif shadow_dataset:
            
            data_shadow = data[sample_ids]
            labels_shadow = labels[sample_ids]
            #print(f'shadow data shape: {data_shadow.shape}')

            if in_dataset:
                target_record = data[target_record_id].reshape(-1,3,32,32).transpose((0,2,3,1))
                data_shadow = np.append(data_shadow, target_record, axis=0)
                labels_shadow = np.append(labels_shadow, labels[target_record_id])
                #print(data_shadow.shape)
            self.data = data_shadow
            self.labels = labels_shadow
        else:
            self.data = data
            self.labels = labels
            
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        img, target = self.data[index], self.labels[index]
        if self.transform is not None:
            img = Image(self.transform(img))

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.labels)

def evaluate_model(model, testloader, model_name=None):
    total = 0
    num_correct = 0
    
    for data in tqdm(testloader):
        with torch.no_grad():
            inputs, labels = data
            y_pred = model(inputs)
            _, pred = torch.max(y_pred.data, 1)
            #loss = criterion(y_pred, labels)
            total += len(labels)
            num_correct += len(np.where(pred==labels)[0])
    if model_name is not None:
        print(f'{model_name} accuracy = {(num_correct/total)*100:.2f}%')
    else:
        print(f'accuracy = {(num_correct/total)*100:.2f}%')

def train_resnet(trainloader, epochs=50):
    resnet18 = models.resnet18(weights=None)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)

    resnet18.train()
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
    
        running_loss = 0.0
        i = 0
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
            i += 1
    
    #print('Finished Training')
    return resnet18

class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(torch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, channels_out,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         groups=groups, bias=False),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(inplace=True)
    )

def train_shadow_model(data_loader, num_classes, seed, size_shadow_dataset, batch_size, epochs):
    
    torch.manual_seed(seed)
    model = resnet_model(num_classes)
    model = model.to(memory_format = torch.channels_last).cuda()
    opt = SGD(model.parameters(), lr=.5, momentum=0.9, weight_decay=5e-4)
    iters_per_epoch = size_shadow_dataset // batch_size + 1
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                        [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
                        [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)

    for e in range(epochs):
        for ims, labs in data_loader:
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
    return model

def get_loss_on_target(model, loader_target, loss_fn):
    model.eval()
    with torch.no_grad():
        for ims, labs in loader_target:
            with autocast():
                target_pred = model(ims)
                loss = loss_fn(target_pred, labs.reshape(1))
        phi = logitize(loss.cpu())
        return phi, loss.cpu()

def evaluate_model(model, data_loader):
    model.eval()
    with torch.no_grad():
        total_correct, total_num = 0., 0.
        for ims, labs in data_loader:
            with autocast():
                out = model(ims)
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]
    print(f'Accuracy: {total_correct / total_num * 100:.1f}%')   

def resnet_model(num_classes):
    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2)
    )
    return model

def get_ratio(model, loader, loss, mean_in, std_in, mean_out, std_out):
    phi, loss = get_loss_on_target(
        model, loader, loss
    )
    l_in = norm.pdf(phi, mean_in, std_in)
    l_out = norm.pdf(phi, mean_out, std_out)

    ratio = l_in / l_out
    return ratio, loss