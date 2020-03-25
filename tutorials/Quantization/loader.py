import os
import torchvision
import torch
from torchvision import transforms
import requests


def download_imagenet_1k():
    url = 'https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip'
    filename = './datasets/imagenet_1k_data.zip'

    r = requests.get(url)

    with open(filename, 'wb') as f:
        f.write(r.content)


def get_imagenet_1k_loaders(data_path, train_batch_size, eval_batch_size):
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    test_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=train_batch_size,
                                               sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=eval_batch_size,
                                              sampler=test_sampler)

    return train_loader, test_loader


if __name__ == "__main__":
    data_path = './datasets/imagenet_1k/'

    if not os.path.exists(data_path):
        download_imagenet_1k()
        print("SUCESS Dataset Download")
    else:
        print("Dataset Exist!!")

    train_batch_size = 30
    eval_batch_size = 30

    train_loader, test_loader = get_imagenet_1k_loaders(data_path, train_batch_size, eval_batch_size)

    print(train_loader.dataset)
    print(test_loader.dataset)