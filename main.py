import os
import torch
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='Selection : [mnist, cifar10, cifar100, tiny-imagenet, imagenet]')
    parser.add_argument('--model', type=str, default='vgg16', help='Selection : [vgg16, resnet18, resnet50]')
    parser.add_argument('--device', type=str, default=None, help='Selection : [cpu, cuda]')
    args = parser.parse_args()

    return args

def main():
    # arguments
    args = get_parser()

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Dataset : {args.dataset}\nmodel : {args.model}\ndevice : {device}")

    

if __name__ == "__main__":
    main()