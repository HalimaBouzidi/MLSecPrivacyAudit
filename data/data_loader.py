from __future__ import print_function

import os,argparse
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder

def build_datasets(args):
    if args['data']['dataset'] == 'IMAGENET':
        return build_default_imagenet_dataset(args)
    elif args['data']['dataset'] == 'TINY-IMAGENET':
        return build_default_tiny_imagenet_dataset(args)
    elif args['data']['dataset'] == 'MNIST':
        return build_default_MNIST_dataset(args)
    elif args['data']['dataset'] == 'CIFAR10':
        return build_default_CIFAR10_dataset(args)
    elif args['data']['dataset'] == 'CIFAR100':
        return build_default_CIFAR100_dataset(args)
    else:
        raise NotImplementedError

def build_data_loader(args):
    if args['data']['dataset'] == 'IMAGENET':
        return build_default_imagenet_data_loader(args)
    elif args['data']['dataset'] == 'TINY-IMAGENET':
        return build_default_tiny_imagenet_data_loader(args)
    elif args['data']['dataset'] == 'MNIST':
        return build_default_MNIST_data_loader(args)
    elif args['data']['dataset'] == 'CIFAR10':
        return build_default_CIFAR10_data_loader(args)
    elif args['data']['dataset'] == 'CIFAR100':
        return build_default_CIFAR100_data_loader(args)
    else:
        raise NotImplementedError

def build_default_MNIST_dataset(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNIST(root=args['data']['dataset_dir'], train=True, download=True, transform=transform)
    val_dataset = MNIST(root=args['data']['dataset_dir'], train=False, download=True, transform=transform)

    return train_dataset, val_dataset

def build_default_MNIST_data_loader(args):

    train_dataset, val_dataset = build_default_MNIST_dataset(args)

    if args['data']['distributed'] == 'True':
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args['data']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last = True,
        num_workers=args['data']['data_loader_workers_per_gpu'],
        pin_memory=True,
    )

    if args['data']['distributed'] == 'True' and getattr(args, 'distributed_val', 'True'):
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    eval_batch_size = min(args['data']['batch_size'], 16) \
        if not getattr(args, 'eval_only', False) else args['data']['batch_size']

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args['data']['data_loader_workers_per_gpu'],
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler

def build_default_CIFAR10_dataset(args):
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])    

    non_train_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    train_dataset = CIFAR10(root=args['data']['dataset_dir'], train=True,
                                        download=True, transform=train_transforms)
    val_dataset = CIFAR10(root=args['data']['dataset_dir'], train=False,
                                        download=True, transform=non_train_transforms)

    return train_dataset, val_dataset

def build_default_CIFAR10_data_loader(args):

    train_dataset, val_dataset = build_default_CIFAR10_dataset(args)

    if args['data']['distributed'] == 'True':
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args['data']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last = True,
        num_workers=args['data']['data_loader_workers_per_gpu'],
        pin_memory=True,
    )

    if args['data']['distributed'] == 'True' and getattr(args, 'distributed_val', 'True'):
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    eval_batch_size = min(args['data']['batch_size'], 16) \
        if not getattr(args, 'eval_only', False) else args['data']['batch_size']

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args['data']['data_loader_workers_per_gpu'],
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler

def build_default_CIFAR100_dataset(args):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])    

    non_train_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    train_dataset = CIFAR100(root=args['data']['dataset_dir'], train=True,
                                        download=True, transform=train_transforms)
    val_dataset = CIFAR100(root=args['data']['dataset_dir'], train=False,
                                        download=True, transform=non_train_transforms)

    return train_dataset, val_dataset

def build_default_CIFAR100_data_loader(args):
    
    train_dataset, val_dataset = build_default_CIFAR100_dataset(args)

    if args['data']['distributed'] == 'True':
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args['data']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last = True,
        num_workers=args['data']['data_loader_workers_per_gpu'],
        pin_memory=True,
    )

    if args['data']['distributed'] == 'True' and getattr(args, 'distributed_val', 'True'):
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    eval_batch_size = min(args['data']['batch_size'], 16) \
        if not getattr(args, 'eval_only', False) else args['data']['batch_size']

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args['data']['data_loader_workers_per_gpu'],
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler

def build_default_imagenet_dataset(args):
    traindir = os.path.join(args['data']['dataset_dir'], "train")
    valdir = os.path.join(args['data']['dataset_dir'], "val")

    #build transforms
    train_transform = get_data_transform(args, is_training=True, augment=args['data']['augment'])
    test_transform = get_data_transform(args, is_training=False, augment=args['data']['augment'])

    train_dataset = ImageFolder(traindir, train_transform)
    val_dataset = ImageFolder(valdir, test_transform)

    return train_dataset, val_dataset

def build_default_imagenet_data_loader(args):
    
    train_dataset, val_dataset = build_default_imagenet_data_loader(args)

    if args['data']['distributed'] == 'True':
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args['data']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last = True,
        num_workers=args['data']['data_loader_workers_per_gpu'],
        pin_memory=True,
    )

    if args['data']['distributed'] == 'True' and getattr(args, 'distributed_val', 'True'):
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    eval_batch_size = min(args['data']['batch_size'], 16) \
        if not getattr(args, 'eval_only', False) else args['data']['batch_size']

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args['data']['data_loader_workers_per_gpu'],
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler

def build_default_tiny_imagenet_dataset(args):
    traindir = os.path.join(args['data']['dataset_dir'], "train")
    valdir = os.path.join(args['data']['dataset_dir'], "val")      

    train_transform = get_data_transform(args, is_training=True, augment=args['data']['augment'])
    test_transform = get_data_transform(args, is_training=False, augment=args['data']['augment'])

    train_dataset = ImageFolder(traindir, train_transform)
    val_dataset = ImageFolder(valdir, test_transform)

    return train_dataset, val_dataset 

def build_default_tiny_imagenet_data_loader(args):
    
    train_dataset, val_dataset = build_default_tiny_imagenet_dataset(args)

    if args['data']['distributed'] == 'True':
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args['data']['batch_size'],
        shuffle=(train_sampler is None),      
        sampler=train_sampler,
        drop_last = True,
        num_workers=args['data']['data_loader_workers_per_gpu'],
        pin_memory=True,
    )    

    if args['data']['distributed'] == 'True' and getattr(args, 'distributed_val', 'True'):
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None  
        
    eval_batch_size = min(args['data']['batch_size'], 16) \
        if not getattr(args, 'eval_only', False) else args['data']['batch_size']  

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args['data']['data_loader_workers_per_gpu'],
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Configuration for dataloders')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'TINY-IMAGENET', 'IMAGENET'])
    parser.add_argument('--dataset_dir', type=str, default='../../datasets/cifar10', help='Path to the dataset folder')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for train and validation dataloaders')
    parser.add_argument('--data_loader_workers_per_gpu', type=int, default=2, help='Number of workers per GPU device')
    parser.add_argument('--distributed', type=str, default='False', help='Whether to use the distributed training')
    parser.add_argument('--augment', type=str, default='auto_augment_tf', help='Type of data augmentation')

    run_args = parser.parse_args()

    train_dataset, val_dataset = build_datasets(run_args)

    print(len(train_dataset.data), len(train_dataset.targets))