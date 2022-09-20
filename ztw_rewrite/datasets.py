import os

from torchvision import datasets, transforms

DOWNLOAD = False


def get_mnist():
    dataset_path = os.environ['MNIST_PATH']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    train_data = datasets.MNIST(dataset_path, train=True, download=DOWNLOAD, transform=transform)
    train_eval_data = train_data
    test_data = datasets.MNIST(dataset_path, train=False, download=DOWNLOAD, transform=transform)
    return train_data, train_eval_data, test_data


def get_cifar10(proper_normalization=True):
    if proper_normalization:
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    dataset_path = os.environ['CIFAR10_PATH']
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])
    train_data = datasets.CIFAR10(dataset_path, train=True, download=DOWNLOAD, transform=transform_train)
    train_eval_data = datasets.CIFAR10(dataset_path, train=True, download=DOWNLOAD, transform=transform_eval)
    test_data = datasets.CIFAR10(dataset_path, train=False, download=DOWNLOAD, transform=transform_eval)
    return train_data, train_eval_data, test_data


def get_cifar100(proper_normalization=True):
    if proper_normalization:
        mean, std = (0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    dataset_path = os.environ['CIFAR100_PATH']
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])
    train_data = datasets.CIFAR100(dataset_path, train=True, download=DOWNLOAD, transform=transform_train)
    train_eval_data = datasets.CIFAR100(dataset_path, train=True, download=DOWNLOAD, transform=transform_eval)
    test_data = datasets.CIFAR100(dataset_path, train=False, download=DOWNLOAD, transform=transform_eval)
    return train_data, train_eval_data, test_data


def get_tinyimagenet(proper_normalization=True):
    if proper_normalization:
        mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    dataset_path = os.environ['TINYIMAGENET_PATH']
    train_path = f'{dataset_path}/train'
    test_path = f'{dataset_path}/val/images'
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.RandomCrop(64, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])
    train_data = datasets.ImageFolder(train_path, transform=transform_train)
    train_eval_data = datasets.ImageFolder(train_path, transform=transform_eval)
    test_data = datasets.ImageFolder(test_path, transform=transform_eval)
    return train_data, train_eval_data, test_data


DATASETS_NAME_MAP = {
    'mnist': get_mnist,
    'cifar10': get_cifar10,
    'cifar100': get_cifar100,
    'tinyimagenet': get_tinyimagenet,
}
