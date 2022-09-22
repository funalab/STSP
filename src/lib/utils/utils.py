import os
import torch
import torch.nn as nn
import torchvision.models as models

from src.lib.datasets.dataset import STDataset
from src.lib.loss.focal_loss import FocalLoss
from src.lib.models.resnet import ResNet50, ResNet101, ResNet152


def get_model(args):
    model_list = ['resnet18', 'alexnet', 'squeezenet1_0', 'vgg16', 'densenet161', 'inception_v3', 'googlenet', 'shufflenet_v2_x1_0', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'resnext50_32x4d', 'wide_resnet50_2', 'mnasnet1_0']
    pretrained = eval(args.pretrained)

    if args.model == 'resnet':
        model = ResNet50(
            img_channel=int(args.n_input_channels),
            num_classes=int(args.n_classes),
            loss=eval(args.lossfun)
        )

    elif args.model in model_list:
        if args.model == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif args.model == 'alexnet':
            model = models.alexnet(pretrained=pretrained)
        elif args.model == 'squeezenet1_0':
            model = models.squeezenet1_0(pretrained=pretrained)
        elif args.model == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
        elif args.model == 'densenet161':
            model = models.densenet161(pretrained=pretrained)
        elif args.model == 'inception_v3':
            model = models.inception_v3(pretrained=pretrained)
        elif args.model == 'googlenet':
            model = models.googlenet(pretrained=pretrained)
        elif args.model == 'shufflenet_v2_x1_0':
            model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        elif args.model == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
        elif args.model == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=pretrained)
        elif args.model == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=pretrained)
        elif args.model == 'resnext50_32x4d':
            model = models.resnext50_32x4d(pretrained=pretrained)
        elif args.model == 'wide_resnet50_2':
            model = models.wide_resnet50_2(pretrained=pretrained)
        elif args.model == 'mnasnet1_0':
            model = models.mnasnet1_0(pretrained=pretrained)
        model.loss = eval(args.lossfun)
    else:
        raise ValueError('Unknown model name: {}'.format(args.model))

    return model

def get_dataset(args):
    train_dataset = STDataset(
        root=args.root_path,
        split_list=args.split_list_train,
        label_list=args.label_list,
        basename=args.basename,
        crop_size=eval(args.crop_size),
        crop_range=eval(args.crop_range),
        train=True,
    )
    validation_dataset = STDataset(
        root=args.root_path,
        split_list=args.split_list_validation,
        label_list=args.label_list,
        basename=args.basename,
        crop_size=eval(args.crop_size),
        crop_range=eval(args.crop_range),
        train=False,
    )

    return train_dataset, validation_dataset


def get_test_dataset(args):
    test_dataset = STDataset(
        root=args.root_path,
        split_list=args.split_list_test,
        label_list=args.label_list,
        basename=args.basename,
        crop_size=eval(args.crop_size),
        crop_range=eval(args.crop_range),
        train=False,
    )

    return test_dataset


def print_args(dataset_args, model_args, updater_args, runtime_args):
    """ Export config file
    Args:
        dataset_args    : Argument Namespace object for loading dataset
        model_args      : Argument Namespace object for Generator and Discriminator
        updater_args    : Argument Namespace object for Updater
        runtime_args    : Argument Namespace object for runtime parameters
    """
    dataset_dict = {k: v for k, v in vars(dataset_args).items() if v is not None}
    model_dict = {k: v for k, v in vars(model_args).items() if v is not None}
    updater_dict = {k: v for k, v in vars(updater_args).items() if v is not None}
    runtime_dict = {k: v for k, v in vars(runtime_args).items() if v is not None}
    print('============================')
    print('[Dataset]')
    for k, v in dataset_dict.items():
        print('%s = %s' % (k, v))
    print('\n[Model]')
    for k, v in model_dict.items():
        print('%s = %s' % (k, v))
    print('\n[Updater]')
    for k, v in updater_dict.items():
        print('%s = %s' % (k, v))
    print('\n[Runtime]')
    for k, v in runtime_dict.items():
        print('%s = %s' % (k, v))
    print('============================\n')


def export_to_config(save_dir, dataset_args, model_args, updater_args, runtime_args):
    """ Export config file
    Args:
        save_dir (str)      : /path/to/save_dir
        dataset_args (dict) : Dataset arguments
        model_args (dict)   : Model arguments
        updater_args (dict) : Updater arguments
        runtime_args (dict) : Runtime arguments
    """
    dataset_dict = {k: v for k, v in vars(dataset_args).items() if v is not None}
    model_dict = {k: v for k, v in vars(model_args).items() if v is not None}
    updater_dict = {k: v for k, v in vars(updater_args).items() if v is not None}
    runtime_dict = {k: v for k, v in vars(runtime_args).items() if v is not None}
    with open(os.path.join(save_dir, 'parameters.cfg'), 'w') as txt_file:
        txt_file.write('[Dataset]\n')
        for k, v in dataset_dict.items():
            txt_file.write('%s = %s\n' % (k, v))
        txt_file.write('\n[Model]\n')
        for k, v in model_dict.items():
            txt_file.write('%s = %s\n' % (k, v))
        txt_file.write('\n[Updater]\n')
        for k, v in updater_dict.items():
            txt_file.write('%s = %s\n' % (k, v))
        txt_file.write('\n[Runtime]\n')
        for k, v in runtime_dict.items():
            txt_file.write('%s = %s\n' % (k, v))
        txt_file.write('\n[MN]\n')
