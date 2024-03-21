from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from .hiposiam import HipoSiam
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(args):    
    if args.model == 'simsiam':
        model =  SimSiam(get_backbone(args.backbone))
        if args.proj_layers is not None:
            model.projector.set_layers(args.proj_layers)
    elif args.model == 'hiposiam':
        model =  HipoSiam(get_backbone(args.backbone), angle=args.angle, rotate_times = args.rotate_times, rnn_nonlin=args.rnn_nonlin, remove_rnn=args.remove_rnn, use_aug = args.use_aug)
        if args.proj_layers is not None:
            model.projector.set_layers(args.proj_layers)
    elif args.model == 'byol':
        model = BYOL(get_backbone(args.backbone))
    elif args.model == 'simclr':
        model = SimCLR(get_backbone(args.backbone))
    elif args.model == 'swav':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model






