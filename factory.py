import os
import time
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
import torch.nn.functional as F
import timm


# load model first time
def load_model(model_path):

    model_kwargs = dict(patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=12,
                        representation_size=None,
                        qkv_bias=False)
    model = timm.models.vision_transformer._create_vision_transformer(
        'vit_base_patch16_224_in21k',
        pretrained=False,
        num_classes=200,
        **model_kwargs)

    state = torch.load(model_path)

    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(
                    key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))

    return model


def load_pretrained_vit_b_16():
    backbone = timm.create_model('vit_base_patch16_224_miil_in21k',
                                 pretrained=True)
    '''
    projector = nn.Sequential(nn.Linear(11221, 2048), nn.BatchNorm1d(2048),
                              nn.ReLU(), nn.Linear(2048, 512),
                              nn.BatchNorm1d(512), nn.ReLU(),
                              nn.Linear(512, 200))

    '''
    '''
    projector = nn.Sequential(nn.Linear(11221, 2048), nn.BatchNorm1d(2048),
                              nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                              nn.Linear(2048, 512), nn.BatchNorm1d(512),
                              nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                              nn.Linear(512, 200))
    '''

    projector = nn.Sequential(nn.Linear(11221, 2048), nn.BatchNorm1d(2048),
                              nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                              nn.Linear(2048, 200))

    model = nn.Sequential(backbone, projector)

    print('build model success!')

    return model
