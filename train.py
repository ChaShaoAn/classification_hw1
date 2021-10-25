# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import time
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
import torch.nn.functional as F
import pathlib
import numpy as np
import factory
import birdDataset
from utils.utility import *
from tqdm import tqdm
from torchsummary import summary

# shift+alt+F -> for auto formatting

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

inference_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

classes = []


# %%
def getClasses():
    tmp_path = str(pathlib.Path().resolve())
    lines = []
    classes = []

    with open(tmp_path + '/classes.txt') as f:
        lines = f.readlines()

    for line in lines:
        # pos = line.find('.')
        classes.append(line[:-1])

    return classes


# %%
def train(model, criterion, optimizer, dataloader, switch='Eval', epoch=1):
    train_list = []
    model = model.to(device)
    if (switch == 'Train'):
        model.train()
    elif (switch == 'Valid'):
        model.eval()
    elif (switch == 'Test'):
        model.eval()
    else:
        model.eval()
    while (epoch):
        total_loss = 0.0
        accTop1 = 0
        accTop5 = 0
        for i_batch, data in tqdm(enumerate(dataloader)):
            image = data[0].to(device)
            label = data[1].to(device)
            output = model(image)

            accTop = evaluteAcc(output, label, topk=(1, 5))
            accTop1 += accTop[0]
            accTop5 += accTop[1]

            loss = criterion(output, label)

            if (switch == 'Train'):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            # total_loss += loss.detach().cpu()
            # print(loss.detach().cpu())

        total_loss /= (i_batch + 1)
        accTop1 /= (i_batch + 1)
        accTop5 /= (i_batch + 1)
        epoch -= 1

    return total_loss, accTop1, accTop5


# %%
def run_image_classification(model, image_path, transform, classes, topk=5):
    """Inference
    """
    testing_dir = 'testing_images\\'
    testing_seq = 'testing_img_order.txt'
    testing_output = 'output/answer.txt'
    fo = open(testing_output, 'w')
    i = 0
    model = model.to(device)
    model.eval()
    with open(testing_seq) as f:
        lines = f.readlines()

        for line in lines:
            i = i + 1
            if (i == 10):
                pass
            # Read image and run prepro
            image = Image.open(testing_dir + line[:-1]).convert("RGB")
            image_tensor = transform(image)
            print(
                f"\n\nImage size after transformation: {image_tensor.size()}")

            image_tensor = image_tensor.unsqueeze(0)
            print(f"Image size after unsqueezing: {image_tensor.size()}")

            # test
            image_tensor = image_tensor.to(device)

            # Feed input
            output = model(image_tensor)
            print(f"Output size: {output.size()}")

            output = output.squeeze()
            print(f"Output size after squeezing: {output.size()}")

            # Result postpro
            _, indices = torch.sort(output, descending=True)
            probs = F.softmax(output, dim=-1)

            fo.write(line[:-1] + ' ' + classes[indices[0]] + '\n')

    fo.close()

    return


# %%
def evaluteAcc(y_pred, y, topk=(1, )):
    maxk = max(topk)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0,
                                                                  keepdim=True)
        res.append(correct_k)
    return res


# %%
def cross_valid(model=None,
                criterion=None,
                optimizer=None,
                dataset=None,
                k_fold=10,
                batch_size=10):

    total_size = len(dataset)
    fraction = 1 / k_fold
    seg = int(total_size * fraction)
    # tr:train,val:valid; r:right,l:left;
    # eg: trrr: right index of right side train subset
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    train_loss = 0.0
    val_loss = 0.0
    train_list = []
    val_list = []
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        # msg
        print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" %
              (trll, trlr, trrl, trrr, vall, valr))

        train_left_indices = list(range(trll, trlr))
        train_right_indices = list(range(trrl, trrr))

        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall, valr))

        train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset, val_indices)

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=batch_size,
                                                 shuffle=False)
        train_loss, train_acc_top1, train_acc_top5 = train(model,
                                                           criterion,
                                                           optimizer,
                                                           train_loader,
                                                           switch='Train',
                                                           epoch=1)
        print('train - loss: ' + str(train_loss) + '\t, top1: ' +
              str(train_acc_top1.cpu() / batch_size) + '\t, top5: ' +
              str(train_acc_top5.cpu() / batch_size))

        val_loss, valid_acc_top1, valid_acc_top5 = train(model,
                                                         criterion,
                                                         optimizer,
                                                         val_loader,
                                                         switch='Valid')
        print('valid - loss: ' + str(val_loss) + '\t, top1: ' +
              str(valid_acc_top1.cpu() / batch_size) + '\t, top5: ' +
              str(valid_acc_top5.cpu() / batch_size))

        train_list.append(train_loss)
        val_list.append(val_loss)

        update_lost_hist(train_list, val_list)

    return train_list, val_list


# %%
def train_and_valid(model=None,
                    criterion=None,
                    optimizer=None,
                    dataset=None,
                    batch_size=10):

    total_size = len(dataset)
    fraction = 1 / 10
    seg = int(total_size * fraction)
    train_loss = 0.0
    val_loss = 0.0
    train_list = []
    val_list = []
    # 9:1
    train_indices = list(range(0, 9 * seg))
    valid_indices = list(range(9 * seg, total_size))

    train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
    val_set = torch.utils.data.dataset.Subset(dataset, valid_indices)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False)
    train_loss, train_acc_top1, train_acc_top5 = train(model,
                                                       criterion,
                                                       optimizer,
                                                       train_loader,
                                                       switch='Train',
                                                       epoch=1)
    print('train - loss: ' + str(train_loss) + '\t, top1: ' +
          str(train_acc_top1.item() / batch_size) + '\t, top5: ' +
          str(train_acc_top5.item() / batch_size))

    val_loss, valid_acc_top1, valid_acc_top5 = train(model,
                                                     criterion,
                                                     optimizer,
                                                     val_loader,
                                                     switch='Valid')
    print('valid - loss: ' + str(val_loss) + '\t, top1: ' +
          str(valid_acc_top1.item() / batch_size) + '\t, top5: ' +
          str(valid_acc_top5.item() / batch_size))

    return train_loss, val_loss, train_acc_top1.item(
    ) / batch_size, valid_acc_top1.item() / batch_size


# %%
if __name__ == "__main__":
    path = str(pathlib.Path().resolve())
    classes = getClasses()
    training_img_dir = path + '\\training_images\\'
    testing_img_dir = path + '\\testing_images\\'

    bird_dataset = birdDataset.myImageFloder(root='training_images/',
                                             label='training_labels.txt',
                                             transform=train_transform)

    # model = torch.load('model/vit_b_16.pth')
    # model = factory.load_pretrained_vit_b_16()
    model = torch.load('model/swin_best_86-iter11.pth')
    # model = factory.load_pretrained_swin_vit_b()

    training = False
    warmup = False
    epoch = 15
    iter = 0
    train_list = []
    train_acc_list = []
    val_list = []
    val_acc_list = []
    train_loss = 0.0
    val_loss = 0.0
    batch_size = 8
    warmup_lr = 1e-8
    # lr = 0.05
    # lr = 0.001
    lr = 0.0005
    # momentum = 0.5
    momentum = 0.9
    # weight_decay = 0.05
    weight_decay = 0.007
    '''
    # best
    batch_size = 12
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.005
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    warmup_optimizer = torch.optim.SGD(model.parameters(),
                                       lr=warmup_lr,
                                       momentum=momentum,
                                       weight_decay=weight_decay)

    while (True):
        if (training is False):
            break

        iter += 1

        if ((iter <= 5) & (warmup is True)):
            print('epoch = ' + str(iter) + ', lr = ' + str(warmup_lr) +
                  ', momentum = ' + str(momentum) + ', weight_decay = ' +
                  str(weight_decay))

            train_loss, val_loss, t_acc_top1, v_acc_top1 = train_and_valid(
                model,
                criterion,
                warmup_optimizer,
                bird_dataset,
                batch_size=batch_size)

            warmup_lr *= 10
            warmup_optimizer = torch.optim.SGD(model.parameters(),
                                               lr=warmup_lr,
                                               momentum=momentum,
                                               weight_decay=weight_decay)
        else:
            print('epoch = ' + str(iter) + ', lr = ' + str(lr) +
                  ', momentum = ' + str(momentum) + ', weight_decay = ' +
                  str(weight_decay))
            train_loss, val_loss, t_acc_top1, v_acc_top1 = train_and_valid(
                model,
                criterion,
                optimizer,
                bird_dataset,
                batch_size=batch_size)

        train_list.append(train_loss)
        val_list.append(val_loss)
        train_acc_list.append(t_acc_top1)
        val_acc_list.append(v_acc_top1)
        update_lost_hist(train_list,
                         val_list,
                         name='loss compare',
                         xlabel='Loss')
        update_lost_hist(train_acc_list,
                         val_acc_list,
                         name='acc compare',
                         xlabel='acc')
        # torch.save(model, 'model/vit_b_16_iter' + str(iter) + '.pth')

        if ((iter >= 5) | (warmup is not True)):
            torch.save(model, 'model/vit_b_16_iter' + str(iter) + '.pth')
            if ((iter % 5) == 0):
                # torch.save(model, 'model/vit_b_16_iter' + str(iter) + '.pth')
                lr /= 5
                optimizer = torch.optim.SGD(model.parameters(),
                                            lr=lr,
                                            momentum=momentum,
                                            weight_decay=weight_decay)

        if (iter == epoch):
            break

    test = True
    # Run model
    if (test is True):
        with torch.no_grad():
            run_image_classification(model, 'not use', inference_transform,
                                     classes)
