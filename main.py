import torch
import torch.nn
import torch.optim as optim

import argparse
import os

from baseline import *
from model_first_half import *
from model_second_half import *

import numpy as np
import matplotlib.pyplot as plt

from time import time

from torch.utils.data import DataLoader
from torchsummary import summary

import torch
import torchvision
import torchvision.transforms as transforms

from random import randint
from random import seed

from sklearn.metrics import confusion_matrix


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format('CUDA' if torch.cuda.is_available() else 'CPU'))

s = 1
torch.manual_seed(s)
seed(s)

change_CJ = float(randint(1, 10))/float(randint(1, 10))

total_images = 1200  # Only for training and validation, remaining 300 are in another folder
tsplit = 0.8  # 0.64 of total data set
vsplit = 0.2  # 0.16 of total data set

# ------------------------------------------- BASELINE ------------------------------------------- #
Baselineimages = torchvision.datasets.ImageFolder(root='./processed', transform=transforms.ToTensor())

Ch1Mean = 0.0
Ch1SD = 0.0
Ch2Mean = 0.0
Ch2SD = 0.0
Ch3Mean = 0.0
Ch3SD = 0.0

for img in Baselineimages:
    Ch1Mean += img[0][0].mean()
    Ch1SD += img[0][0].std()
    Ch2Mean += img[0][1].mean()
    Ch2SD += img[0][1].std()
    Ch3Mean += img[0][2].mean()
    Ch3SD += img[0][2].std()

Ch1Mean = Ch1Mean / len(Baselineimages)
Ch1SD = Ch1SD / len(Baselineimages)
Ch2Mean = Ch2Mean / len(Baselineimages)
Ch2SD = Ch2SD / len(Baselineimages)
Ch3Mean = Ch3Mean / len(Baselineimages)
Ch3SD = Ch3SD / len(Baselineimages)

# print ("channel1 mean: %f",Ch1Mean)
# print ("channel2 mean: %f",Ch2Mean)
# print ("channel3 mean: %f",Ch3Mean)
# print ("channel1 sd: %f",Ch1SD)
# print ("channel2 sd: %f",Ch2SD)
# print ("channel3 sd: %f",Ch3SD)


transform = transforms.Compose([transforms.ToTensor(), transforms.ColorJitter(brightness=change_CJ, contrast=change_CJ,
                                                                              saturation=change_CJ, hue=change_CJ),
                                transforms.Normalize((Ch1Mean.item(),
                                                      Ch2Mean.item(), Ch3Mean.item()),
                                                     (Ch1SD.item(), Ch2SD.item(), Ch3SD.item()))])


Baselineimages = torchvision.datasets.ImageFolder(root='./processed', transform=transform)
BaselineimagesLoader = torch.utils.data.DataLoader(Baselineimages, batch_size=8, shuffle=True)

# Baselineclasses = ('avocados_large', 'avocados_medium', 'avocados_small', 'baby_carrots_large', 'baby_carrots_medium',
#                    'baby_carrots_small',
#                    'cereal_large', 'cereal_medium', 'cereal_small', 'clementine_large', 'clementine_medium',
#                    'clementine_small',
#                    'cookies_large', 'cookies_medium', 'cookies_small', 'sliced_apples_large', 'sliced_apples_medium',
#                    'sliced_apple_small',
#                    'strawberry_large', 'strawberry_medium', 'strawberry_small')

lengths = [int(len(Baselineimages) * tsplit), int(len(Baselineimages) * vsplit)]

baselinetraindata, baselinevaliddata = torch.utils.data.random_split(Baselineimages, lengths)

baselinetrainloader = torch.utils.data.DataLoader(baselinetraindata, batch_size=8,
                                                  shuffle=True)
baselinevalidloader = torch.utils.data.DataLoader(baselinevaliddata, batch_size=8,
                                                  shuffle=False)


Baselinetest = torchvision.datasets.ImageFolder(root='./processed_test', transform=transforms.ToTensor())

Ch1Mean = 0.0
Ch1SD = 0.0
Ch2Mean = 0.0
Ch2SD = 0.0
Ch3Mean = 0.0
Ch3SD = 0.0

for img in Baselinetest:
    Ch1Mean += img[0][0].mean()
    Ch1SD += img[0][0].std()
    Ch2Mean += img[0][1].mean()
    Ch2SD += img[0][1].std()
    Ch3Mean += img[0][2].mean()
    Ch3SD += img[0][2].std()

Ch1Mean = Ch1Mean / len(Baselinetest)
Ch1SD = Ch1SD / len(Baselinetest)
Ch2Mean = Ch2Mean / len(Baselinetest)
Ch2SD = Ch2SD / len(Baselinetest)
Ch3Mean = Ch3Mean / len(Baselinetest)
Ch3SD = Ch3SD / len(Baselinetest)


transformtest = transforms.Compose([transforms.ToTensor(), transforms.ColorJitter(brightness=change_CJ, contrast=change_CJ,
                                                                                  saturation=change_CJ, hue=change_CJ),
                                transforms.Normalize((Ch1Mean.item(),
                                                      Ch2Mean.item(), Ch3Mean.item()),
                                                     (Ch1SD.item(), Ch2SD.item(), Ch3SD.item()))])


Baselinetest = torchvision.datasets.ImageFolder(root='./processed_test', transform=transformtest)
BaselinetestLoader = torch.utils.data.DataLoader(Baselinetest, batch_size=8, shuffle=False)

confbaseline = torch.utils.data.DataLoader(Baselinetest, batch_size=int(len(Baselinetest)), shuffle=False)

# ------------------------------------------- FOOD TYPE ------------------------------------------- #
Foodtypeimages = torchvision.datasets.ImageFolder(root='./food_type', transform=transforms.ToTensor())

Ch1Mean = 0.0
Ch1SD = 0.0
Ch2Mean = 0.0
Ch2SD = 0.0
Ch3Mean = 0.0
Ch3SD = 0.0

for img in Foodtypeimages:
    Ch1Mean += img[0][0].mean()
    Ch1SD += img[0][0].std()
    Ch2Mean += img[0][1].mean()
    Ch2SD += img[0][1].std()
    Ch3Mean += img[0][2].mean()
    Ch3SD += img[0][2].std()

Ch1Mean = Ch1Mean / len(Foodtypeimages)
Ch1SD = Ch1SD / len(Foodtypeimages)
Ch2Mean = Ch2Mean / len(Foodtypeimages)
Ch2SD = Ch2SD / len(Foodtypeimages)
Ch3Mean = Ch3Mean / len(Foodtypeimages)
Ch3SD = Ch3SD / len(Foodtypeimages)

# print("channel1 mean: ", Ch1Mean)
# print("channel2 mean: ", Ch2Mean)
# print("channel3 mean: ", Ch3Mean)
# print("channel1 sd: ", Ch1SD)
# print("channel2 sd: ", Ch2SD)
# print("channel3 sd: ", Ch3SD)

transform = transforms.Compose([transforms.ToTensor(), transforms.ColorJitter(brightness=change_CJ, contrast=change_CJ,
                                                                              saturation=change_CJ, hue=change_CJ),
                                transforms.Normalize((Ch1Mean.item(), Ch2Mean.item(),
                                                      Ch3Mean.item()), (Ch1SD.item(), Ch2SD.item(), Ch3SD.item()))])

Foodtypeimages = torchvision.datasets.ImageFolder(root='./food_type', transform=transform)
FoodtypeimagesLoader = torch.utils.data.DataLoader(Foodtypeimages, batch_size=8, shuffle=True)
Foodtypeclasses = ('avocados', 'baby_carrots', 'cereal', 'clementine', 'cookies', 'sliced_apples', 'strawberry')

lengths = [int(len(Foodtypeimages) * tsplit), int(len(Foodtypeimages) * vsplit)]

Foodtypetraindata, Foodtypevaliddata = torch.utils.data.random_split(Foodtypeimages, lengths)

foodtypetrainloader = torch.utils.data.DataLoader(Foodtypetraindata, batch_size=8,
                                                  shuffle=True)
foodtypevalidloader = torch.utils.data.DataLoader(Foodtypevaliddata, batch_size=8,
                                                  shuffle=False)


typetest = torchvision.datasets.ImageFolder(root='./processed_test1', transform=transforms.ToTensor())

Ch1Mean = 0.0
Ch1SD = 0.0
Ch2Mean = 0.0
Ch2SD = 0.0
Ch3Mean = 0.0
Ch3SD = 0.0

for img in typetest:
    Ch1Mean += img[0][0].mean()
    Ch1SD += img[0][0].std()
    Ch2Mean += img[0][1].mean()
    Ch2SD += img[0][1].std()
    Ch3Mean += img[0][2].mean()
    Ch3SD += img[0][2].std()

Ch1Mean = Ch1Mean / len(typetest)
Ch1SD = Ch1SD / len(typetest)
Ch2Mean = Ch2Mean / len(typetest)
Ch2SD = Ch2SD / len(typetest)
Ch3Mean = Ch3Mean / len(typetest)
Ch3SD = Ch3SD / len(typetest)


transformtypetest = transforms.Compose([transforms.ToTensor(), transforms.ColorJitter(brightness=change_CJ, contrast=change_CJ,
                                                                                      saturation=change_CJ, hue=change_CJ),
                                        transforms.Normalize((Ch1Mean.item(),
                                                            Ch2Mean.item(), Ch3Mean.item()),
                                                            (Ch1SD.item(), Ch2SD.item(), Ch3SD.item()))])


typetest = torchvision.datasets.ImageFolder(root='./processed_test1', transform=transformtypetest)
typetestLoader = torch.utils.data.DataLoader(typetest, batch_size=8, shuffle=False)

conftype = torch.utils.data.DataLoader(typetest, batch_size=int(len(typetest)), shuffle=False)

# ------------------------------------------- FOOD SIZE ------------------------------------------- #
Portionsizeimages = torchvision.datasets.ImageFolder(root='./portion_size', transform=transforms.ToTensor())

Ch1Mean = 0.0
Ch1SD = 0.0
Ch2Mean = 0.0
Ch2SD = 0.0
Ch3Mean = 0.0
Ch3SD = 0.0

for img in Portionsizeimages:
    Ch1Mean += img[0][0].mean()
    Ch1SD += img[0][0].std()
    Ch2Mean += img[0][1].mean()
    Ch2SD += img[0][1].std()
    Ch3Mean += img[0][2].mean()
    Ch3SD += img[0][2].std()

Ch1Mean = Ch1Mean / len(Portionsizeimages)
Ch1SD = Ch1SD / len(Portionsizeimages)
Ch2Mean = Ch2Mean / len(Portionsizeimages)
Ch2SD = Ch2SD / len(Portionsizeimages)
Ch3Mean = Ch3Mean / len(Portionsizeimages)
Ch3SD = Ch3SD / len(Portionsizeimages)

# print("channel1 mean: ", Ch1Mean)
# print("channel2 mean: ", Ch2Mean)
# print("channel3 mean: ", Ch3Mean)
# print("channel1 sd: ", Ch1SD)
# print("channel2 sd: ", Ch2SD)
# print("channel3 sd: ", Ch3SD)

transform = transforms.Compose([transforms.ToTensor(), transforms.ColorJitter(brightness=change_CJ, contrast=change_CJ,
                                                                              saturation=change_CJ, hue=change_CJ),
                                transforms.Normalize((Ch1Mean.item(),
                                                      Ch2Mean.item(), Ch3Mean.item()),
                                                     (Ch1SD.item(), Ch2SD.item(), Ch3SD.item()))])

Portionsizeimages = torchvision.datasets.ImageFolder(root='./portion_size', transform=transform)
PortionsizeimagesLoader = torch.utils.data.DataLoader(Portionsizeimages, batch_size=8, shuffle=True)
Portionsizeclasses = ('large', 'medium', 'small')

lengths = [int(len(Portionsizeimages) * tsplit), int(len(Portionsizeimages) * vsplit)]

portionsizetraindata, portionsizevaliddata = torch.utils.data.random_split(Portionsizeimages, lengths)

portionsizetrainloader = torch.utils.data.DataLoader(portionsizetraindata, batch_size=8,
                                                     shuffle=True)
portionsizevalidloader = torch.utils.data.DataLoader(portionsizevaliddata, batch_size=8,
                                                     shuffle=False)


sizetest = torchvision.datasets.ImageFolder(root='./processed_test2', transform=transforms.ToTensor())

Ch1Mean = 0.0
Ch1SD = 0.0
Ch2Mean = 0.0
Ch2SD = 0.0
Ch3Mean = 0.0
Ch3SD = 0.0

for img in sizetest:
    Ch1Mean += img[0][0].mean()
    Ch1SD += img[0][0].std()
    Ch2Mean += img[0][1].mean()
    Ch2SD += img[0][1].std()
    Ch3Mean += img[0][2].mean()
    Ch3SD += img[0][2].std()

Ch1Mean = Ch1Mean / len(sizetest)
Ch1SD = Ch1SD / len(sizetest)
Ch2Mean = Ch2Mean / len(sizetest)
Ch2SD = Ch2SD / len(sizetest)
Ch3Mean = Ch3Mean / len(sizetest)
Ch3SD = Ch3SD / len(sizetest)


transformsizetest = transforms.Compose([transforms.ToTensor(), transforms.ColorJitter(brightness=change_CJ, contrast=change_CJ,
                                                                                      saturation=change_CJ, hue=change_CJ),
                                        transforms.Normalize((Ch1Mean.item(),
                                                            Ch2Mean.item(), Ch3Mean.item()),
                                                            (Ch1SD.item(), Ch2SD.item(), Ch3SD.item()))])


sizetest = torchvision.datasets.ImageFolder(root='./processed_test2', transform=transformsizetest)
sizetestLoader = torch.utils.data.DataLoader(sizetest, batch_size=8, shuffle=False)

confsize = torch.utils.data.DataLoader(sizetest, batch_size=int(len(sizetest)), shuffle=False)


def count_predictions(outputs, labels):
    acc = 0
    for i in range(len(outputs)):
        if torch.argmax(outputs[i]) == labels[i]:
            acc += 1
    return acc / len(outputs)


def evaluate(model, load, criterion):
    total_corr = 0
    total_loss = 0.0

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(load):
            feats, labels = batch
            feats = feats.to(device)
            labels = labels.to(device)

            prediction = model(feats.float())
            corr = torch.argmax(prediction, dim=1) == labels
            total_corr += int(corr.sum())
            loss = criterion(prediction, labels)
            total_loss += loss
        total_loss = total_loss / i

    model.train()

    return total_loss, float(total_corr) / (total_images * vsplit)


def plot_acc(a, steps, x, y):
    acc = np.asarray(a)
    step = np.asarray(steps)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(step, acc, color='red', linewidth=0.8)
    ax.set(title='Accuracy vs. {}'.format(y), ylabel='Accuracy', xlabel=y)
    ax.legend([x])
    plt.show()


def plot_loss(l, steps, x, y):
    loss = np.asarray(l)
    step = np.asarray(steps)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(step, loss, color='red', linewidth=0.8)
    ax.set(title='Loss vs. {}'.format(y), ylabel='Loss', xlabel=y)
    ax.legend([x])
    plt.show()


def plot_acc_1(ta, va, steps):
    tacc = np.asarray(ta)
    vacc = np.asarray(va)
    step = np.asarray(steps)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(step, tacc, color='red', linewidth=0.8)
    ax.plot(step, vacc, color='blue', linewidth=0.8)
    ax.set(title='Accuracy vs. Epochs', ylabel='Accuracy', xlabel='Epochs')
    ax.legend(['Training', 'Validation'])
    plt.show()


def plot_loss_1(ta, va, steps):
    tloss = np.asarray(ta)
    vloss = np.asarray(va)
    step = np.asarray(steps)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(step, tloss, color='red', linewidth=0.8)
    ax.plot(step, vloss, color='blue', linewidth=0.8)
    ax.set(title='Loss vs. Epochs', ylabel='Loss', xlabel='Epochs')
    ax.legend(['Training', 'Validation'])
    plt.show()


def conf_matrix(model, val_loader, criterion):
    pred = []
    true = []

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            feats, label = batch
            feats = feats.to(device)
            label = label.to(device)
            prediction = model(feats.float())

            labels = label.tolist()
            predictions = (torch.argmax(prediction, dim=1)).tolist()
            for j in range(0, len(labels)):
                true.append(labels[j])
                pred.append(predictions[j])

    model.train()

    result = confusion_matrix(true, pred)

    return result


def main(args):
    e = args.epochs
    print('Batch size: {}  Learning rate: {}  Epochs: {}'.format(args.batch_size, args.lr, e))
    if args.model == 'baseline':
        print('Model: {}'.format('Baseline'))
    elif args.model == 'model_type':
        print('Model: {}'.format('Food Type'))
    else:
        print('Model: {}'.format('Portion Size'))
    print('Saving model? {}'.format('Yes' if args.save_model == 1 else 'No'))
    print(' ')

    init_time = time()

    # Baseline
    if args.model == 'baseline':
        net = Baseline(args.input, args.first, args.second, args.third, args.output).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

        tacc = []
        tloss = []
        vacc = []
        vloss = []
        epochs = []
        times = []

        for epoch in range(0, e):
            running_loss = 0.0
            t_temp_acc = []
            for i, data in enumerate(baselinetrainloader, 0):
                # print(i)
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                t_temp_acc.append(count_predictions(outputs, labels))

                running_loss += loss.item()

            v_temp_loss, v_temp_acc = evaluate(net, baselinevalidloader, criterion)

            epochs.append(epoch)

            tloss.append(running_loss / (total_images * tsplit))
            tacc.append(sum(t_temp_acc) / len(t_temp_acc))

            vloss.append(v_temp_loss)
            vacc.append(v_temp_acc)

            current_time = time() - init_time
            times.append(current_time)

            print('Epoch: {}'.format(epoch + 1))
            print('Training Accuracy: {}  Training Loss: {}'.format(tacc[len(tacc) - 1], tloss[len(tloss) - 1]))
            print('Validation Accuracy: {}  Validation Loss: {}'.format(vacc[len(vacc) - 1], vloss[len(vloss) - 1]))
            print('Time Elapsed: {}'.format(current_time))

            if epoch == e-1:
                c_matrix = conf_matrix(net, confbaseline, criterion)

        print('Confusion Matrix')
        print(c_matrix)

        if args.plot == 1:  # vs. epoch
            plot_acc(tacc, epochs, 'Training', 'Epochs')
            plot_loss(tloss, epochs, 'Training', 'Epochs')
            plot_acc(vacc, epochs, 'Validation', 'Epochs')
            plot_loss(vloss, epochs, 'Validation', 'Epochs')
            plot_acc_1(tacc, vacc, epochs)
            plot_loss_1(tloss, vloss, epochs)
        elif args.plot == 2:  # vs. time
            plot_acc(tacc, times, 'Training', 'Time')
            plot_loss(tloss, times, 'Training', 'Time')
            plot_acc(vacc, times, 'Validation', 'Time')
            plot_loss(vloss, times, 'Validation', 'Time')

        if args.save_model == 1:
            torch.save(net, 'baseline.pt')

        summary(net, (3, 128, 128))

    # Model (Food type)
    elif args.model == 'model_type':
        net_1 = model_1(args.k1, args.k2, args.k3, args.k4, args.lt1, args.lt2, args.outputs_t).to(device)
        criterion_1 = nn.CrossEntropyLoss()
        optimizer_1 = torch.optim.SGD(net_1.parameters(), lr=args.lr)

        tacc = []
        tloss = []
        vacc = []
        vloss = []
        epochs = []
        times = []

        for epoch in range(0, e):
            running_loss = 0.0
            t_temp_acc = []
            for i, data in enumerate(foodtypetrainloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer_1.zero_grad()

                outputs = net_1(inputs)
                loss = criterion_1(outputs, labels)
                loss.backward()
                optimizer_1.step()

                t_temp_acc.append(count_predictions(outputs, labels))

                running_loss += loss.item()

            v_temp_loss, v_temp_acc = evaluate(net_1, foodtypevalidloader, criterion_1)

            epochs.append(epoch)

            tloss.append(running_loss / (total_images * tsplit))
            tacc.append(sum(t_temp_acc) / len(t_temp_acc))

            vloss.append(v_temp_loss)
            vacc.append(v_temp_acc)

            current_time = time() - init_time
            times.append(current_time)

            print('Epoch: {}'.format(epoch + 1))
            print('Training Accuracy: {}  Training Loss: {}'.format(tacc[len(tacc) - 1], tloss[len(tloss) - 1]))
            print('Validation Accuracy: {}  Validation Loss: {}'.format(vacc[len(vacc) - 1], vloss[len(vloss) - 1]))
            print('Time Elapsed: {}'.format(current_time))

            if epoch == e-1:
                c_matrix = conf_matrix(net_1, conftype, criterion_1)

        print("Confusion Matrix")
        print(c_matrix)

        if args.plot == 1:  # vs. epoch
            plot_acc(tacc, epochs, 'Training', 'Epochs')
            plot_loss(tloss, epochs, 'Training', 'Epochs')
            plot_acc(vacc, epochs, 'Validation', 'Epochs')
            plot_loss(vloss, epochs, 'Validation', 'Epochs')
            plot_acc_1(tacc, vacc, epochs)
            plot_loss_1(tloss, vloss, epochs)
        elif args.plot == 2:  # vs. time
            plot_acc(tacc, times, 'Training', 'Time')
            plot_loss(tloss, times, 'Training', 'Time')
            plot_acc(vacc, times, 'Validation', 'Time')
            plot_loss(vloss, times, 'Validation', 'Time')

        if args.save_model == 1:
            torch.save(net_1, 'model_type.pt')

        summary(net_1, (3, 128, 128))

        # Model (Food size)
    else:
        net_2 = model_2(args.ks1, args.ks2, args.ks3, args.ks4, args.ks5, args.ls1, args.ls2, args.outputs_t).to(device)
        criterion_2 = nn.CrossEntropyLoss()
        optimizer_2 = torch.optim.SGD(net_2.parameters(), lr=args.lr)

        tacc = []
        tloss = []
        vacc = []
        vloss = []
        epochs = []
        times = []

        for epoch in range(0, e):
            running_loss = 0.0
            t_temp_acc = []
            for i, data in enumerate(portionsizetrainloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer_2.zero_grad()

                outputs = net_2(inputs)
                loss = criterion_2(outputs, labels)
                loss.backward()
                optimizer_2.step()

                t_temp_acc.append(count_predictions(outputs, labels))

                running_loss += loss.item()

            v_temp_loss, v_temp_acc = evaluate(net_2, portionsizevalidloader, criterion_2)

            epochs.append(epoch)

            tloss.append(running_loss / (total_images * tsplit))
            tacc.append(sum(t_temp_acc) / len(t_temp_acc))

            vloss.append(v_temp_loss)
            vacc.append(v_temp_acc)

            current_time = time() - init_time
            times.append(current_time)

            print('Epoch: {}'.format(epoch + 1))
            print('Training Accuracy: {}  Training Loss: {}'.format(tacc[len(tacc) - 1], tloss[len(tloss) - 1]))
            print('Validation Accuracy: {}  Validation Loss: {}'.format(vacc[len(vacc) - 1], vloss[len(vloss) - 1]))
            print('Time Elapsed: {}'.format(current_time))

            if epoch == e - 1:
                c_matrix = conf_matrix(net_2, confsize, criterion_2)

        print("Confusion Matrix")
        print(c_matrix)

        if args.plot == 1:  # vs. epoch
            plot_acc(tacc, epochs, 'Training', 'Epochs')
            plot_loss(tloss, epochs, 'Training', 'Epochs')
            plot_acc(vacc, epochs, 'Validation', 'Epochs')
            plot_loss(vloss, epochs, 'Validation', 'Epochs')
            plot_acc_1(tacc, vacc, epochs)
            plot_loss_1(tloss, vloss, epochs)
        elif args.plot == 2:  # vs. time
            plot_acc(tacc, times, 'Training', 'Time')
            plot_loss(tloss, times, 'Training', 'Time')
            plot_acc(vacc, times, 'Validation', 'Time')
            plot_loss(vloss, times, 'Validation', 'Time')

        if args.save_model == 1:
            torch.save(net_2, 'model_size.pt')

        summary(net_2, (3, 128, 128))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0025)
    parser.add_argument('--epochs', type=int, default=40)

    parser.add_argument('--model', type=str, default='model_size',
                        help="Model type: baseline, model_type, model_size")

    parser.add_argument('--save_model', type=int, default=0, help="0 for don't save, 1 for save")

    parser.add_argument('--plot', type=int, default=1, help="1 for plot vs epoch, 2 for vs time")

    # Baseline
    parser.add_argument('-input', help='MLP in', type=int, default=49152)  # 128 * 128 * 3 images
    parser.add_argument('-first', help='MLP first layer', type=int, default=12288)
    parser.add_argument('-second', help='MLP second layer', type=int, default=8192)
    parser.add_argument('-third', help='MLP third layer', type=int, default=6144)
    parser.add_argument('-output', help='MLP out', type=int, default=30)

    # Model (Food Type)
    parser.add_argument('-k1', help='# Kernels first', type=int, default=64)
    parser.add_argument('-k2', help='# Kernels second', type=int, default=128)
    parser.add_argument('-k3', help='# Kernels third', type=int, default=256)
    parser.add_argument('-k4', help='# Kernels fourth', type=int, default=192)
    parser.add_argument('-lt1', help='MLP first', type=int, default=6144)
    parser.add_argument('-lt2', help='MLP second', type=int, default=4092)
    parser.add_argument('-outputs_t', help='outputs of first conv. NN', type=int, default=10)

    # Model (Portion Size)
    parser.add_argument('-ks1', help='# Kernels first', type=int, default=160)
    parser.add_argument('-ks2', help='# Kernels second', type=int, default=256)
    parser.add_argument('-ks3', help='# Kernels third', type=int, default=384)
    parser.add_argument('-ks4', help='# Kernels fourth', type=int, default=512)
    parser.add_argument('-ks5', help='# Kernels fifth', type=int, default=512)
    parser.add_argument('-ls1', help='MLP first', type=int, default=8192)
    parser.add_argument('-ls2', help='MLP second', type=int, default=6144)
    parser.add_argument('-outputs_s', help='outputs of first conv. NN', type=int, default=3)

    args = parser.parse_args()

    main(args)
