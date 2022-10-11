
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.modules.batchnorm as batchnorm

import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt
import numpy as np



####################################### BASE LINE #######################################
Baselineimages = torchvision.datasets.ImageFolder(root='./processed',transform=transforms.ToTensor())

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

Ch1Mean = Ch1Mean/len(Baselineimages)
Ch1SD = Ch1SD/len(Baselineimages)
Ch2Mean = Ch2Mean/len(Baselineimages)
Ch2SD = Ch2SD/len(Baselineimages)
Ch3Mean = Ch3Mean/len(Baselineimages)
Ch3SD = Ch3SD/len(Baselineimages)

print ("channel1 mean: ",Ch1Mean)
print ("channel2 mean: ",Ch2Mean)
print ("channel3 mean: ",Ch3Mean)
print ("channel1 sd: ",Ch1SD)
print ("channel2 sd: ",Ch2SD)
print ("channel3 sd: ",Ch3SD)


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((Ch1Mean.item(),Ch2Mean.item(),Ch3Mean.item()),(Ch1SD.item(),Ch2SD.item(),Ch3SD.item())),transform.RandomHorizontalFlip()])

Baselineimages = torchvision.datasets.ImageFolder(root='./processed',transform=transform)
BaselineimagesLoader= torch.utils.data.DataLoader(Baselineimages, batch_size=4, shuffle=True, num_workers=2)
Baselineclasses = ('avocados_large','avocados_medium','avocados_small','baby_carrots_large','baby_carrots_medium','baby_carrots_small',
           'cereal_large','cereal_medium','cereal_small','clementine_large','clementine_medium','clementine_small',
           'cookies_large','cookies_medium','cookies_small','sliced_apples_large','sliced_apples_medium','sliced_apple_small',
           'strawberry_large','strawberry_medium','strawberry_small')


lengths= [int(len(Baselineimages)*0.8), int(len(Baselineimages)*0.2)]

baselinetraindata, baselinevaliddata= torch.utils.data.random_split(Baselineimages,lengths)

baselinetrainloader = torch.utils.data.DataLoader(baselinetraindata, batch_size=4,
                                          shuffle=True,num_workers=2)
baselinevalidloader = torch.utils.data.DataLoader(baselinevaliddata, batch_size=4,
                                         shuffle=False,num_workers=2)

###################################### FOOD TYPE ##############################################################


Foodtypeimages = torchvision.datasets.ImageFolder(root='./food type',transform=transforms.ToTensor())

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

Ch1Mean = Ch1Mean/len(Foodtypeimages)
Ch1SD = Ch1SD/len(Foodtypeimages)
Ch2Mean = Ch2Mean/len(Foodtypeimages)
Ch2SD = Ch2SD/len(Foodtypeimages)
Ch3Mean = Ch3Mean/len(Foodtypeimages)
Ch3SD = Ch3SD/len(Foodtypeimages)

print ("channel1 mean: ",Ch1Mean)
print ("channel2 mean: ",Ch2Mean)
print ("channel3 mean: ",Ch3Mean)
print ("channel1 sd: ",Ch1SD)
print ("channel2 sd: ",Ch2SD)
print ("channel3 sd: ",Ch3SD)


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((Ch1Mean.item(),Ch2Mean.item(),Ch3Mean.item()),(Ch1SD.item(),Ch2SD.item(),Ch3SD.item())),transform.RandomHorizontalFlip()])

Foodtypeimages = torchvision.datasets.ImageFolder(root='./food type',transform=transform)
FoodtypeimagesLoader= torch.utils.data.DataLoader(Foodtypeimages, batch_size=4, shuffle=True, num_workers=2)
Foodtypeclasses = ('avocados','baby_carrots', 'cereal','clementine','cookies','sliced_apples','strawberry')


lengths= [int(len(Foodtypeimages)*0.8), int(len(Foodtypeimages)*0.2)]

Foodtypetraindata, Foodtypevaliddata= torch.utils.data.random_split(Foodtypeimages,lengths)

Foodtypetrainloader = torch.utils.data.DataLoader(Foodtypetraindata, batch_size=4,
                                          shuffle=True,num_workers=2)
Foodtypevalidloader = torch.utils.data.DataLoader(Foodtypevaliddata, batch_size=4,
                                         shuffle=False,num_workers=2)


##########################################Portion size########################################


Portionsizeimages = torchvision.datasets.ImageFolder(root='./portion size',transform=transforms.ToTensor())

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

Ch1Mean = Ch1Mean/len(Portionsizeimages)
Ch1SD = Ch1SD/len(Portionsizeimages)
Ch2Mean = Ch2Mean/len(Portionsizeimages)
Ch2SD = Ch2SD/len(Portionsizeimages)
Ch3Mean = Ch3Mean/len(Portionsizeimages)
Ch3SD = Ch3SD/len(Portionsizeimages)

print ("channel1 mean: ",Ch1Mean)
print ("channel2 mean: ",Ch2Mean)
print ("channel3 mean: ",Ch3Mean)
print ("channel1 sd: ",Ch1SD)
print ("channel2 sd: ",Ch2SD)
print ("channel3 sd: ",Ch3SD)


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((Ch1Mean.item(),Ch2Mean.item(),Ch3Mean.item()),(Ch1SD.item(),Ch2SD.item(),Ch3SD.item())),transform.RandomHorizontalFlip()])

Portionsizeimages = torchvision.datasets.ImageFolder(root='./portion size',transform=transform)
PortionsizeimagesLoader= torch.utils.data.DataLoader(Portionsizeimages, batch_size=4, shuffle=True, num_workers=2)
Portionsizeclasses = ('large','medium', 'small')



lengths= [int(len(Portionsizeimages)*0.8), int(len(Portionsizeimages)*0.2)]

portionsizetraindata, portionsizevaliddata= torch.utils.data.random_split(Portionsizeimages,lengths)

portionsizetrainloader = torch.utils.data.DataLoader(portionsizetraindata, batch_size=4,
                                          shuffle=True,num_workers=2)
portionsizevalidloader = torch.utils.data.DataLoader(portionsizevaliddata, batch_size=4,
                                         shuffle=False,num_workers=2)



################################ PRINT SOME IMAGES FOR REPORT#########################################

def imshow(img):
    img = img / 2 + 0.5    #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(FoodtypeimagesLoader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % Foodtypeclasses[labels[j]] for j in range(4)))

