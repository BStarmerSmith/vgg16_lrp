from os import listdir, mkdir
from os.path import isfile, join, exists
from src.variables import *
import numpy as np
import torch.nn
import matplotlib.pyplot as plt
import copy
from torchsummary import summary
from src.vgg16_classes import imgclasses


# Returns all the files with their attached paths.
def get_all_photos(path):
    return [(join(path, f), f) for f in listdir(path) if isfile(join(path, f))]


# This function takes a list of Linear layers and converts them to Conv2D layers.
# The layer at index 0 needs to be specifically formatted to deal with the adjustment
# of a 2d network to a 1d network.
# Code from: https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/main/utils.py
def toconv(layers):
    newlayers = []
    for i,layer in enumerate(layers):
        if isinstance(layer, torch.nn.Linear):
            newlayer = None
            if i == 0:
                m,n = 512,layer.weight.shape[0]
                newlayer = torch.nn.Conv2d(m,n,7)
                newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n,m,7,7))
            else:
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = torch.nn.Conv2d(m,n,1)
                newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n,m,1,1))
            newlayer.bias = torch.nn.Parameter(layer.bias)
            newlayers += [newlayer]
        else:
            newlayers += [layer]
    return newlayers


# Code from https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/main/utils.py
def newlayer(layer, g):
    layer = copy.deepcopy(layer)
    try:
        layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError:
        pass
    try:
        layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError:
        pass
    return layer

# -------------------------------------------------------

def show_tensor(tensor, figsize=(8, 4), title=None):
    plt.figure(figsize=figsize)
    plt.matshow(tensor)
    if title: plt.title(title)
    plt.show()


# This function is used to show an image of whatever tensor is presented.
def show_image_tensor(tensor, figsize=(8, 4), title=None):
    tensor = tensor.reshape(28, 28)
    plt.figure(figsize=figsize)
    plt.matshow(tensor, cmap='gray')
    if title: plt.title(title)
    plt.show()


# This function takes in an image, a figure size and a title and presents the image on a chart.
def show_image(image, figsize=(8, 4), title=None):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray')
    if title: plt.title(title)
    plt.show()


# This function is used to print the model structure.
def print_model(model):
    model.eval()
    model.to(device)
    print(summary(model, (3, 224, 224)))
    print(model)


def make_outputfolder():
    if not exists(OUTPUT_PATH):
        mkdir(OUTPUT_PATH)
    e_path = join(OUTPUT_PATH, "e")
    ye_path = join(OUTPUT_PATH, "ye")
    if not exists(e_path):
        mkdir(e_path)
    if not exists(ye_path):
        mkdir(ye_path)



# This function processes the data of the relevancy so its in the correct form
# to be presented.
def process_array(arr, R):
    output = []
    for lable, index in arr:
        output.append((lable, np.array(R[index][0]).sum(axis=0)))
    return output


# This function takes the 5 most likely outputs and presents them as a string for display.
def process_percentage(tuple):
    out_str = ""
    for index, percentage in tuple:
        if percentage == 1.0:
            out_str += "{}: 100% ".format(imgclasses[index])
        else:
            out_str += "{}: {:.5%} ".format(index, percentage)
    return out_str

