#!/usr/bin/env python
import src.lrp as lrp
import torchvision
import argparse
import os
import sys
from torch.autograd import Variable
from src.variables import *
from PIL import Image, ImageOps
from src.utility_funcs import print_model, make_outputfolder
import numpy as np


def predict_image(image, path):
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    cnn_input = Variable(image_tensor)
    model = torchvision.models.vgg16(pretrained=True)
    model.eval()
    lrp.preform_e_lrp(model, cnn_input, path, image)
    lrp.preform_ye_lrp(model, cnn_input, path, np.squeeze(image))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-te", "--Test", help="Test the Network on local files", action='store_true')
    parser.add_argument("-tea", "--TestAll", help="Test the Network on ImageNetV2 dataset", action='store_true')
    args = parser.parse_args()
    make_outputfolder()
    if not len(sys.argv) > 1:
        print("Please use the flags -te for testing locally, and -tea for testing on ImageNetV2 dataset.")
    files = lrp.get_all_photos(os.path.join(DATA_DIRECTORY, "images"))
    if args.Test:
        print("Preforming LRP on {} images.".format(len(files)))
        for photo, name in files:
            image = Image.open(photo)
            image = ImageOps.exif_transpose(image)
            new_height = 224
            width, height = image.size
            new_width = int (new_height * width / height)
            image = image.resize((new_width, new_height), Image.ANTIALIAS)
            predict_image(image, name)
            print("="*100)
        print("Finished!!! Find files located in data/output")
        exit()
    if args.TestAll:
        lrp.testImgNetV2Images(TEST_DIR, 10)
        exit()

