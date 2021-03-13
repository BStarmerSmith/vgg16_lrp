import src.lrp as lrp
import torchvision
import os
from torch.autograd import Variable
from src.variables import *
from PIL import Image
from src.vgg16_classes import imgclasses
from src.utility_funcs import print_model


def predict_image(image):
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    cnn_input = Variable(image_tensor)
    model = torchvision.models.vgg16(pretrained=True)
    model.eval()
    output = model(cnn_input)
    _, predicted = torch.max(output.data, 1)
    lrp.preform_lrp(model, cnn_input)
    return predicted.sum().item()


if __name__ == '__main__':
    files = lrp.get_all_photos(os.path.join(DATA_DIRECTORY, "images"))
    for photo in files:
        image = Image.open(photo)
        print("This image is {}: {}".format(imgclasses[predict_image(image)], photo))
        print("="*100)

