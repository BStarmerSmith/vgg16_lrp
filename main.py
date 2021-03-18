import src.lrp as lrp
import torchvision
import os
from torch.autograd import Variable
from src.variables import *
from PIL import Image
from src.utility_funcs import print_model, make_outputfolder


def predict_image(image, path):
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    cnn_input = Variable(image_tensor)
    model = torchvision.models.vgg16(pretrained=True)
    model.eval()
    lrp.preform_lrp(model, cnn_input, path, image)


if __name__ == '__main__':
    make_outputfolder()
    files = lrp.get_all_photos(os.path.join(DATA_DIRECTORY, "images"))
    print("Preforming LRP on {} images.".format(len(files)))
    for photo, name in files:
        image = Image.open(photo)
        predict_image(image, name)
        print("="*100)
    print("Finished!!! Find files located in data/output")

