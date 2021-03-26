import src.lrp as lrp
import torchvision
import os
from torch.autograd import Variable
from src.variables import *
from PIL import Image, ImageOps
from src.utility_funcs import print_model, make_outputfolder


def predict_image(image, path):
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    cnn_input = Variable(image_tensor)
    model = torchvision.models.vgg16(pretrained=True)
    model.eval()
    lrp.preform_e_lrp(model, cnn_input, path, image)
    lrp.preform_ye_lrp(model, cnn_input, path, image)


if __name__ == '__main__':
    make_outputfolder()
    files = lrp.get_all_photos(os.path.join(DATA_DIRECTORY, "images"))
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

