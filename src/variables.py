from torchvision import transforms
import torch

conv_layers = [("Conv2d-12", 28), ("Conv2d-11", 26), ("Conv2d-10", 24), ("Conv2d-8", 19), ("Conv2d-7", 14), ("Conv2d-6", 12),
                ("Conv2d-5", 10), ("Conv2d-4", 7), ("Conv2d-3", 5), ("Conv2d-2", 2), ("Conv2d-1", 0)]
DATA_DIRECTORY = "data/"
OUTPUT_PATH = 'data/Output'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape(1,-1,1,1)
STD  = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,-1,1,1)
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
TEST_DIR = "data/ImageNetV2/top-images/imagenetv2-top-images-format-val"