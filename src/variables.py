from torchvision import transforms
import torch

DATA_DIRECTORY = "data\\"
OUTPUT_PATH = 'data/Output'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape(1,-1,1,1)
STD  = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,-1,1,1)
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])