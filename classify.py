import torch
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"

qualityModel = torch.load('roadquality.pth')
qualityModel.eval()

test_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
    ])

# image needs to be in PIL format
def predict_Quality(image):
  image_tensor = test_transforms(image).float()
  image_tensor = image_tensor.unsqueeze_(0)
  input = Variable(image_tensor)
  input = input.to(device)
  output = qualityModel(input)
  index = output.data.cpu().numpy().argmax()
  return index

qualityClasses = ["few cracks rough", "many cracks ruined", "smooth"]

def classify(imagePath):
  
  #image = to_pil(images[ii])
  image = Image.open(imagePath)
  index = predict_Quality(image)
  classification = str(qualityClasses[index])

  return classification


myPath = "C:\\Users\\ssour\\Documents\\surp2023sourishsaswade\\photo\\Fri Aug 11 184651 PDT 2023photoasphaultsmooth.jpg"

print(classify(myPath))
