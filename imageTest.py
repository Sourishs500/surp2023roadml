import torch
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torch import _nnpack_available
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
from pathlib import Path


BATCH_SIZE = 2

cpu_count = os.cpu_count()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = models.resnet50(weights='DEFAULT')
model.to(device)

data_path = Path("photo/")
image_path = data_path 

train_dir = image_path / "train"
test_dir = image_path / "test"


def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# Set seed
random.seed(42) # <- try changing this and see what happens

# 1. Get all image paths (* means "any combination")
# ADD ANOTHER /* TO THIS ONCE YOU ADD CLASSES IN THE TRAIN FOLDER
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem

# 4. Open image
#img = Image.open(random_image_path)


# Write transform for image
train_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(512, 512)),
    # Flip the images randomly on the horizontal (data augmentation)
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

test_transform = transforms.Compose([
    transforms.ToTensor()
    ])

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """
    Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
    # Show the plots
    plt.show()



# Now to turn our image data into a Dataset capable of being used with Pytorch

train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=train_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=test_transform)

class_names = train_data.classes
class_dict = train_data.class_to_idx
train_length = len(train_data)

img, label = train_data[0][0], train_data[0][1]

# Images are now in the form of a tensor (shape [3, 64, 64]) and labels are in form of an integer relating to a specific class (as referenced by class_to_idx)
# Pytorch image dimensions have CHW format, ut matplotlib prefers HWC

# Rearrange the order of dimensions
img_permute = img.permute(1, 2, 0)

#img.shape gives you dimensions of the tensor image

# Turning Dataset into DataLoaders makes them iterable so a model can go through them to learn relationships b/w samples and targets (features and labels)

train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=BATCH_SIZE, # how many samples per batch?
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?


test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=BATCH_SIZE, 
                             num_workers=1, 
                             shuffle=False) # don't usually need to shuffle testing data


if __name__ == '__main__':
    # Freezing pre-trained layers so we don't backprop through them during training
    for param in model.parameters():
        param.requires_grad = False

    # Redefine last layer
    model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 10), nn.LogSoftmax(dim=1))
    model.to(device)

    # Setting loss function and optimizer and learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

    numEpochs = 5
    steps = 0
    running_loss = 0.0
    print_every = 10
    train_losses, test_losses = [], []

    for epoch in range(numEpochs):

        for inputs, labels in train_dataloader:
            steps += 1
            # Put data tensors on the GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # Optimizer zero grad
            optimizer.zero_grad()

            # Forward pass
            logps = model.forward(inputs)

            # Calculate the loss
            loss = criterion(logps, labels)

            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                train_losses.append(running_loss / len(train_dataloader))
                test_losses.append(test_loss / len(test_dataloader))
                print(f"Epoch {epoch+1}/{numEpochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(test_dataloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(test_dataloader):.3f}")
                running_loss = 0
                model.train()

    torch.save(model, 'firstModel.pth')


   

