# Imports here

from collections import OrderedDict
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
import glob, os, sys
import argparse 
import json

#############################

def data_loader(data_dir, batchSize):
    #Module to load image data sets for training and testing the AI model

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #############################

    # TODO: Define your transforms for the training, validation, and testing sets

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batchSize,shuffle=True)   
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batchSize)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batchSize)

    return trainloader, testloader, validloader, train_data.class_to_idx

#############################

def check_model (model, device, testloader, criterion):

    model.eval()
    test_loss = 0
    accuracy = torch.zeros(1)    
    images = torch.zeros(1) 
    labels = torch.zeros(1) 
    with torch.no_grad():
        for images, labels in testloader:
            # Move image and label tensors to the GPU
            images, labels = images.to(device), labels.to(device)
            log_ps = model.forward(images)
            test_loss += criterion(log_ps, labels).item() 
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = (top_class == labels.view(*top_class.shape))
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    
    accuracy1 = accuracy.cpu().numpy()[0]   
    accuracy_percent = 100*accuracy1 / len(testloader)

    return test_loss, accuracy_percent


#############################

def save_checkpoint(model, fileName, optimizer, epochs):
    #Module to save the AI model and associated parameters

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    checkpoint = {'epochs': epochs,
              'model': model,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'cat_to_name': cat_to_name,
              'classifier':model.classifier
             }
    
       
    torch.save(checkpoint, fileName)
    print ("File has been saved as(%s)"%(fileName))
    

#############################

def load_checkpoint(filepath):
    #Module to load a previously saved AI model along with associated parameters

    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer']
    model.epochs = checkpoint['epochs']
    model.cat_to_name = checkpoint['cat_to_name']

    
    for param in model.parameters():
       param.requires_grad = False
        
    return model

#############################

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array'''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    im = Image.open(image)

    im.thumbnail((255,255))
    
    left = (im.width-224)/2
    bottom = (im.height-224)/2
    right = left + 224
    top = bottom + 224
    
    im = im.crop((left, bottom, right, top))
    
    image = np.array(im)/255
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]) 
    image = np.transpose(image, (2, 0, 1))

    return(image)

#############################

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

#############################

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    
    image = torch.FloatTensor([process_image(image_path)])
    image = image.to(device)
    
    logps = model.forward(image)
    
    ps = torch.exp(logps)
    top_p, top_c = ps.topk(topk, dim=1)
  
    probs = top_p.data.cpu().numpy()[0]    
    classes = top_c.data.cpu().numpy()[0]
    
    return(probs, classes)

#############################