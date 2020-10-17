# Imports here

from collections import OrderedDict
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
import glob, os
import argparse 
import argparse 
import json
import ai_utils

#############################

parser = argparse.ArgumentParser(
    description='AI training command line parser',
)

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('datadir', type=str, default="./flowers/", help ='this is the path to the images directory')
parser.add_argument('--save_dir', dest="sdir", default='./checkpoint.pth', help = "path and filename for saving AI model")
parser.add_argument('--arch', dest="arch", choices=('vgg', 'dense'), help = "AI model architecture choice - vgg or densenet")
parser.add_argument('--learning_rate', dest="lr", default=0.005, help = "learning rate, specify between 0 and 1")
parser.add_argument('--hidden', dest="hiddensize", type=int, default=512, help = "size of hidden layer")
parser.add_argument('--epochs', dest="epochs", type=int, default=2, help = "number of times AI model will iterate through learning data")
parser.add_argument('--gpu', action="store_true", dest="gpu", default=False, help = "option for using GPU - will be false unless specified")

pa = parser.parse_args()

if (pa.arch=='help'):
    print ("List of available CNN networks:-")
    print(" vgg (default, vgg11)")
    print(" dense (densenet121)")
    quit()
    
if (not(pa.lr>0 and pa.lr<1)):
    print("Error: Invalid learning rate")
    print("Must be between 0 and 1 exclusive")
    quit()

if (pa.epochs<=0):
    print("Error: Invalid epoch value")
    print("Must be greater than 0")
    quit()    
    
if (pa.hiddensize<=0):
    print("Error: Invalid number of hidden units given")
    print("Must be greater than 0")
    quit()    

arches = ["vgg", "dense"]    

if pa.arch not in arches:
    print("Error: Invalid architecture name received")
    print ("Type \"python train.py -a help\" for more information")
    quit()
    


cli_ddir = pa.datadir
cli_sdir = pa.sdir
cli_arch = pa.arch
cli_lr = pa.lr
cli_hsize = pa.hiddensize
cli_epochs = pa.epochs
cli_gpu = pa.gpu
 

#############################

# TODO: Build and train your network
t_models = {'vgg':models.vgg16(pretrained=True),
            'dense':models.densenet121(pretrained=True)}

model = t_models.get(cli_arch)
classifier = None
optimizer = None
output_size = 102

    
# Only train the classifier parameters, feature parameters are frozen
for param in model.parameters():
    param.requires_grad = False


if cli_arch == "vgg":
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, cli_hsize)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(0.4)),
                          ('fc2', nn.Linear(cli_hsize, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
elif cli_arch == "dense":
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, cli_hsize)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(cli_hsize, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

#############################

if cli_gpu == False:
    device = 'cpu'
    running_on = 'CPU'
elif cli_gpu == True:
    if torch.cuda.is_available():
        device = 'cuda'
        running_on = 'GPU'
    else:
        print('Torch cuda is not available!!')
        device = 'cpu'
        running_on = 'CPU'

print('Torch running on {}.'.format(running_on))        

#############################

batchSize = 32
#build data loaders for training and testing the AI model
trainloader, testloader, validloader, model.class_to_idx =  ai_utils.data_loader(cli_ddir, batchSize)

#############################

import time

print("start")


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=cli_lr)

model.to(device)
epochs = cli_epochs
print_every = 15

for e in range(epochs):
    train_loss = 0
    test_loss = 0
    steps = 0
    inputs = torch.zeros(1) 
    labels = torch.zeros(1)    
    
    for inputs, labels in trainloader: 
        start = time.time()

        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)                
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()         
        train_loss += loss.item()       
                    
        if steps % print_every == 0:    
            test_loss, test_accuracy = ai_utils.check_model (model, device, validloader, criterion)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Train Loss: {:.3f}, ".format(train_loss/print_every),
                    "Validation Loss: {:.3f}.. ".format(test_loss),
                    "Validation Accuracy: %{:.3f}".format(test_accuracy))     
            
            train_loss = 0
    
        model.train()
        torch.set_grad_enabled(True)
        
    print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")
    test_loss, test_accuracy = ai_utils.check_model (model, device, testloader, criterion)
    print("Test Loss: {:.3f}.. ".format(test_loss),
            "Test Accuracy: %{:.3f}".format(test_accuracy))   
        
            
print("finish")       

#############################

#save the model and associated parameters
ai_utils.save_checkpoint(model, cli_sdir, optimizer, epochs)