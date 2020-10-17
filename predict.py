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
import ai_utils

#############################

parser = argparse.ArgumentParser(
    description='AI prediction command line parser',
)

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('image', type=str, help ='path and filename of image to be analyzed')
parser.add_argument('model', type=str, help ='path and filename of AI model')
parser.add_argument('--category_names', dest="catname", help='Mapping file used to map categories to real names')
parser.add_argument('--top_k', dest="topk", type=int, default=2, help='Return top k most likely classes')
parser.add_argument('--gpu', action="store_true", dest="gpu", help='Use GPU for prediction')
parser.set_defaults(gpu=False)

pa = parser.parse_args()
cli_image = pa.image
cli_model = pa.model
cli_cat = pa.catname
cli_topk = pa.topk
cli_gpu = pa.gpu


#############################

#Load AI Model
model = ai_utils.load_checkpoint(cli_model)

#Normailze image to be analyzed/identified
image = ai_utils.process_image(cli_image)

#Extract label for image to be analyzed, name is based on directory number
label = cli_image.split('/')[-2]

#Translate directory number to flower name
if cli_cat == None:
    cat_to_name = model.cat_to_name
else:
     with open(cli_cat, 'r') as f:
        cat_to_name = json.load(f)
    
image_name = cat_to_name[f'{label}']
print ('correct image name =', image_name)


#Get prediction by feeding image to AI model
probs, classes = ai_utils.predict(cli_image, model, cli_topk)


# Print top k classes and names
print (classes)

# Lookup and print corresponding names
idx_to_class = {v: k for k, v in model.class_to_idx.items()}
names = list(map(lambda x: cat_to_name[f"{idx_to_class[x]}"], classes))
print (names)
