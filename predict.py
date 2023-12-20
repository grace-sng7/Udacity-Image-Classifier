import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import json
import argparse
import importlib

os.environ['PATH'] = f"{os.environ['PATH']}:/root/.local/bin"
os.environ['PATH'] = f"{os.environ['PATH']}:/opt/conda/lib/python3.6/site-packages"

importlib.reload(argparse)
parser = argparse.ArgumentParser(description='predict image')
parser.add_argument('--image_path', type=str, help='image path')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='model checkpoint')
parser.add_argument('--top_k', type=int, default=5, help='top k most likely classes')
parser.add_argument('--category_names', default='cat_to_name.json', help='map of categories to real names')
parser.add_argument('--gpu', action='store_true', help='use GPU')

args = parser.parse_args()

# set script parameters as chosen by user
image_path = args.image_path
checkpoint_path = args.checkpoint
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu

# load json file to map class values to flower names
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# predict with gpu or not as chosen by user
if gpu and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")


# ============= Helper functions ============================

def get_classifier(arch, hidden_units):
    ''' get classifier for model based on user-selected architecture
    '''
    if arch == 'densenet121':
        return nn.Sequential(nn.Linear(1024, hidden_units),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(hidden_units, 102),
                             nn.LogSoftmax(dim=1))

    else:
        return nn.Sequential(nn.Linear(25088, hidden_units),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(hidden_units, 102),
                             nn.LogSoftmax(dim=1))


def load_checkpoint(filepath):
    ''' load model from checkpoint as saved from training and set classifier
    '''
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']

    hidden_units = checkpoint['hidden_units']

    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = get_classifier(arch, hidden_units)
    model.classifier = classifier

    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # load and process a PIL image for use in a PyTorch model
    img = Image.open(image)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
    img = transform(img)

    return img


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.float()
    image = image.to(device)

    output = model.forward(image)
    output = torch.exp(output)
    probability = output.topk(topk)

    top_probs, top_labels = probability
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[i] for i in top_labels]
    top_flowers = [cat_to_name[i] for i in top_labels]

    return top_probs, top_labels, top_flowers


# =====================================

print('-------- loading checkpoint --------')
# load the image classifier from checkpoint
model = load_checkpoint(checkpoint_path)
model.to(device)

print('-------- predicting image --------')
# predict the top K classes of image
probs, labels, flowers = predict(image_path, model, top_k)
# print top K classes with probabilities
print('top probabilities: ', probs)
print('top labels: ', labels)
print('top flower names: ', flowers)
