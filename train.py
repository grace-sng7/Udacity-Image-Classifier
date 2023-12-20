import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from time import time
import argparse
import json

os.environ['PATH'] = f"{os.environ['PATH']}:/root/.local/bin"
os.environ['PATH'] = f"{os.environ['PATH']}:/opt/conda/lib/python3.6/site-packages"

parser = argparse.ArgumentParser(description='Train a neural network on a given data set')
parser.add_argument('--data_dir', type=str, help='path to training data set')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='path to checkpoint directory')
parser.add_argument('--arch', type=str, default='vgg16',
                    help='network architecture (options are vgg16 and densenet121; defaults to vgg16 if not provided or invalid option')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='number of hidden units')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--gpu', action='store_true', help='use GPU')

args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
# model architecture as chosen by user
arch = args.arch
# setting model hyperparameters as chosen by user
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu

print('-------- creating data loaders --------')

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transform = transforms.Compose([transforms.RandomRotation(60),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

valid_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
test_data = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

print('-------- mapping class values to flower names --------')

# load json file to map class values to flower names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

print('-------- creating model --------')


# set model architecture as chosen by user

def get_model(arch):
    ''' get model as selected by user and set the classifier
    '''
    
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(hidden_units, 102),
                                         nn.LogSoftmax(dim=1))
    else:
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(hidden_units, 102),
                                         nn.LogSoftmax(dim=1))

    return model


model = get_model(arch)


# train with gpu or not as chosen by user
if gpu and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")

model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# training network
print('-------- training model --------')
start_time = time()

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    else:
        accuracy = 0
        test_loss = 0

        with torch.no_grad():
            # validation loss and accuracy
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))

        end_time = time()
        tot_time = end_time - start_time
        print("\nTotal Elapsed Runtime:", str(int((tot_time / 3600))) + ":"
              + str(int((tot_time % 3600) / 60)) + ":"
              + str(int((tot_time % 3600) % 60)))

        print("Epoch: {}/{}.. ".format(e + 1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(test_loss / len(valid_loader)),
              "Validation Accuracy: {:.3f}".format(accuracy / len(valid_loader)))

print('-------- calculating accuracy --------')

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy: {:.3f}".format(100 * correct / total))

print('-------- creating checkpoint --------')

checkpoint = {
    'epochs': epochs,
    'learning_rate': learning_rate,
    'arch': arch,
    'hidden_units': hidden_units,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'criterion_state_dict': criterion.state_dict(),
    'class_to_idx': train_data.class_to_idx
}

torch.save(checkpoint, save_dir)