from torchvision import models, transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader
from data import SameDifferentDataset
import torch.nn as nn
import torch
import argparse
import os
from sklearn.metrics import accuracy_score
import wandb
from datetime import datetime


def train_model(model, device, model_type, data_loader, dataset_size, batch_size, optimizer,
                scheduler, num_epochs, log_dir, val_datasets, val_dataloaders, val_labels=None):
    if not val_labels:
        val_labels = list(range(len(val_dataloaders)))

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        model.train()

        running_loss = 0.0
        running_acc = 0.0

        # Iterate over data.
        for bi, (d, f) in enumerate(data_loader):

            if model_type == 'vit':
                inputs = d['pixel_values'].squeeze(1).to(device)
            else:
                inputs = d['image'].to(device)
            labels = d['label'].to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                if model_type == 'vit':
                    outputs = outputs.logits

                loss = criterion(outputs, labels)
                acc = accuracy_score(labels.to('cpu'), outputs.to('cpu').argmax(1))

                loss.backward()
                optimizer.step()

            print('\t({0}/{1}) Batch loss: {2:.4f}'.format(bi + 1, dataset_size // batch_size, loss.item()))
            running_loss += loss.item() * inputs.size(0)
            running_acc += acc * inputs.size(0)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_acc / dataset_size
        print('Epoch loss: {:.4f}'.format(epoch_loss))
        print('Epoch accuracy: {:.4f}'.format(epoch_acc))

        metric_dict = {'epoch': epoch, 'loss': epoch_loss, 'acc': epoch_acc,
                       'lr': scheduler.get_last_lr()[0]}

        scheduler.step()

        # Save the model
        torch.save(model.state_dict(), '{0}/model_{1}.pth'.format(log_dir, epoch))

        # Perform evaluations
        model.eval()
        for i in range(len(val_dataloaders)):
            val_dataloader = val_dataloaders[i]
            val_dataset = val_datasets[i]
            val_label = val_labels[i]

            print()
            print('{0} validation step:'.format(val_label))

            with torch.no_grad():
                running_loss_val = 0.0
                running_acc_val = 0.0

                for bi, (d, f) in enumerate(val_dataloader):
                    print('\t{0}'.format(bi))
                    if model_type == 'vit':
                        inputs = d['pixel_values'].squeeze(1).to(device)
                    else:
                        inputs = d['image'].to(device)
                    labels = d['label'].to(device)

                    outputs = model(inputs)
                    if model_type == 'vit':
                        outputs = outputs.logits

                    loss = criterion(outputs, labels)
                    acc = accuracy_score(labels.to('cpu'), outputs.to('cpu').argmax(1))

                    running_acc_val += acc * inputs.size(0)
                    running_loss_val += loss.item() * inputs.size(0)

                epoch_loss_val = running_loss_val / len(val_dataset)
                epoch_acc_val = running_acc_val / len(val_dataset)

                print()
                print('Validation: {0}'.format(val_label))
                print('Val loss: {:.4f}'.format(epoch_loss_val))
                print('Val acc: {:.4f}'.format(epoch_acc_val))

                metric_dict['val_loss_{0}'.format(val_label)] = epoch_loss_val
                metric_dict['val_acc_{0}'.format(val_label)] = epoch_acc_val

        # Log metrics
        wandb.log(metric_dict)

    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_type', help='Model to train: resnet, vit, clip-rn, clip-vit.',
                    type=str, required=True)
parser.add_argument('--patch_size', type=int, default=32, help='Size of patch (eg. 16 or 32).')
parser.add_argument('--cnn_size', type=int, default=50,
                    help='Number of layers for ResNet (eg. 50 = ResNet-50).', required=False)
parser.add_argument('--k', type=int, default=2, help='Number of objects per scene.')
parser.add_argument('--unaligned', action='store_true', default=False,
                    help='Misalign the objects from ViT patches.')
parser.add_argument('--multiplier', type=int, default=1, help='Factor by which to scale up '
                                                              'stimulus size.')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=64, help='Train/validation batch size.')
parser.add_argument('--feature_extract', action='store_true', default=False,
                    help='Only train the final layer; freeze all other layers..')
parser.add_argument('--optim', type=str, default='adamw',
                    help='Training optimizer, eg. adam, adamw, sgd.')


args = parser.parse_args()

# Parse command line arguments
model_type = args.model_type
fixed = args.fixed
unaligned = args.unaligned
patch_size = args.patch_size
k = args.k
multiplier = args.multiplier
cnn_size = args.cnn_size
num_epochs = args.num_epochs
batch_size = args.batch_size
feature_extract = args.feature_extract
optim = args.optim

if feature_extract:
    fe_string = '_fe'
else:
    fe_string = ''

# Other hyperparameters/variables
im_size = 224
lr = 2e-6  # learning rate
decay_rate = 0.95  # scheduler decay rate
int_to_label = {0: 'different', 1: 'same'}
label_to_int = {'different': 0, 'same': 1}

# Check arguments
assert im_size % patch_size == 0
assert k == 2 or k == 4 or k == 8
assert model_type == 'resnet' or model_type == 'vit'

# Create necessary directories
try:
    os.mkdir('logs')
except FileExistsError:
    pass

if unaligned:
    pos_condition = 'unaligned'
else:
    pos_condition = 'aligned'

if model_type == 'resnet':
    model_str = 'resnet_{0}'.format(cnn_size)

    if cnn_size == 18:
        model = models.resnet18(pretrained=True)
    elif cnn_size == 50:
        model = models.resnet50(pretrained=True)
    elif cnn_size == 152:
        model = models.resnet152(pretrained=True)

    # Freeze layers if feature_extract
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
elif model_type == 'vit':
    model_str = 'vit_b{0}_21k'.format(patch_size)
    model_path = 'google/vit-base-patch{0}-{1}-in21k'.format(patch_size, im_size)

    model = ViTForImageClassification.from_pretrained(
        model_path,
        num_labels=2,
        id2label=int_to_label,
        label2id=label_to_int
    )

    if feature_extract:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    transform = ViTFeatureExtractor.from_pretrained(model_path)

model = model.to(device)  # Move model to GPU if possible
model_str += fe_string  # Add 'fe' if applicable
model_str += '_{0}'.format(optim)  # Optimizer string

path_elements = [model_str, k, pos_condition, patch_size * multiplier]

for root in ['logs']:
    stub = root

    for p in path_elements:
        try:
            os.mkdir('{0}/{1}'.format(stub, p))
        except FileExistsError:
            pass
        stub = '{0}/{1}'.format(stub, p)

log_dir = 'logs/{0}/{1}/{2}/{3}'.format(model_str, k, pos_condition, patch_size * multiplier)
root_dir = 'stimuli/{0}/{1}x{1}/{2}'.format(pos_condition, patch_size * multiplier, k)

# Initialize Weights & Biases project
now = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
wandb_str = '{0}_{1}_{2}_{3}'.format(model_str, pos_condition, patch_size * multiplier, now)
wandb.init(project=wandb_str, name=wandb_str)

wandb.config = {
    'model_type': model_str,
    'learning_rate': lr,
    'gamma': decay_rate,
    'epochs': num_epochs,
    'batch_size': batch_size,
    'optimizer': optim,
    'stimulus_size': '{0}x{0}'.format(patch_size * multiplier),
    'k': k,
    'pos_condition': pos_condition,
}

# Create datasets/dataloaders
train_dataset = SameDifferentDataset(root_dir + '/train', transform=transform)
val_dataset = SameDifferentDataset(root_dir + '/val', transform=transform)
val_dataset_abstract = SameDifferentDataset(
    'stimuli/DEVELOPMENTAL/{0}/{1}x{1}/{2}/val'.format(pos_condition, patch_size * multiplier, k),
    transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
val_dataloader_abstract = DataLoader(val_dataset_abstract, batch_size=batch_size, shuffle=True)

val_datasets = [val_dataset, val_dataset_abstract]
val_dataloaders = [val_dataloader, val_dataloader_abstract]

val_labels = ['in_distribution', 'shape_bias']

# Optimizer and scheduler
if optim == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
elif optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

# Run training loop + evaluations
model = train_model(model, device, model_type, train_dataloader, len(train_dataset), batch_size,
                    optimizer, scheduler, num_epochs, log_dir, val_datasets, val_dataloaders,
                    val_labels)
