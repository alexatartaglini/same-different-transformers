from torchvision import models, transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader
from data import SameDifferentDataset, call_create_stimuli
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
        print()

        metric_dict = {'epoch': epoch, 'loss': epoch_loss, 'acc': epoch_acc,
                       'lr': optimizer.param_groups[0]['lr']}

        # Save the model
        torch.save(model.state_dict(), '{0}/model_{1}.pth'.format(log_dir, epoch))

        # Perform evaluations
        model.eval()
        for i in range(len(val_dataloaders)):
            val_dataloader = val_dataloaders[i]
            val_dataset = val_datasets[i]
            val_label = val_labels[i]

            with torch.no_grad():
                running_loss_val = 0.0
                running_acc_val = 0.0

                for bi, (d, f) in enumerate(val_dataloader):
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
                print()

                metric_dict['val_loss_{0}'.format(val_label)] = epoch_loss_val
                metric_dict['val_acc_{0}'.format(val_label)] = epoch_acc_val
                
        scheduler.step(metric_dict['val_acc_{}'.format(val_labels[i])])  # Reduce LR based on validation accuracy

        # Log metrics
        wandb.log(metric_dict)

    return model

# Set device
try:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
except AttributeError:  # if MPS is not available
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
parser.add_argument('--wandb_proj', type=str, default='same-different-transformers',
                    help='Name of WandB project to store the run in.')
parser.add_argument('-vd','--val_datasets', nargs='+', required=False, default=['DEVELOPMENTAL', 'OMNIGLOT'],
                    help='Names of all out-of-distribution stimulus subdirectories to draw validation datasets from.')
parser.add_argument('--n_train', type=int, default=6400, help='Size of training dataset to use.' 
                    'Brady lab: 6400, Developmental: 1024, Omniglot: 2088.')
parser.add_argument('--n_val', type=int, default=640,
                    help='Total # validation stimuli. Brady lab: 640, Developmental: 256, Omniglot: 522.')
parser.add_argument('--n_test', type=int, default=640,
                    help='Total # test stimuli. Brady lab: 640, Developmental: 256, Omniglot: 522.')
parser.add_argument('--n_train_ood', nargs='+', required=False, default=[1024, 2088],
                    help='Size of OOD training sets.')
parser.add_argument('--n_val_ood', nargs='+', required=False, default=[256, 522],
                    help='Size of OOD validation sets.')
parser.add_argument('--n_test_ood', nargs='+', required=False, default=[256, 522],
                    help='Size of OOD test sets.')
parser.add_argument('--rotation', action='store_true', default=False,
                    help='Randomly rotate the objects in the stimuli.')
parser.add_argument('--scaling', action='store_true', default=False,
                    help='Randomly scale the objects in the stimuli.')

args = parser.parse_args()

# Parse command line arguments
model_type = args.model_type
unaligned = args.unaligned
patch_size = args.patch_size
k = args.k
multiplier = args.multiplier
cnn_size = args.cnn_size
num_epochs = args.num_epochs
batch_size = args.batch_size
feature_extract = args.feature_extract
optim = args.optim
wandb_proj = args.wandb_proj
vds = args.val_datasets
n_train = args.n_train
n_val = args.n_val
n_test = args.n_test
n_train_ood = args.n_train_ood
n_val_ood = args.n_val_ood
n_test_ood = args.n_test_ood
rotation = args.rotation
scaling = args.scaling

if model_type == 'vit':
    if patch_size == 16:
        multiplier = multiplier*2

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
assert len(n_train_ood) == len(vds)
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
    
aug_str = ''
if rotation:
    aug_str += 'R'
if scaling:
    aug_str += 'S'
if len(aug_str) == 0:
    aug_str = 'N'

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

path_elements = [model_str, pos_condition, 'trainsize_{}'.format(n_train), 
                 '{}x{}'.format(patch_size * multiplier, patch_size * multiplier), k, aug_str]

for root in ['logs']:
    stub = root

    for p in path_elements:
        try:
            os.mkdir('{0}/{1}'.format(stub, p))
        except FileExistsError:
            pass
        stub = '{0}/{1}'.format(stub, p)

log_dir = 'logs/{0}/{1}/{2}/{3}x{3}/{4}/{5}'.format(model_str, pos_condition, f'trainsize_{n_train}', 
                                                    patch_size * multiplier, k, aug_str)
root_dir = 'stimuli/{0}/{1}/{2}x{2}/{3}/{4}'.format(pos_condition, f'trainsize_{n_train}', 
                                                    patch_size * multiplier, k, aug_str)

if not os.path.exists(root_dir):
    call_create_stimuli(patch_size, n_train, n_val, n_test, k, unaligned, multiplier, 
                        'OBJECTSALL', rotation, scaling)

# Extra information to store
exp_config = {
    'model_type': model_str,
    'learning_rate': lr,
    'gamma': decay_rate,
    'epochs': num_epochs,
    'batch_size': batch_size,
    'optimizer': optim,
    'stimulus_size': '{0}x{0}'.format(patch_size * multiplier),
    'k': k,
    'aug': aug_str,
    'pos_condition': pos_condition,
    'trainsize': n_train,
    'train_device': device
}

# Initialize Weights & Biases project
now = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
wandb_str = '{0}_{1}_{2}_{3}x{3}_{4}_{5}'.format(model_str, pos_condition, f'trainsize{n_train}', 
                                                 patch_size * multiplier, aug_str, now)
wandb.init(project=wandb_proj, name=wandb_str, config=exp_config)

# Create Datasets & DataLoaders
train_dataset = SameDifferentDataset(root_dir + '/train', transform=transform, rotation=rotation, scaling=scaling)
val_dataset = SameDifferentDataset(root_dir + '/val', transform=transform, rotation=rotation, scaling=scaling)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

val_datasets = [val_dataset]
val_dataloaders = [val_dataloader]
val_labels = ['in_distribution']

# Construct OOD validation sets
for v in range(len(vds)):
    val_dir = 'stimuli/{0}/{1}/{2}/{3}x{3}/{4}/{5}/val'.format(vds[v], pos_condition, f'trainsize_{n_train_ood[v]}', 
                                                               patch_size * multiplier, k, aug_str)
    
    if not os.path.exists(val_dir):
        call_create_stimuli(patch_size, n_train, n_val_ood[v], n_test_ood[v], k, unaligned, multiplier, 
                            vds[v], rotation, scaling)
    
    val_dataset_ood = SameDifferentDataset(val_dir, transform=transform, rotation=rotation, scaling=scaling)
    val_dataloader_ood = DataLoader(val_dataset_ood, batch_size=batch_size, shuffle=True)
    
    val_datasets.append(val_dataset_ood)
    val_dataloaders.append(val_dataloader_ood)
    val_labels.append(vds[v].lower())

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
