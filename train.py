from torchvision import models, transforms
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
#CLIPProcessor, CLIPModel, CLIPConfig
import clip
from torch.utils.data import DataLoader
from data import SameDifferentDataset, call_create_stimuli
import torch.nn as nn
import torch
import argparse
import os
from sklearn.metrics import accuracy_score
import wandb
import numpy as np
import sys


def train_model(args, model, device, data_loader, dataset_size, optimizer,
                scheduler, log_dir, val_datasets, val_dataloaders, 
                test_table, val_labels=None):
    
    if not val_labels:
        val_labels = list(range(len(val_dataloaders)))
    
    int_to_label = {0: 'different', 1: 'same'}
    
    model_type = args.model_type
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    save_model_freq = args.save_model_freq
    log_preds_freq = args.log_preds_freq
    
    save_model_epochs = np.linspace(0, num_epochs, save_model_freq, dtype=int)
    log_preds_epochs = np.linspace(0, num_epochs, log_preds_freq, dtype=int)

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
        if epoch in save_model_epochs:
            torch.save(model.state_dict(), f'{log_dir}/model_{epoch}_{lr}_{wandb.run.id}.pth')

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
                    preds = outputs.argmax(1)
                    acc = accuracy_score(labels.to('cpu'), preds.to('cpu'))
                    
                    # Log error examples
                    if epoch in log_preds_epochs:
                        error_idx = (labels + preds == 1).cpu()
                        error_ims = inputs[error_idx, :, :, :]
                        error_paths = [name.split('/')[-1] for name in np.asarray(list(f), dtype=object)[error_idx]]
                        error_preds = [int_to_label[p.item()] for p in preds[error_idx]]
                        error_truths = [int_to_label[l.item()] for l in labels[error_idx]]
                        same_scores = outputs[error_idx, 0]
                        diff_scores = outputs[error_idx, 1]
                        same_acc = len(labels[labels + preds == 2]) / len(labels[labels == 1])
                        diff_acc = len(labels[labels + preds == 0]) / len(labels[labels == 0])
                        for j in range(len(same_scores)):
                            test_table.add_data(epoch, error_paths[j], wandb.Image(error_ims[j, :, :, :]),
                                                val_label, error_preds[j], error_truths[j], same_scores[j], 
                                                diff_scores[j], same_acc, diff_acc)

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
           
        if epoch in log_preds_epochs:
            test_data_at = wandb.Artifact(f'test_errors_{run_id}_{epoch}', type='predictions')
            test_data_at.add(test_table, 'predictions')
            wandb.run.log_artifact(test_data_at).wait() 
            
        scheduler.step(metric_dict[f'val_acc_{val_labels[0]}'])  # Reduce LR based on validation accuracy

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
parser.add_argument('--wandb_proj', type=str, default='same-different-transformers',
                    help='Name of WandB project to store the run in.')

# Model/architecture arguments
parser.add_argument('-m', '--model_type', help='Model to train: resnet, vit, clip_rn, clip_vit, small_cnn.',
                    type=str, required=True)
parser.add_argument('--patch_size', type=int, default=32, help='Size of patch (eg. 16 or 32).')
parser.add_argument('--cnn_size', type=int, default=50,
                    help='Number of layers for ResNet (eg. 50 = ResNet-50).', required=False)
parser.add_argument('--feature_extract', action='store_true', default=False,
                    help='Only train the final layer; freeze all other layers.')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Use ImageNet pretrained models. If false, models are trained from scratch.')

# Training arguments
parser.add_argument('-td', '--train_datasets', nargs='+', required=False, 
                    help='Names of all stimulus subdirectories to draw train stimuli from.', 
                        default=['OBJECTSALL'])
parser.add_argument('-vd','--val_datasets', nargs='+', required=False, 
                    default='all',
                    help='Names of all stimulus subdirectories to draw validation datasets from. \
                        Input "all" in order to test on all existing sets.')
parser.add_argument('--optim', type=str, default='adamw',
                    help='Training optimizer, eg. adam, adamw, sgd.')
parser.add_argument('--lr', default=2e-6, help='Learning rate.')
parser.add_argument('--lr_scheduler', default='reduce_on_plateau', help='LR scheduler.')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=64, help='Train/validation batch size.')

# Stimulus arguments
parser.add_argument('--k', type=int, default=2, help='Number of objects per scene.')
parser.add_argument('--rotation', action='store_true', default=False,
                    help='Randomly rotate the objects in the stimuli.')
parser.add_argument('--scaling', action='store_true', default=False,
                    help='Randomly scale the objects in the stimuli.')
parser.add_argument('--unaligned', action='store_true', default=True,
                    help='Misalign the objects from ViT patches.')
parser.add_argument('--multiplier', type=int, default=2, help='Factor by which to scale up \
                    stimulus size. For example, if patch_size=32 and multiplier=2, the \
                    stimuli will be 64x64.')

# Dataset size arguments
parser.add_argument('--n_train', type=int, default=6400, help='Size of training dataset to use.')
parser.add_argument('--n_val', type=int, default=-1,
                    help='Total # validation stimuli. Default: equal to n_train.')
parser.add_argument('--n_test', type=int, default=-1,
                    help='Total # test stimuli. Default: equal to n_train.')
parser.add_argument('--n_train_ood', nargs='+', required=False, default=[],
                    help='Size of OOD training sets.')
parser.add_argument('--n_val_ood', nargs='+', required=False, default=[],
                    help='Size of OOD validation sets. Default: equal to n_train_ood.')
parser.add_argument('--n_test_ood', nargs='+', required=False, default=[],
                    help='Size of OOD test sets. Default: equal to n_train_ood.')

# Paremeters for logging, storing models, etc.
parser.add_argument('--save_model_freq', help='Number of times to save model checkpoints \
                    throughout training. Saves are equally spaced from 0 to num_epoch.', type=int,
                    default=3)
parser.add_argument('--log_preds_freq', help='Number of times to log model predictions \
                    on test sets throughout training. Saves are equally spaced from 0 to num_epochs.',
                    type=int, default=3)

args = parser.parse_args()

# Parse command line arguments
wandb_proj = args.wandb_proj

model_type = args.model_type
patch_size = args.patch_size
cnn_size = args.cnn_size
feature_extract = args.feature_extract
pretrained = args.pretrained

train_dataset_names = args.train_datasets
val_datasets_names = args.val_datasets
optim = args.optim
lr = args.lr
lr_scheduler = args.lr_scheduler
num_epochs = args.num_epochs
batch_size = args.batch_size

k = args.k
rotation = args.rotation
scaling = args.scaling
unaligned = args.unaligned
multiplier = args.multiplier

n_train = args.n_train
n_val = args.n_val
n_test = args.n_test
n_train_ood = args.n_train_ood
n_val_ood = args.n_val_ood
n_test_ood = args.n_test_ood

# Default behavior for n_val, n_test
if val_datasets_names == 'all':
    val_datasets_names = [name for name in os.listdir('stimuli/source') if os.path.isdir(f'stimuli/source/{name}')]
    
    for td in train_dataset_names:
        val_datasets_names.remove(td)
    
if n_val == -1:
    n_val = n_train
if n_test == -1:
    n_test = n_train
if len(n_train_ood) == 0:
    n_train_ood = [n_train for _ in range(len(val_datasets_names))]
if len(n_val_ood) == 0:
    n_val_ood = n_train_ood
if len(n_test_ood) == 0:
    n_test_ood = n_train_ood

# Ensure that stimuli are 64x64
if model_type == 'vit' or model_type == 'clip_vit':
    if patch_size == 16:
        multiplier = multiplier*2

# Other hyperparameters/variables
im_size = 224
decay_rate = 0.95  # scheduler decay rate for Exponential type
int_to_label = {0: 'different', 1: 'same'}
label_to_int = {'different': 0, 'same': 1}

# Check arguments
assert not (model_type == 'clip_rn' and cnn_size != 50)  # Only CLIP ResNet-50 is defined
assert len(n_train_ood) == len(val_datasets_names)
assert len(n_test_ood) == len(val_datasets_names)
assert len(n_val_ood) == len(val_datasets_names)
assert im_size % patch_size == 0
assert k == 2 or k == 4 or k == 8
assert model_type == 'resnet' or model_type == 'vit' or model_type == 'clip_rn' \
    or model_type == 'clip_vit' or model_type == 'small_cnn'

# Create necessary directories 
try:
    os.mkdir('logs')
except FileExistsError:
    pass

# Create strings for paths and directories 
if unaligned:
    pos_string = 'unaligned'
else:
    pos_string = 'aligned'
    
if pretrained:
    pretrained_string = '_pretrained'
else:
    pretrained_string = ''
    
if feature_extract:
    fe_string = '_fe'
else:
    fe_string = ''
    
aug_string = ''
if rotation:
    aug_string += 'R'
if scaling:
    aug_string += 'S'
if len(aug_string) == 0:
    aug_string = 'N'

# Load models
if model_type == 'resnet':
    model_string = 'resnet_{0}'.format(cnn_size)
    
    if cnn_size == 18:
        model = models.resnet18(pretrained=pretrained)
    elif cnn_size == 50:
        model = models.resnet50(pretrained=pretrained)
    elif cnn_size == 152:
        model = models.resnet152(pretrained=pretrained)

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
    model_string = 'vit_b{0}'.format(patch_size)
    
    model_path = f'google/vit-base-patch{patch_size}-{im_size}-in21k'

    if pretrained:
        model = ViTForImageClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label=int_to_label,
            label2id=label_to_int
        )
    else:
        configuration = ViTConfig(patch_size=patch_size, image_size=im_size)
        model = ViTForImageClassification(configuration)
        
    transform = ViTImageProcessor(do_resize=False).from_pretrained(model_path)

    if feature_extract:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
            
elif 'clip' in model_type:                
    if 'vit' in model_type:
        model_string = model_string = 'clip_vit_b{0}'.format(patch_size)
        
        if pretrained:
            model, transform = clip.load(f'ViT-B/{patch_size}', device=device)
        else:
            sys.exit(1)
    else:
        model_string = 'clip_resnet50'
        
        if pretrained:
            model, transform = clip.load('RN50', device=device)
        else:
            sys.exit(1)
    
    if feature_extract:
        for name, param in model.visual.named_parameters():
            param.requires_grad = False
    
    # Add classification head to vision encoder
    in_features = model.visual.proj.shape[1]
    fc = nn.Linear(in_features, 2).to(device)
    model = nn.Sequential(model.visual, fc).float()

elif model_type == 'small_cnn':
    def stripalpha(x):
        return x[:3, :, :]
    transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Resize(64), 
                    transforms.ToTensor(),
                    stripalpha
                ]
            )
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size=3, stride=2, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size=3, stride=2, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size=3, stride=2, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Flatten(),
        nn.Linear(576, 2) # leave as logits
    ).float()

model = model.to(device)  # Move model to GPU if possible

# Create paths
model_string += pretrained_string  # Indicate if model is pretrained
model_string += fe_string  # Add 'fe' if applicable
model_string += '_{0}'.format(optim)  # Optimizer string

if len(train_dataset_names) == 1:
    train_dataset_string = train_dataset_names[0]
else:
    train_dataset_string = ''
    
    for td in train_dataset_names:
        if 'GRAYSCALE' in td:
            train_dataset_string += f'GRAY_{td.split("_")[1][:3]}-'
        elif 'MASK' in td:
            train_dataset_string += f'MASK_{td.split("_")[1][:3]}-'
        else:
            train_dataset_string += f'{td[:3]}-'
    train_dataset_string = train_dataset_string[:-1]

path_elements = [model_string, train_dataset_string, pos_string, aug_string, f'trainsize_{n_train}']

for root in ['logs']:
    stub = root

    for p in path_elements:
        try:
            os.mkdir('{0}/{1}'.format(stub, p))
        except FileExistsError:
            pass
        stub = '{0}/{1}'.format(stub, p)

log_dir = 'logs/{0}/{1}/{2}/{3}/{4}'.format(model_string, train_dataset_string, pos_string, aug_string, f'trainsize_{n_train}')

# Construct train set + DataLoader
train_dir = 'stimuli/{0}/{1}/{2}/{3}'.format(train_dataset_string, pos_string, aug_string, f'trainsize_{n_train}')

if not os.path.exists(train_dir):
    call_create_stimuli(patch_size, n_train, n_val, n_test, k, unaligned, multiplier, 
                        train_dir, rotation, scaling)
    
train_dataset = SameDifferentDataset(train_dir + '/train', transform=transform, rotation=rotation, scaling=scaling)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = SameDifferentDataset(train_dir + '/val', transform=transform, rotation=rotation, scaling=scaling)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
# Construct other validation sets
val_datasets = [val_dataset]
val_dataloaders = [val_dataloader]
val_labels = [train_dataset_string]

for v in range(len(val_datasets_names)):
    val_dir = 'stimuli/{0}/{1}/{2}/{3}'.format(val_datasets_names[v], pos_string, aug_string, f'trainsize_{n_train}')
    
    if not os.path.exists(val_dir):
        call_create_stimuli(patch_size, n_train_ood[v], n_val_ood[v], n_test_ood[v], k, unaligned, multiplier, 
                            val_dir, rotation, scaling)
    
    val_dataset = SameDifferentDataset(val_dir + '/val', transform=transform, rotation=rotation, scaling=scaling)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    val_datasets.append(val_dataset)
    val_dataloaders.append(val_dataloader)
    val_labels.append(val_datasets_names[v])
    
# Optimizer and scheduler
if optim == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
elif optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

if lr_scheduler == 'reduce_on_plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=num_epochs//5)
elif lr_scheduler == 'exponential':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

# Information to store
exp_config = {
    'model_type': model_type,
    'patch_size': patch_size,
    'cnn_size': cnn_size,
    'feature_extract': feature_extract,
    'pretrained': pretrained,
    'train_device': device,
    'train_dataset': train_dataset_string,
    'pos_condition': pos_string,
    'aug': aug_string,
    'train_size': n_train,
    'learning_rate': lr,
    'scheduler': lr_scheduler,
    'decay_rate': decay_rate,
    'patience': num_epochs//5,
    'optimizer': optim,
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'stimulus_size': '{0}x{0}'.format(patch_size * multiplier),
}

# Initialize Weights & Biases project & table
wandb.init(project=wandb_proj, config=exp_config)

run_id = wandb.run.id
wandb.run.name = '{0}_{1}{2}_{3}_LR{4}_{5}'.format(model_string, train_dataset_string, n_train, aug_string, lr, run_id)

# Log model predictions
pred_columns = ['Training Epoch', 'File Name', 'Image', 'Dataset', 'Prediction',
                'Truth', 'Same Score', 'Different Score', 'Same Accuracy', 
                'Different Accuracy']
test_table = wandb.Table(columns=pred_columns)

# Run training loop + evaluations
model = train_model(args, model, device, train_dataloader, len(train_dataset), 
                    optimizer, scheduler, log_dir, val_datasets, val_dataloaders,
                    test_table, val_labels)
