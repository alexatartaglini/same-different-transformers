from torchvision import models, transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader
from data import SameDifferentDataset
import torch.nn as nn
import torch
import argparse
import os
import json


def evaluate_model(model, device, model_type, log_dir, val_dataloaders, val_labels=None):
    if not val_labels:
        val_labels = list(range(len(val_dataloaders)))

    eval_dict = {e: {'correct': [], 'incorrect': []} for e in val_labels}

    # Perform evaluations
    model.eval()
    for i in range(len(val_dataloaders)):
        val_dataloader = val_dataloaders[i]
        val_label = val_labels[i]

        print()
        print('{0} validation step:'.format(val_label))

        with torch.no_grad():
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

                outputs = outputs.argmax(1)

                for i in range(len(outputs)):
                    if outputs[i] == labels[i]:
                        eval_dict[val_label]['correct'].append(f[i])
                    else:
                        eval_dict[val_label]['incorrect'].append(f[i])

            print()
            print('Validation: {0}'.format(val_label))
            print('# Correct: {0}'.format(len(eval_dict[val_label]['correct'])))
            print('# Incorrect: {0}'.format(len(eval_dict[val_label]['incorrect'])))

    with open('{0}/eval_ims.json'.format(log_dir), 'w') as f:
        json.dump(eval_dict, f)

    return eval_dict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_type', help='Model to train: resnet or vit.',
                    type=str, required=True)
parser.add_argument('--patch_size', type=int, default=32, help='Size of patch (eg. 16 or 32).')
parser.add_argument('--cnn_size', type=int, default=50,
                    help='Number of layers (eg. 50 = ResNet-50).', required=False)
parser.add_argument('--k', type=int, default=2, help='Number of objects per scene.')
parser.add_argument('--fixed', action='store_true', default=False,
                    help='Fix the objects, which are randomly placed by default.')
parser.add_argument('--unaligned', action='store_true', default=False,
                    help='Misalign the objects from ViT patches.')
parser.add_argument('--multiplier', type=int, default=1, help='Factor by which to scale up '
                                                              'stimulus size.')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=64, help='Train/validation batch size.')
parser.add_argument('--feature_extract', action='store_true', default=False,
                    help='Only train the final layer; freeze all other layers..')


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
assert not (fixed and unaligned)
assert im_size % patch_size == 0
assert k == 2 or k == 4 or k == 8
assert model_type == 'resnet' or model_type == 'vit'

# Create necessary directories
try:
    os.mkdir('logs')
except FileExistsError:
    pass

if fixed:
    pos_condition = 'fixed'
elif unaligned:
    pos_condition = 'unaligned'
else:
    pos_condition = 'aligned'

if model_type == 'resnet':
    model_str = 'resnet_{0}'.format(cnn_size)

    if cnn_size == 50:
        model = models.resnet50(pretrained=True)

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

model_path = 'logs/{0}/{1}/{2}/{3}/model_19.pth'.format(model_str, k, pos_condition,
                                                        patch_size * multiplier)
model.load_state_dict(torch.load(model_path, map_location=device))

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

# Create datasets/dataloaders
val_dataset = SameDifferentDataset(root_dir + '/val', transform=transform)
val_dataset_abstract = SameDifferentDataset(
    'stimuli/DEVELOPMENTAL/{0}/{1}x{1}/{2}/val'.format(pos_condition, patch_size * multiplier, k),
    transform=transform)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
val_dataloader_abstract = DataLoader(val_dataset_abstract, batch_size=batch_size, shuffle=True)

val_dataloaders = [val_dataloader, val_dataloader_abstract]
val_labels = ['in_distribution', 'shape_bias']

# Run evaluations
eval_dict = evaluate_model(model, device, model_type, log_dir, val_dataloaders,
                           val_labels=val_labels)
