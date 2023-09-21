import os
import clip
import time
import torch
import pickle
import numpy as np
import argparse
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from data import SameDifferentDataset, call_create_stimuli
from sklearn.metrics import accuracy_score

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
int_to_label = {0: 'different', 1: 'same'}
label_to_int = {'different': 0, 'same': 1}
os.makedirs('wideness', exist_ok=True)

# function to test "wideness" for all stimuli
all_datasets = ['OBJECTSALL','GRAYSCALE_OBJECTSALL', 'MASK_OBJECTSALL', 'DEVELOPMENTAL',
                'GRAYSCALE_DEVELOPMENTAL', 'MASK_DEVELOPMENTAL', 'ALPHANUMERIC',
                'SQUIGGLES']

def cossim(main, compare):
    return np.dot(main, compare.T) / (np.outer(np.sqrt(np.sum(np.square(main.numpy()), axis=1)),
                                               np.sqrt(np.sum(np.square(compare.numpy()), axis=1)).T))

def test_acc(val_dataloader, val_dataset, model, model_type):
    model.eval()
    with torch.no_grad():
        running_acc_val = 0.0

        for bi, (d, f) in enumerate(val_dataloader):
            if model_type == 'vit16img':
                inputs = d['pixel_values'].squeeze(1).to(device)
            else:
                inputs = d['image'].to(device)
            
            labels = d['label'].to(device)

            outputs = model(inputs)
            if model_type == 'vit16img':
                outputs = outputs.logits

            preds = outputs.argmax(1)
            acc = accuracy_score(labels.to('cpu'), preds.to('cpu'))

            running_acc_val += acc * inputs.size(0)

        epoch_acc_val = running_acc_val / len(val_dataset)
    return epoch_acc_val

def save_similarities(dataset, model_type, batch_size, num_batches, where='last'):
    assert where in ['last', 'first']
    similarities = np.ones((6400, 6400)) * -np.inf
    for i in range(num_batches):
        main = torch.load(f'wideness/{model_type}/{dataset}/{where}/{i}.pt')
        for j in range(i, num_batches):
            compare = torch.load(f'wideness/{model_type}/{dataset}/{where}/{j}.pt')

            # this matrix now has [i...i+batch_size-1] similarities with [j...j+batch_size-1]
            res = cossim(main, compare)
            similarities[i*batch_size:i*batch_size+batch_size, j*batch_size:j*batch_size+batch_size] = res
    np.save(f'wideness/{model_type}/{dataset}/{where}/similarities.npy', similarities)

def all_wideness(model, model_type, transform, checkpoint, dataset_names=all_datasets, batch_size=64):
    num_batches = int(6400/batch_size)
    if checkpoint is not None:
        model_type += '_' + ''.join(checkpoint.split('.')[:-1])

    # make a bunch of random noise as a baseline
    # TODO actually save the 'first layer' representations
    os.makedirs(f'wideness/{model_type}/rand/last', exist_ok=True)
    os.makedirs(f'wideness/{model_type}/rand/first', exist_ok=True)
    with torch.set_grad_enabled(False):
        for i in range(num_batches):
            rand = torch.normal(0, 1, size=[batch_size, 3, 224, 224]).to(device)
            last = model(rand)
            last = last.logits if model_type=='vit16img' else last
            torch.save(last, f'wideness/{model_type}/rand/last/{i}.pt')
    save_similarities('rand', model_type, batch_size, num_batches, where='last')
    # save_similarities('rand', model_type, batch_size, num_batches, where='first')

    for s in dataset_names:
        # load in or generate dataset
        dir = 'stimuli/{0}/{1}/{2}/{3}'.format(s, 'unaligned', 'N', 'trainsize_1200_6400-300-100')

        if not os.path.exists(dir):
            print(f"generating {dir}")
            call_create_stimuli(16, 6400, 6400, 6400, 2, True, 2, dir, False, False,
                                n_train_tokens=1200, n_val_tokens=300, n_test_tokens=100)

        dataset = SameDifferentDataset(dir + '/train', transform=transform, rotation=False, scaling=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        test_dataset = SameDifferentDataset(dir + '/test', transform=transform, rotation=False, scaling=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(model_type, s, test_acc(test_dataloader, test_dataset, model, model_type))

        # convert all the images to representations and cache them
        os.makedirs(f'wideness/{model_type}/{s}/last', exist_ok=True)
        os.makedirs(f'wideness/{model_type}/{s}/first', exist_ok=True)
        fnames = []
        with torch.set_grad_enabled(False):
            for bi, (d, f) in enumerate(dataloader):
                fnames += f
                if model_type == 'vit16img':
                    inputs = d['pixel_values'].squeeze(1).to(device)
                else:
                    inputs = d['image'].to(device)
                outputs = model(inputs)
                outputs = outputs.logits if model_type=='vit16img' else outputs
                torch.save(outputs, f'wideness/{model_type}/{s}/last/{bi}.pt')
        with open(f'wideness/{model_type}/{s}/fnames.pkl', 'wb') as f:
            pickle.dump(fnames, f)

        # for each batch load in the other batches and calculate cross-similarity to fill out nxn matrix
        save_similarities(s, model_type, batch_size, num_batches, where='last')
        # save_similarities(s, model_type, batch_size, num_batches, where='first')
        print(f'done {s}')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--checkpoint', type=str, default=None, help='path to a checkpoint to use for a model thats been fine-tuned.')
parser.add_argument('--out', action='store_true', help='output csv of averages for similarities matrices that have already been computed.')
args = parser.parse_args()

base_models = ['rn50img', 'vit16img', 'rn50clip', 'vit16clip']
if args.model:
    assert(args.model in base_models)
if args.out:
    sim_avgs = pd.DataFrame(columns=['Model', 'checkpoint?', 'Measured Dataset', 'avg', 'sameavg', 'diffavg', 'samediffavg'])
    for m in os.listdir('wideness'):
        if m == 'sim_avgs.csv': continue
        for d in all_datasets + ['rand']:
            checkpoint = m not in base_models

            similarities = np.load(f'wideness/{m}/{d}/last/similarities.npy')
            similarities = np.triu(similarities)
            similarities[np.tril_indices(similarities.shape[0], -1)] = np.nan

            avg = np.nanmean(similarities)
            sameavg = np.nanmean(similarities[0:3200, 0:3200])
            diffavg = np.nanmean(similarities[3200:6400, 3200:6400])
            samediffavg = np.nanmean(similarities[0:3200, 3200:6400])
            sim_avgs.loc[len(sim_avgs.index)] = (m, checkpoint, d, avg, sameavg, diffavg, samediffavg)
    sim_avgs.to_csv('wideness/sim_avgs.csv')


tick = time.time()
if args.model=='rn50img':
    model = resnet50(pretrained=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if args.checkpoint is not None: #UNTESTED
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    model.to(device)

elif args.model=='vit16img':
    vit16img_path = 'google/vit-base-patch16-224-in21k'
    model = ViTForImageClassification.from_pretrained(
        vit16img_path,
        num_labels=2,
        id2label=int_to_label,
        label2id=label_to_int
    )
    transform = ViTImageProcessor(do_resize=False).from_pretrained(vit16img_path)
    if args.checkpoint is not None: #UNTESTED
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

elif args.model=='rn50clip':
    rn50clip, transform = clip.load('RN50', device=device)
    rn50clip.to(device)        
    if args.checkpoint is not None:
        # load in the state dict with the head on
        in_features = rn50clip.visual.output_dim
        fc = nn.Linear(in_features, 2).to(device)
        rn50clip = nn.Sequential(rn50clip.visual, fc).float()
        rn50clip.load_state_dict(torch.load(args.checkpoint, map_location=device))
        
        # take the head back off 
        rn50clip = nn.Sequential(*list(rn50clip.children())[:-1])

    model = rn50clip

elif args.model=='vit16clip':
    vit16clip, transform = clip.load(f'ViT-B/16', device=device)
    vit16clip.to(device)
    if args.checkpoint is not None:
        # load in the state dict and then take the head back off 
        in_features = vit16clip.visual.proj.shape[1]
        fc = nn.Linear(in_features, 2).to(device)
        vit16clip = nn.Sequential(vit16clip.visual, fc).float()
        vit16clip.load_state_dict(torch.load(args.checkpoint, map_location=device))

        # take the head back off 
        vit16clip = nn.Sequential(*list(vit16clip.children())[:-1])

    model = vit16clip


model.eval()  
all_wideness(model, args.model, transform, args.checkpoint)

tock = time.time()
print(f'seconds for {args.model}:', tock - tick, 'min:', (tock - tick)/60)