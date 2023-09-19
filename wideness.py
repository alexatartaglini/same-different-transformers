import os
import clip
import time
import torch
import pickle
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from data import SameDifferentDataset, call_create_stimuli

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

def all_wideness(model, model_type, transform, dataset_names=all_datasets, batch_size=64):
    num_batches = int(6400/batch_size)

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
        val_dir = 'stimuli/{0}/{1}/{2}/{3}'.format(s, 'unaligned', 'N', 'trainsize_1200_6400-300-100')

        if not os.path.exists(val_dir):
            print(f"generating {val_dir}")
            call_create_stimuli(16, 6400, 6400, 6400, 2, True, 2, val_dir, False, False,
                                n_train_tokens=1200, n_val_tokens=300, n_test_tokens=100)

        dataset = SameDifferentDataset(val_dir + '/train', transform=transform, rotation=False, scaling=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # convert all the images to representations and cache them
        os.makedirs(f'wideness/{model_type}/{s}/last', exist_ok=True)
        os.makedirs(f'wideness/{model_type}/{s}/first', exist_ok=True)
        fnames = []
        with torch.set_grad_enabled(False):
            for bi, (d, f) in enumerate(dataloader):
                fnames += f
                if model_type == 'vit':
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
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

assert(args.model in ['rn50img', 'vit16img', 'rn50clip', 'vit16clip'])

tick = time.time()
if args.model=='rn50img':
    model = resnet50(pretrained=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
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
    model.to(device)
elif args.model=='rn50clip':
    rn50clip, transform = clip.load('RN50', device=device)
    rn50clip.to(device)
    model = rn50clip.encode_image
elif args.model=='vit16clip':
    vit16clip, transform = clip.load(f'ViT-B/16', device=device)
    vit16clip.to(device)
    model = vit16clip.encode_image

all_wideness(model, args.model, transform)

tock = time.time()
print(f'seconds for {args.model}:', tock - tick, 'min:', (tock - tick)/60)