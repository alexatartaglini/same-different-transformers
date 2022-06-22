from PIL import Image
import numpy as np
import os
import random
import argparse
import glob
from torch.utils.data import Dataset
import itertools
from math import factorial


def nCr(n, r):
    f = factorial
    return f(n) / f(r) / f(n-r)


def load_dataset(root_dir):
    int_to_label = {1: 'same', 0: 'different'}
    ims = {}
    idx = 0

    for l in int_to_label.keys():
        im_paths = glob.glob('{0}/{1}/*.png'.format(root_dir, int_to_label[l]))

        for im in im_paths:
            pixels = Image.open(im)
            im_dict = {'image': pixels, 'image_path': im, 'label': l}
            ims[idx] = im_dict
            idx += 1
            pixels.close()

    return ims


class SameDifferentDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.im_dict = load_dataset(root_dir)
        self.transform = transform

    def __len__(self):
        return len(list(self.im_dict.keys()))

    def __getitem__(self, idx):
        im_path = self.im_dict[idx]['image_path']
        im = Image.open(im_path)
        label = self.im_dict[idx]['label']

        if self.transform:
            if str(type(self.transform)) == "<class 'torchvision.transforms.transforms.Compose'>":
                item = self.transform(im)
                item = {'image': item, 'label': label}
            else:
                item = self.transform(im, return_tensors='pt')
                item['label'] = label

        return item, im_path


def initialize_pairs(k, n, object_files, obj_size, im_size):
    '''
    Creates len(object_files) unique k-length combinations selected from object_files without
    the computational cost incurred by itertools, which generates all possible
    combinations and becomes intractable when k = 8 and n > 100. Then, all permutations
    where the same object is repeated k times are appended (same condition).

    :param k: length of combination.
    :param n: number of different pairs to produce (total # pairs = n * 2). If n == 'MAX', n
              is set to the total number of possible different object pairs.
    :param object_files: population to sample from.

    :returns: a length n list of different object pairs, and a length n list of same object pairs.
              Note that if n > len(object_files), the same list will contain repeats, and if
              n > len(all possible different-object combinations), then the different list will
              contain repeats.
    '''

    all_combos = list(itertools.combinations(object_files, k))
    lim = (im_size // obj_size)**2
    lim = nCr(lim, k)

    obj_counts = {o: 0 for o in object_files}

    if n == 'MAX':
        n = len(all_combos)

    if n <= len(all_combos):
        different_pairs = random.sample(all_combos, n)
    else:
        different_pairs = [random.sample(object_files, k) for _ in range(n)]

    same_pairs = []
    for _ in range(n):
        obj = random.choice(object_files)

        while obj_counts[obj] == lim:
            obj = random.choice(object_files)

        same_pairs.append([obj for _ in range(k)])
        obj_counts[obj] += 1

    '''
    mat = [[None for _ in range(k)] for _ in range(n * 2)]  # kx(n*2) matrix

    # Different pairs:
    for i in range(n):
        while True:
            sample = random.sample(object_files, k)
            for prev in range(i - 1):
                if set(sample) == set(mat[prev]):
                    break
            break

        mat[i] = sample

    # Same pairs:
    for i in range(n, n * 2):
        obj = random.choice(object_files)
        mat[i] = [obj for _ in range(k)]
        
    '''

    return different_pairs, same_pairs


def create_stimuli(pairs, coords, setting, fixed, unaligned, im_size, patch_size, multiplier):
    p = 0
    # This is used to ensure that if the same pair of objects is being used in multiple stimuli,
    # the objects will be placed in a different configuration than before.
    placement_dict = {''.join(sorted(list(pair))): [] for pair in pairs}

    obj_size = patch_size * multiplier

    for pair in pairs:
        key = ''.join(sorted(list(pair)))
        base = Image.new('RGB', (im_size, im_size), (255, 255, 255))

        while True:  # Choose unique position for a given pair
            object_coords = []

            for i in range(k):
                if fixed:  # Place objects in fixed locations
                    c = coords[i]
                elif unaligned:  # Randomly position objects anywhere
                    c = random.sample(list(coords), k=2)

                    if len(object_coords) == 1:
                        o = object_coords[0]

                        # need to alter the following code for k > 2
                        while (c[0] >= (o[0] - obj_size) and c[0] <= (o[0] + obj_size)) \
                                and (c[1] >= (o[1] - obj_size) and c[1] <= (o[1] + obj_size)):
                            c = random.sample(list(coords), k=2)

                else:  # Randomly position objects in aligned positions
                    c = random.sample(list(coords), k=2)

                    # Do not repeat object positions
                    while c in object_coords:
                        c = random.sample(list(coords), k=2)

                object_coords.append(c)

            if fixed:
                break
            else:
                obj_set = set(tuple(c) for c in object_coords)
                if not obj_set in placement_dict[key]:
                    break

        for c in range(len(object_coords)):
            base.paste(object_ims[pair[c]], box=object_coords[c])

        base.save('{0}/{1}_{2}.png'.format(setting, p, k))
        p += 1
        if not fixed:
            placement_dict[key].append(obj_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data.')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of patch (eg. 16 or 32).')
    parser.add_argument('--k', type=int, default=2, help='Number of objects per scene.')
    parser.add_argument('--fixed', action='store_true', default=False,
                        help='Fix the objects, which are randomly placed by default.')
    parser.add_argument('--unaligned', action='store_true', default=False,
                        help='Misalign the objects from ViT patches.')
    parser.add_argument('--multiplier', type=int, default=1, help='Factor by which to scale up '
                                                                  'stimulus size.')
    parser.add_argument('--stim_dir', type=str, help='Stimulus directory.', default='OBJECTSALL')

    args = parser.parse_args()

    fixed = args.fixed  # False = objects are randomly placed
    unaligned = args.unaligned  # False = objects align with ViT patches

    # Brady Lab:
    n_train = 1920  # Max number of unique objects to present during training
    n_val = 240  # Max number of unique objects to present during validation
    n_test = 240  # Max number of unique objects to present during testing
    n = 3200  # Number of different stimuli to create. Total # stimuli = n*2.
    n_2 = 320

    '''
    # Developmental:
    n_train = 204
    n_val = 26
    n_test = 26
    n = 512
    n_2 = 128
    '''

    k = args.k  # Number of objects per image
    im_size = 224  # Size of base image
    patch_size = args.patch_size  # Object size
    multiplier = args.multiplier
    stim_dir = args.stim_dir

    assert not (fixed and unaligned)
    assert im_size % patch_size == 0

    try:
        os.mkdir('stimuli')
    except FileExistsError:
        pass

    if stim_dir == 'OBJECTSALL':
        stim_subdir = ''
    else:
        stim_subdir = stim_dir + '/'

        try:
            os.mkdir('stimuli/{0}'.format(stim_dir))
        except FileExistsError:
            pass

    if fixed:
        pos_dir = 'stimuli/{0}fixed'.format(stim_subdir)
    elif unaligned:
        pos_dir = 'stimuli/{0}unaligned'.format(stim_subdir)
    else:
        pos_dir = 'stimuli/{0}aligned'.format(stim_subdir)

    try:
        os.mkdir(pos_dir)
    except FileExistsError:
        pass

    patch_dir = '{0}/{1}x{1}'.format(pos_dir, patch_size * multiplier)

    try:
        os.mkdir(patch_dir)
    except FileExistsError:
        pass

    try:
        os.mkdir('{0}/{1}'.format(patch_dir, k))
    except FileExistsError:
        pass

    patch_dir = patch_dir + '/' + str(k)

    object_files = [f for f in os.listdir(stim_dir) if os.path.isfile(os.path.join(stim_dir, f))
                    and f != '.DS_Store']

    object_files_train = random.sample(object_files, k=n_train)
    object_files_val = [x for x in object_files if x not in object_files_train]
    object_files_test = random.sample(object_files_val, k=n_test)

    for x in object_files_test:
        object_files_val.remove(x)

    object_files_val = random.sample(object_files_val, k=n_val)

    assert len(object_files_train) == n_train
    assert len(object_files_test) == n_test
    assert len(object_files_val) == n_val
    assert set(object_files_train).isdisjoint(object_files_test) \
           and set(object_files_train).isdisjoint(object_files_val) \
           and set(object_files_test).isdisjoint(object_files_val)

    object_ims = {}

    for o in object_files:
        im = Image.open('{0}/{1}'.format(stim_dir, o))
        im = im.resize((patch_size * multiplier, patch_size * multiplier))
        object_ims[o] = im

    train_different_pairs, train_same_pairs = initialize_pairs(k, n, object_files_train,
                                                               (patch_size * multiplier),
                                                               im_size)
    test_different_pairs, test_same_pairs = initialize_pairs(k, n_2, object_files_test,
                                                             (patch_size * multiplier),
                                                             im_size)
    val_different_pairs, val_same_pairs = initialize_pairs(k, n_2, object_files_val,
                                                           (patch_size * multiplier),
                                                           im_size)

    different_list = [train_different_pairs, test_different_pairs, val_different_pairs]
    same_list = [train_same_pairs, test_same_pairs, val_same_pairs]

    coords = np.linspace(0, im_size, num=(im_size // patch_size), endpoint=False, dtype=int)
    new_coords = []
    for i in range(0, len(coords) - multiplier + 1, multiplier):
        new_coords.append(coords[i])

    coords = new_coords

    if unaligned:  # Place randomly
        coords = np.linspace(0, im_size - (patch_size * multiplier),
                             num=(im_size - (patch_size * multiplier)), dtype=int)
    elif fixed:  # Place in fixed coordinates
        if patch_size == 16:
            if k == 2:
                coords = [[80, 96], [128, 96]]
            elif k == 4:
                coords = [[80, 80], [80, 128], [128, 80], [128, 128]]
            elif k == 8:
                coords = [[80, 80], [32, 80], [176, 80], [128, 80],
                          [80, 128], [32, 128], [176, 128], [128, 128]]
        elif patch_size == 32:
            if k == 2:
                coords = [[64, 96], [128, 96]]
            elif k == 4:
                coords = [[64, 64], [128, 64], [64, 128], [128, 128]]
            elif k == 8:
                coords = [[32, 32], [96, 32], [160, 32], [160, 96],
                          [160, 160], [96, 160], [32, 160], [32, 96]]

    for condition in ['train', 'test', 'val']:
        try:
            os.mkdir('{0}/{1}'.format(patch_dir, condition))
        except FileExistsError:
            pass

        try:
            os.mkdir('{0}/{1}/same'.format(patch_dir, condition))
        except FileExistsError:
            pass

        try:
            os.mkdir('{0}/{1}/different'.format(patch_dir, condition))
        except FileExistsError:
            pass

    for pairs, condition in zip(different_list, ['train', 'test', 'val']):
        create_stimuli(pairs, coords, '{0}/{1}/different'.format(patch_dir, condition),
                       fixed, unaligned, im_size, patch_size, multiplier)

    for pairs, condition in zip(same_list, ['train', 'test', 'val']):
        create_stimuli(pairs, coords, '{0}/{1}/same'.format(patch_dir, condition),
                       fixed, unaligned, im_size, patch_size, multiplier)

    '''
    p = 0
    
    for pair in train_pairs:
        object_coords = []
        base = Image.new('RGB', (im_size, im_size), (255, 255, 255))
        same = all(x == pair[0] for x in pair)
    
        if same:
            setting = '{0}/{1}/same'.format(patch_dir, 'train')
        else:
            setting = '{0}/{1}/different'.format(patch_dir, 'train')
    
        for i in range(k):
            if fixed:
                c = coords[i]
            elif unaligned:
                c = random.sample(list(coords), k=2)
    
                while True:
                    for o in object_coords:
                        if (c[0] >= o[0] and c[0] <= o[0] + patch_size) or (c[1] >= o[1] and c[1] <= o[1] + patch_size):
                            c = random.sample(list(coords), k=2)
                            break
                    break
            else:  # Randomly position objects
                c = random.sample(list(coords), k=2)
    
                # Do not repeat object positions
                while c in object_coords:
                    c = random.sample(list(coords), k=2)
    
            object_coords.append(c)
    
        for c in range(len(object_coords)):
            base.paste(object_ims[pair[c]], box=object_coords[c])
    
        base.save('{0}/{1}_{2}.png'.format(setting, p, k))
        p += 1
    
    for pair in test_pairs:
        object_coords = []
        base = Image.new('RGB', (im_size, im_size), (255, 255, 255))
        same = all(x == pair[0] for x in pair)
    
        if same:
            setting = '{0}/{1}/same'.format(patch_dir, 'test')
        else:
            if not repetitions:
                if len(pair) != len(set(pair)):  # Repeated objects
                    continue
    
            if c_test == n_test:
                continue
    
            setting = '{0}/{1}/different'.format(patch_dir, 'test')
            c_test += 1
    
        for i in range(k):
            if fixed:
                c = coords[i]
            elif unaligned:
                c = random.sample(list(coords), k=2)
    
                while True:
                    for o in object_coords:
                        if (c[0] >= o[0] and c[0] <= o[0] + patch_size) or (c[1] >= o[1] and c[1] <= o[1] + patch_size):
                            c = random.sample(list(coords), k=2)
                            break
                    break
            else:  # Randomly position objects
                c = random.sample(list(coords), k=2)
    
                # Do not repeat object positions
                while c in object_coords:
                    c = random.sample(list(coords), k=2)
    
            object_coords.append(c)
    
        for c in range(len(object_coords)):
            base.paste(object_ims[pair[c]], box=object_coords[c])
    
        base.save('{0}/{1}_{2}.png'.format(setting, p, k))
        p += 1
    
    for pair in val_pairs:
        object_coords = []
        base = Image.new('RGB', (im_size, im_size), (255, 255, 255))
        same = all(x == pair[0] for x in pair)
    
        if same:
            setting = '{0}/{1}/same'.format(patch_dir, 'val')
        else:
            if not repetitions:
                if len(pair) != len(set(pair)):  # Repeated objects
                    continue
    
            if c_val == n_val:
                continue
    
            setting = '{0}/{1}/different'.format(patch_dir, 'val')
            c_val += 1
    
        for i in range(k):
            if fixed:
                c = coords[i]
            elif unaligned:
                c = random.sample(list(coords), k=2)
    
                while True:
                    for o in object_coords:
                        if (c[0] >= o[0] and c[0] <= o[0] + patch_size) or (c[1] >= o[1] and c[1] <= o[1] + patch_size):
                            c = random.sample(list(coords), k=2)
                            break
                    break
            else:  # Randomly position objects
                c = random.sample(list(coords), k=2)
    
                # Do not repeat object positions
                while c in object_coords:
                    c = random.sample(list(coords), k=2)
    
            object_coords.append(c)
    
        for c in range(len(object_coords)):
            base.paste(object_ims[pair[c]], box=object_coords[c])
    
        base.save('{0}/{1}_{2}.png'.format(setting, p, k))
        p += 1
    '''