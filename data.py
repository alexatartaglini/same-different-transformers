from PIL import Image
import numpy as np
import os
import random
import argparse
import glob
from torch.utils.data import Dataset
import itertools
from math import factorial, floor


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

    return different_pairs, same_pairs


'''
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
'''
def create_stimuli(k, n, objects, unaligned, patch_size, multiplier, im_size, stim_dir,
                   patch_dir, condition):

    obj_size = patch_size * multiplier

    if unaligned:  # Place randomly
        coords = np.linspace(0, im_size - obj_size,
                             num=(im_size - obj_size), dtype=int)
    else:  # Place in ViT patch grid
        coords = np.linspace(0, im_size, num=(im_size // patch_size), endpoint=False,
                             dtype=int)
        new_coords = []
        for i in range(0, len(coords) - multiplier + 1, multiplier):
            new_coords.append(coords[i])

        coords = new_coords
        possible_coords = list(itertools.product(coords, repeat=k))

    n_per_class = n // 2

    if n_per_class <= len(objects):
        obj_sample = random.sample(objects, k=n_per_class)  # Objects to use

        all_different_pairs = list(itertools.combinations(obj_sample, k))
        different_sample = random.sample(all_different_pairs, k=n_per_class)

        same_pairs = {tuple([o] * k): [] for o in obj_sample}
        different_pairs = {o: [] for o in different_sample}

        # Assign positions for each object pair: one position each
        for pair in same_pairs.keys():
            if not unaligned:
                c = random.sample(possible_coords, k=k)
                same_pairs[pair].append(c)
            else:  # Code needs to be altered for k > 2
                c1 = random.sample(list(coords), k=2)
                c2 = random.sample(list(coords), k=2)

                # Ensure there is no overlap
                while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                        and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                    c2 = random.sample(list(coords), k=2)

                same_pairs[pair].append([c1, c2])

        for pair in different_pairs.keys():
            if not unaligned:
                c = random.sample(possible_coords, k=k)
                different_pairs[pair].append(c)
            else:  # Code needs to be altered for k > 2
                c1 = tuple(random.sample(list(coords), k=2))
                c2 = tuple(random.sample(list(coords), k=2))

                # Ensure there is no overlap
                while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                        and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                    c2 = tuple(random.sample(list(coords), k=2))

                different_pairs[pair].append([c1, c2])
    else:
        all_different_pairs = list(itertools.combinations(objects, k))
        different_sample = random.sample(all_different_pairs, k=len(objects))

        same_pairs = {tuple([o] * k): [] for o in objects}
        different_pairs = {o: [] for o in different_sample}

        n_same = len(objects)

        # Assign at least one position to each same pair
        for pair in same_pairs.keys():
            if not unaligned:
                c = random.sample(possible_coords, k=k)
                same_pairs[pair].append(c)
            else:  # Code needs to be altered for k > 2
                c1 = tuple(random.sample(list(coords), k=2))
                c2 = tuple(random.sample(list(coords), k=2))

                # Ensure there is no overlap
                while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                        and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                    c2 = tuple(random.sample(list(coords), k=2))

                same_pairs[pair].append([c1, c2])

        # Generate unique positions for pairs until desired number is achieved
        same_keys = list(same_pairs.keys())
        different_keys = list(different_pairs.keys())

        same_counts = [1] * n_same

        while n_same < n_per_class:
            key = random.choice(same_keys)

            if not unaligned:
                while len(same_pairs[key]) == len(possible_coords):
                    key = random.choice(same_keys)

            idx = same_keys.index(key)

            existing_positions = [set(c) for c in same_pairs[key]]

            if not unaligned:
                c = random.sample(possible_coords, k=k)

                while set(c) in existing_positions:  # Ensure unique position
                    c = random.sample(possible_coords, k=k)

                same_pairs[key].append(c)
            else:  # Code needs to be altered for k > 2
                c1 = tuple(random.sample(list(coords), k=2))
                c2 = tuple(random.sample(list(coords), k=2))

                # Ensure there is no overlap
                while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                        and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                    c2 = tuple(random.sample(list(coords), k=2))

                while set([c1, c2]) in existing_positions:  # Ensure unique position
                    c1 = tuple(random.sample(list(coords), k=2))
                    c2 = tuple(random.sample(list(coords), k=2))

                    # Ensure there is no overlap
                    while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                            and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                        c2 = tuple(random.sample(list(coords), k=2))

                same_pairs[key].append([c1, c2])

            n_same += 1
            same_counts[idx] += 1

        assert sum(same_counts) == n_per_class

        for i in range(len(different_keys)):
            key = different_keys[i]
            count = same_counts[i]

            for j in range(count):
                existing_positions = [set(c) for c in different_pairs[key]]

                if not unaligned:
                    c = random.sample(possible_coords, k=k)

                    while set(c) in existing_positions:  # Ensure unique position
                        c = random.sample(possible_coords, k=k)

                    different_pairs[key].append(c)
                else:  # Code needs to be altered for k > 2
                    c1 = tuple(random.sample(list(coords), k=2))
                    c2 = tuple(random.sample(list(coords), k=2))

                    # Ensure there is no overlap
                    while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                            and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                        c2 = tuple(random.sample(list(coords), k=2))

                    while set([c1, c2]) in existing_positions:  # Ensure unique position
                        c1 = tuple(random.sample(list(coords), k=2))
                        c2 = tuple(random.sample(list(coords), k=2))

                        # Ensure there is no overlap
                        while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                                and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                            c2 = tuple(random.sample(list(coords), k=2))

                    different_pairs[key].append([c1, c2])

    # Create the stimuli generated above
    object_ims = {}

    for o in objects:
        im = Image.open('{0}/{1}'.format(stim_dir, o))
        im = im.resize((obj_size, obj_size))
        object_ims[o] = im

    for sd_class, dict in zip(['same', 'different'], [same_pairs, different_pairs]):

        setting = '{0}/{1}/{2}'.format(patch_dir, condition, sd_class)

        for key in dict.keys():
            positions = dict[key]

            for i in range(len(positions)):
                p = positions[i]
                base = Image.new('RGB', (im_size, im_size), (255, 255, 255))

                # Needs to be altered for k > 2
                obj1 = key[0]
                obj2 = key[1]
                objs = [obj1, obj2]

                for c in range(len(p)):
                    base.paste(object_ims[objs[c]], box=p[c])

                base.save('{0}/{1}_{2}_{3}.png'.format(setting, obj1[:-4], obj2[:-4], i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data.')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of patch (eg. 16 or 32).')
    parser.add_argument('--n_train', type=int, default=6400,
                        help='Total # of training stimuli. eg. if n_train=6400, a dataset'
                             'will be generated with 3200 same and 3200 different stimuli.'
                             'Brady lab: 6400, Developmental: 1024.')
    parser.add_argument('--n_val', type=int, default=640,
                        help='Total # validation stimuli. Brady lab: 640, Developmental: 256.')
    parser.add_argument('--n_test', type=int, default=640,
                        help='Total # test stimuli. Brady lab: 640, Developmental: 256.')
    parser.add_argument('--k', type=int, default=2, help='Number of objects per scene.')
    parser.add_argument('--unaligned', action='store_true', default=False,
                        help='Misalign the objects from ViT patches (ie. place randomly).')
    parser.add_argument('--multiplier', type=int, default=1, help='Factor by which to scale up '
                                                                  'stimulus size.')
    parser.add_argument('--stim_dir', type=str, help='Stimulus directory.', default='OBJECTSALL')

    args = parser.parse_args()

    # Command line arguments
    patch_size = args.patch_size  # Object size: patch_size x patch_size
    n_train = args.n_train  # Size of training set
    n_val = args.n_val  # Size of validation set
    n_test = args.n_test  # Size of test set
    k = args.k  # Number of objects per image
    unaligned = args.unaligned  # False = objects align with ViT patches
    multiplier = args.multiplier
    stim_dir = args.stim_dir

    '''
    # Brady Lab:
    n_train = 1920  # Max number of unique objects to present during training
    n_val = 240  # Max number of unique objects to present during validation
    n_test = 240  # Max number of unique objects to present during testing
    n = 3200  # Number of different stimuli to create. Total # stimuli = n*2.
    n_2 = 320

    # Developmental:
    n_train = 204
    n_val = 26
    n_test = 26
    n = 512
    n_2 = 128
    '''

    # Other parameters
    im_size = 224  # Size of base image

    assert im_size % patch_size == 0

    # Make directories
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

    if unaligned:
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

    # Collect object image paths
    object_files = [f for f in os.listdir(stim_dir) if os.path.isfile(os.path.join(stim_dir, f))
                    and f != '.DS_Store']

    # Compute number of unique objects that should be allocated to train/val/test sets
    percent_train = n_train / (n_train + n_val + n_test)
    percent_val = n_val / (n_train + n_val + n_test)
    percent_test = n_test / (n_train + n_val + n_test)

    n_unique = len(object_files)
    n_unique_train = floor(n_unique * percent_train)
    n_unique_val = floor(n_unique * percent_val)
    n_unique_test = floor(n_unique * percent_test)

    '''
    object_files_train = random.sample(object_files, k=n_train)
    object_files_val = [x for x in object_files if x not in object_files_train]
    object_files_test = random.sample(object_files_val, k=n_test)
    '''

    # Allocate unique objects
    ofs = object_files  # Copy of object_files to sample from

    object_files_train = random.sample(ofs, k=n_unique_train)
    ofs = [o for o in ofs if o not in object_files_train]

    object_files_val = random.sample(ofs, k=n_unique_val)
    ofs = [o for o in ofs if o not in object_files_val]

    object_files_test = random.sample(ofs, k=n_unique_test)

    '''
    for x in object_files_test:
        object_files_val.remove(x)

    object_files_val = random.sample(object_files_val, k=n_val)


    assert len(object_files_train) == n_train
    assert len(object_files_test) == n_test
    assert len(object_files_val) == n_val
    '''

    assert len(object_files_train) == n_unique_train
    assert len(object_files_val) == n_unique_val
    assert len(object_files_test) == n_unique_test
    assert set(object_files_train).isdisjoint(object_files_test) \
           and set(object_files_train).isdisjoint(object_files_val) \
           and set(object_files_test).isdisjoint(object_files_val)

    create_stimuli(k, n_train, object_files_train, unaligned, patch_size, multiplier,
                   im_size, stim_dir, patch_dir, 'train')
    create_stimuli(k, n_val, object_files_val, unaligned, patch_size, multiplier,
                   im_size, stim_dir, patch_dir, 'val')
    create_stimuli(k, n_test, object_files_test, unaligned, patch_size, multiplier,
                   im_size, stim_dir, patch_dir, 'test')

    '''
    object_ims = {}

    for o in object_files:
        im = Image.open('{0}/{1}'.format(stim_dir, o))
        im = im.resize((patch_size * multiplier, patch_size * multiplier))
        object_ims[o] = im

    '''
    '''
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