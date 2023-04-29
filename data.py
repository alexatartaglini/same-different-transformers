from PIL import Image
import numpy as np
import os
import random
import argparse
import glob
from torch.utils.data import Dataset
import itertools
from math import floor


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
    def __init__(self, root_dir, transform=None, rotation=False, scaling=False):
        self.root_dir = root_dir
        self.im_dict = load_dataset(root_dir)
        self.transform = transform
        self.rotation = rotation
        self.scaling = scaling

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


def create_stimuli(k, n, objects, unaligned, patch_size, multiplier, im_size, stim_dir,
                   patch_dir, condition, rotation=False, scaling=False):
    '''
    Creates n same_different stimuli with (n // 2) stimuli assigned to each class. If
    n > the number of unique objects used to create the dataset, then randomly selected
    object pairs will be repeated in unique randomly selected positions until n unique
    stimuli are created. This code ensures that the number of repeated stimuli is the
    same between the 'same' and 'different' classes; for example, if a given object
    pair in the 'same' set is repeated in unique positions three times, another
    randomly selected object pair in the 'different' set is repeated in three (separate)
    unique positions.

    :param k: The number of objects per image (eg. 2).
    :param n: The total desired size of the stimulus set.
    :param objects: a list of filenames for each unique object to be used in creating
                    the set. NOTE: it's possible that not all objects in this list will
                    be used. The actual objects used are randomly selected.
    :param unaligned: True if stimuli should be randomly placed rather than aligned with
                      ViT patches.
    :param patch_size: Size of ViT patches.
    :param multiplier: Scalar by which to multiply object size. (object size = patch_size
                       * multiplier)
    :param im_size: Size of the base image.
    :param stim_dir: Directory of the objects to be used in creating the set.
    :param patch_dir: Relevant location to store the created stimuli.
    :param condition: train, test, or val.
    '''

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
        im = Image.open('stimuli/source/{0}/{1}'.format(stim_dir, o))
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
                             'Brady lab: 6400, Developmental: 1024, Omniglot: 2088.')
    parser.add_argument('--n_val', type=int, default=640,
                        help='Total # validation stimuli. Brady lab: 640, Developmental: 256, Omniglot: 522.')
    parser.add_argument('--n_test', type=int, default=640,
                        help='Total # test stimuli. Brady lab: 640, Developmental: 256, Omniglot: 522.')
    parser.add_argument('--k', type=int, default=2, help='Number of objects per scene.')
    parser.add_argument('--unaligned', action='store_true', default=False,
                        help='Misalign the objects from ViT patches (ie. place randomly).')
    parser.add_argument('--multiplier', type=int, default=1, help='Factor by which to scale up '
                                                                  'stimulus size.')
    parser.add_argument('--stim_dir', type=str, help='Stimulus subdirectory name (inside stimuli/source).', 
                        default='OBJECTSALL')
    parser.add_argument('--rotation', action='store_true', default=False,
                        help='Randomly rotate the objects in the stimuli.')
    parser.add_argument('--scaling', action='store_true', default=False,
                        help='Randomly scale the objects in the stimuli.')

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
    rotation = args.rotation
    scaling = args.scaling

    # Other parameters
    im_size = 224  # Size of base image

    assert im_size % patch_size == 0
    
    aug_str = ''
    if rotation:
        aug_str += 'R'
    if scaling:
        aug_str += 'S'
    if len(aug_str) == 0:
        aug_str = 'N'

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
    
    size_dir = '{0}/{1}'.format(pos_dir, f'trainsize_{n_train}')
    
    try:
        os.mkdir(size_dir)
    except FileExistsError:
        pass

    patch_dir = '{0}/{1}x{1}'.format(size_dir, patch_size * multiplier)

    try:
        os.mkdir(patch_dir)
    except FileExistsError:
        pass

    try:
        os.mkdir('{0}/{1}'.format(patch_dir, k))
    except FileExistsError:
        pass

    patch_dir = patch_dir + '/' + str(k)
    
    try:
        os.mkdir('{0}/{1}'.format(patch_dir, aug_str))
    except FileExistsError:
        pass
    
    patch_dir = patch_dir + '/' + aug_str

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
    object_files = [f for f in os.listdir(f'stimuli/source/{stim_dir}') 
                    if os.path.isfile(os.path.join(f'stimuli/source/{stim_dir}', f)) and f != '.DS_Store']

    # Compute number of unique objects that should be allocated to train/val/test sets
    percent_train = n_train / (n_train + n_val + n_test)
    percent_val = n_val / (n_train + n_val + n_test)
    percent_test = n_test / (n_train + n_val + n_test)

    n_unique = len(object_files)
    n_unique_train = floor(n_unique * percent_train)
    n_unique_val = floor(n_unique * percent_val)
    n_unique_test = floor(n_unique * percent_test)

    # Allocate unique objects
    ofs = object_files  # Copy of object_files to sample from

    object_files_train = random.sample(ofs, k=n_unique_train)
    ofs = [o for o in ofs if o not in object_files_train]

    object_files_val = random.sample(ofs, k=n_unique_val)
    ofs = [o for o in ofs if o not in object_files_val]

    object_files_test = random.sample(ofs, k=n_unique_test)

    assert len(object_files_train) == n_unique_train
    assert len(object_files_val) == n_unique_val
    assert len(object_files_test) == n_unique_test
    assert set(object_files_train).isdisjoint(object_files_test) \
           and set(object_files_train).isdisjoint(object_files_val) \
           and set(object_files_test).isdisjoint(object_files_val)

    create_stimuli(k, n_train, object_files_train, unaligned, patch_size, multiplier,
                   im_size, stim_dir, patch_dir, 'train', rotation=rotation, scaling=scaling)
    create_stimuli(k, n_val, object_files_val, unaligned, patch_size, multiplier,
                   im_size, stim_dir, patch_dir, 'val', rotation=rotation, scaling=scaling)
    create_stimuli(k, n_test, object_files_test, unaligned, patch_size, multiplier,
                   im_size, stim_dir, patch_dir, 'test', rotation=rotation, scaling=scaling)
