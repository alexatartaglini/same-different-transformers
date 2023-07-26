from PIL import Image, ImageOps
import numpy as np
import os
import random
import itertools
from math import floor
from itertools import product

# Possible palette: colors that are distinct based on human perception
# https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
kelly = dict(vivid_yellow=(255, 179, 0),
            strong_purple=(128, 62, 117),
            vivid_orange=(255, 104, 0),
            very_light_blue=(166, 189, 215),
            vivid_red=(193, 0, 32),
            grayish_yellow=(206, 162, 98),
            medium_gray=(129, 112, 102),
            vivid_green=(0, 125, 52),
            strong_purplish_pink=(246, 118, 142),
            strong_blue=(0, 83, 138),
            strong_yellowish_pink=(255, 122, 92),
            strong_violet=(83, 55, 122),
            vivid_orange_yellow=(255, 142, 0),
            strong_purplish_red=(179, 40, 81),
            vivid_greenish_yellow=(244, 200, 0),
            strong_reddish_brown=(127, 24, 13),
            vivid_yellowish_green=(147, 170, 0),
            deep_yellowish_brown=(89, 51, 21),
            vivid_reddish_orange=(241, 58, 19),
            dark_olive_green=(35, 44, 22))

# alternative palette that is just a grid of equally spaced RGB points
oned = np.linspace(0, 255, 4, dtype=int)
grid = np.array(list(product(oned,oned,oned)))
grid = {f'color{i}':tuple(rgb) for i,rgb in enumerate(grid)}

# given a PIL image, grayscales and then tints the image the given color.
# modified from https://gist.github.com/WChargin/d8eb0cbafc4d4479d004 
# https://stackoverflow.com/questions/32578346/how-to-change-color-of-image-using-python
def tint(img, color, factor=0.4):
    img = ImageOps.grayscale(img).convert('RGB')
    color = np.array(color)
    operation = (1 - factor) * np.eye(3)
    r, c = operation.shape 
    temp = np.eye(4)
    temp[:r, :c] = operation
    operation = temp
    operation[:3, 3] = factor * color
    tinted = img.convert('RGB', tuple(operation[:3,:].flatten()))
    # turn tinted background back to white
    arr = np.array(tinted)
    mask = (arr == arr[0][0]).all(axis=-1)
    arr[mask] = (255,255,255)
    return Image.fromarray(arr).convert('RGB')

def create_devdis(k, n, objects, unaligned, patch_size, multiplier, im_size, devdis,
                   out_dir, rotation=False, scaling=False, palette=kelly):
    '''
    TODO edit comment 
    Creates n same_different stimuli, all labelled "same." If n > the number of unique 
    objects used to create the dataset, then randomly selected object pairs will be repeated 
    in unique randomly selected positions until n unique stimuli are created. This code ensures 
    that the number of repeated stimuli is the
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
    :param devdis: Name of source dataset used to construct data (e.g. OBJECTSALL, DEVELOPMENTAL, DEVDIS001). 
    :param out_dir: Relevant location to store the created stimuli.
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

    # start doing work to separate out texture and color 
    binary_strs = ['000', '001', '010', '011', '100', '101', '110', '111']
    combination = devdis[-3:]
    assert combination in binary_strs
    
    # expand objects by making a version of each object in every possible tint
    tinted_objects = []
    for c in palette.keys():
        tinted_objects += [s.split('.png')[0] + f'-{c}.png' for s in objects]
    objects = tinted_objects

    # keep pairs that fulfill our s/d definition 
    if combination == '111': # all same 
        pairs = {tuple([o] * k): [] for o in objects}
    else: # at least one different
        all_different_pairs = list(itertools.combinations(objects, k))
        relevant_different_pairs = []

        for pair in all_different_pairs:
            assert '-' in pair[0] and '-' in pair[1]
            shape1, texture1, color1 = pair[0].split('-')
            shape2, texture2, color2 = pair[1].split('-')
                        
            if combination == '000':
                if (color1 != color2 and texture1 != texture2 and shape1 != shape2):
                    relevant_different_pairs.append(pair)
            elif combination == '001':
                if (color1 != color2 and texture1 != texture2 and shape1 == shape2):
                    relevant_different_pairs.append(pair)
            elif combination == '010':
                if (color1 != color2 and texture1 == texture2 and shape1 != shape2):
                    relevant_different_pairs.append(pair)
            elif combination == '011':
                if (color1 != color2 and texture1 == texture2 and shape1 == shape2):
                    relevant_different_pairs.append(pair)
            elif combination == '100':
                if (color1 == color2 and texture1 != texture2 and shape1 != shape2):
                    relevant_different_pairs.append(pair)
            elif combination == '101':
                if (color1 == color2 and texture1 != texture2 and shape1 == shape2):
                    relevant_different_pairs.append(pair)
            elif combination == '110':
                if (color1 == color2 and texture1 == texture2 and shape1 != shape2):
                    relevant_different_pairs.append(pair)
        
        if len(relevant_different_pairs) < len(objects):
            print(relevant_different_pairs)
        pairs = {o: [] for o in random.sample(relevant_different_pairs, k=len(objects))}
        
    # Assign at least one position to each pair
    for pair in pairs.keys():
        if not unaligned:
            c = random.sample(possible_coords, k=k)
            pairs[pair].append(c)
        else:  # Code needs to be altered for k > 2
            c1 = tuple(random.sample(list(coords), k=2))
            c2 = tuple(random.sample(list(coords), k=2))

            # Ensure there is no overlap
            while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                    and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                c2 = tuple(random.sample(list(coords), k=2))

            pairs[pair].append([c1, c2])

    # Generate more unique positions for pairs until desired number is achieved
    pair_keys = list(pairs.keys())
    examples_generated = len(pair_keys)
    pair_counts = [1] * examples_generated

    while examples_generated < n:
        key = random.choice(pair_keys)

        # check whether this key has already exhausted all placements
        if not unaligned: 
            while len(pairs[key]) == len(possible_coords):
                key = random.choice(pair_keys)

        idx = pair_keys.index(key)

        existing_positions = [set(c) for c in pairs[key]]

        if not unaligned:
            c = random.sample(possible_coords, k=k)

            while set(c) in existing_positions:  # Ensure unique position
                c = random.sample(possible_coords, k=k)

            pairs[key].append(c)
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

            pairs[key].append([c1, c2])

        examples_generated += 1
        pair_counts[idx] += 1

    assert sum(pair_counts) == n

    # Save the stimuli generated above
    object_ims_all = {}

    for o in objects:
        fname = '-'.join(o.split('-')[:2]) + '.png' # remove tint
        color = palette[o.split('-')[-1][:-4]]
        im = Image.open('stimuli/source/{0}/{1}'.format("DEVELOPMENTAL", fname)).convert('RGB')
        im = im.resize((obj_size, obj_size))
        shape_mask = im.point(lambda p: 255 if p < 255 else 0).convert('1')
        base = Image.new('RGB', (obj_size, obj_size), (255, 255, 255))
        im = tint(im, color)
        base.paste(im, mask=shape_mask)
        object_ims_all[o] = base

    sd_class = "same"
    setting = '{0}/{1}'.format(out_dir, sd_class)

    for key in pairs.keys():
        positions = pairs[key]

        for i, p in enumerate(positions):
            base = Image.new('RGB', (im_size, im_size), (255, 255, 255))

            # TODO: fix for k > 2
            obj1 = key[0]
            obj2 = key[1]
            object_ims = [object_ims_all[obj1].copy(), object_ims_all[obj2].copy()]

            if rotation:
<<<<<<< HEAD
                rotation_deg = random.randint(0, 359)
                
                for o in range(len(object_ims)):
                    rotated_obj_o = object_ims[o].rotate(rotation_deg, expand=1, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
                    
                    if rotated_obj_o.size != (obj_size, obj_size):
                        scale_base_o = Image.new('RGB', (max(rotated_obj_o.size), max(rotated_obj_o.size)), (255, 255, 255))
                        scale_base_o.paste(rotated_obj_o, ((max(rotated_obj_o.size) - rotated_obj_o.size[0]) // 2, 
                                                          (max(rotated_obj_o.size) - rotated_obj_o.size[1]) // 2))
                        rotated_obj_o = scale_base_o
                        
                    scale_base_o = Image.new('RGB', (obj_size, obj_size), (255, 255, 255))
                    scale_base_o.paste(rotated_obj_o.resize((obj_size, obj_size)))  
                    
                    object_ims[o] = scale_base_o
                    
            if scaling:
                scale_factor = random.uniform(0.45, 0.9)
                scaled_size = floor(obj_size * scale_factor)
                
                for o in range(len(object_ims)):
                    scale_base = Image.new('RGB', (obj_size, obj_size), (255, 255, 255))
                    scaled_obj_im = object_ims[o].resize((scaled_size, scaled_size))
                    scale_base.paste(scaled_obj_im, ((obj_size - scaled_size) // 2, (obj_size - scaled_size) // 2))
                    object_ims[o] = scale_base
=======
                    rotation_deg = random.randint(0, 359)
                    
                    for o in range(len(object_ims)):
                        rotated_obj_o = object_ims[o].rotate(rotation_deg, expand=1, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
                        
                        if rotated_obj_o.size != (obj_size, obj_size):
                            scale_base_o = Image.new('RGB', (max(rotated_obj_o.size), max(rotated_obj_o.size)), (255, 255, 255))
                            scale_base_o.paste(rotated_obj_o, ((max(rotated_obj_o.size) - rotated_obj_o.size[0]) // 2, 
                                                              (max(rotated_obj_o.size) - rotated_obj_o.size[1]) // 2))
                            rotated_obj_o = scale_base_o
                            
                        scale_base_o = Image.new('RGB', (obj_size, obj_size), (255, 255, 255))
                        scale_base_o.paste(rotated_obj_o.resize((obj_size, obj_size)))  
                        
                        object_ims[o] = scale_base_o
                    
            if scaling:
                    #scaled_obj_idx = random.choice([0, 1])
                    scale_factor = random.uniform(0.45, 0.9)
                    scaled_size = floor(obj_size * scale_factor)
                    
                    for o in range(len(object_ims)):
                        scale_base = Image.new('RGB', (obj_size, obj_size), (255, 255, 255))
                        scaled_obj_im = object_ims[o].resize((scaled_size, scaled_size))
                        scale_base.paste(scaled_obj_im, ((obj_size - scaled_size) // 2, (obj_size - scaled_size) // 2))
                        object_ims[o] = scale_base
>>>>>>> ca524c34bf35f3bad058bb0746e8e927b21997bf
                
                    
            for c in range(len(p)):
                base.paste(object_ims[c], box=p[c])

            base.save('{0}/{1}_{2}_{3}.png'.format(setting, obj1.split('.')[0], obj2.split('.')[0], i), quality=100)

# TODO expand logic to k>2
def call_create_devdis(patch_size, n_val, k, unaligned, multiplier, out_dir, 
                       rotation, scaling, devdis, im_size=224, n_val_tokens=300):

    assert im_size % patch_size == 0
    
    path_elements = out_dir.split('/')
    
    stub = 'stimuli'
    for p in path_elements[1:]:
        try:
            os.mkdir('{0}/{1}'.format(stub, p))
        except FileExistsError:
            pass
        stub = '{0}/{1}'.format(stub, p)

    # we only need one same directory underneath the name
    # e.g. DEVDIS001/unaligned/RS/valsize_6400/same/img.png
    try:
        os.mkdir('{0}/same'.format(out_dir))
    except FileExistsError:
        pass

    # Collect object image paths. Because this is devdis, we're using the 
    # DEVELOPMENTAL source stimuli (and making some tint modifications online). 
    object_files = [f for f in os.listdir(f'stimuli/source/DEVELOPMENTAL') 
                    if os.path.isfile(os.path.join(f'stimuli/source/DEVELOPMENTAL', f)) 
                    and f != '.DS_Store']

    # There are 1792 DEVELOPMENTAL objects to work with. Use 300 unique tokens
    # by default to match other experiments with this dataset size.
    n_unique = len(object_files)
    object_files_val = random.sample(object_files, k=n_val_tokens)

    create_devdis(k, n_val, object_files_val, unaligned, patch_size, multiplier,
                im_size, devdis, out_dir, rotation=rotation, scaling=scaling)