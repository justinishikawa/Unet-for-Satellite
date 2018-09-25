import random
import numpy as np
from scipy import ndarray, ndimage

def rotate(image_array: ndarray):
    random_degree = 15
    return ndimage.rotate(image_array,random_degree, reshape=False)

def sobel(image_array: ndarray): 
    blurred = ndimage.uniform_filter(image_array)
    return blurred

def very_blurred(image_array: ndarray): 
    random_degree = .84
    blurred = ndimage.gaussian_laplace(image_array, sigma=random_degree)
    return blurred

def blur(image_array: ndarray): 
    random_degree = .23
    blurred = ndimage.gaussian_filter(image_array, sigma=random_degree)
    return blurred

def vertical_flip(image_array: ndarray):
    # add vertical flip
    return image_array[::-1,:]

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def nothing(image_array: ndarray):
    return image_array

# dictionary of the transformations we defined earlier
available_transformations = {
    'blur': blur,
    'vertical_flip': vertical_flip,
    'horizontal_flip': horizontal_flip,
    'very_blurred': very_blurred,
    'sobel': sobel,
    'rotate': rotate,
    'nothing': nothing,
    'nothing2': nothing}


def get_rand_patch(img, mask, sz):
    """
    :param img: ndarray with shape (x_sz, y_sz, num_channels)
    :param mask: binary ndarray with shape (x_sz, y_sz, num_classes)
    :param sz: size of random patch
    :return: patch with shape (sz, sz, num_channels)
    """
    assert len(img.shape) == 3 and img.shape[0] > sz and img.shape[1] > sz and img.shape[0:2] == mask.shape[0:2]
    xc = random.randint(0, img.shape[0] - sz)
    yc = random.randint(0, img.shape[1] - sz)
    patch_img = img[xc:(xc + sz), yc:(yc + sz)]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]
    return patch_img, patch_mask


def get_patches(x_dict, y_dict, n_patches, sz):
    x = list()
    y = list()
    total_patches = 0
    while total_patches < n_patches:
        img_id = random.sample(x_dict.keys(), 1)[0]
        img = x_dict[img_id]
        mask = y_dict[img_id]
        img_patch_temp, mask_patch_temp = get_rand_patch(img, mask, sz)
        key = random.choice(list(available_transformations))
        img_patch = available_transformations[key](img_patch_temp)
        mask_patch = available_transformations[key](mask_patch_temp)
        x.append(img_patch)
        y.append(mask_patch)
        total_patches += 1
    print('Generated {} patches'.format(total_patches))
    return np.array(x), np.array(y)


