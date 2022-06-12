import json
import os


import numpy as np
import cv2
import pdb

import colorsys
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from matplotlib import pyplot as plt
import random


def get_polygon_mask(polygon_coord, img_size):

    '''Generates a boolean mask array from a list of 2D coordinates defining the coordinates of the mask area'''

    height, width = img_size

    boolean_mask = np.zeros((height, width))

    cv2.fillPoly(img = boolean_mask, pts = [polygon_coord], color = 1) #fill the polygon coordinates with '1's

    return boolean_mask


def convert_polygon_percentage(polygon_coord, img_size):

    '''converts polygon coord in % to pixel values'''

    height, width = img_size

    polygon_coord[:,0] = (polygon_coord[:,0]/100)*width
    polygon_coord[:,1] = (polygon_coord[:,1]/100)*height

    return polygon_coord

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] *(1 - alpha) + alpha * color[c] * 255, image[:, :, c])
    return image

def display_instances(image, masks, class_names, caption_coords, figsize=(16, 16), ax=None, show_mask=True, title = None, colors=None):
    """
    masks: [height, width, num_instances]
    class_names: list of class names of the dataset
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    """
    # Number of instances
    N = masks.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
        exit()


    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    # ax.set_ylim(height + 10, -10)
    # ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()

    for i in range(N):
        color = colors[i]

        # Label
        label = class_names[i]
        caption = "gt_{}".format(label)

        (x1,y1) = caption_coords[i]

        ax.text(x1, y1 + 8, caption, color='w', size=7, backgroundcolor="black")

        # Mask
        mask = masks[i, :, :]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color, alpha = 0.4)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig(title)
    if auto_show:
        plt.show()


ANNOTATION_PATH = os.path.relpath('../../Datasets/dog_video_dataset/annotations.json')
SAMPLES_PATH = os.path.relpath('../ground_truth/samples')
SAVE_PATH = os.path.relpath('../ground_truth')

if not os.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

#load json annotations file
with open(ANNOTATION_PATH,'r') as f:
    content = json.loads(f.read())

f.close()

#iterate sample test images + collecting annotation information
for i, file in enumerate(os.listdir(SAMPLES_PATH)):



    file_info = content[file]['annotations'][0]['result']
    img_array = plt.imread(  os.path.join(SAMPLES_PATH,file))


    #dictionary storing areas of annotations: sorted output
    annotation_array = []; class_list = []; caption_coords = []


    for i,annotation in enumerate(file_info):

        img_size = (annotation['original_height'], annotation['original_width'])

        polygon_coord = np.array(annotation['value']['points'])
        polygon_coord = convert_polygon_percentage(polygon_coord, img_size)


        annot_mask = get_polygon_mask(polygon_coord.astype(np.int32), img_size)

        mask_coords = np.where(annot_mask==1)
        x_cor = mask_coords[0][0]; y_cor = mask_coords[1][0]


        annot_mask = np.ma.masked_where(annot_mask == 0, annot_mask)

        class_name = annotation['value']['polygonlabels'][0].lower()




        #appending data to be plotted
        annotation_array.append(annot_mask)
        class_list.append(class_name)
        caption_coords.append([y_cor, x_cor])


    #convert annotation to 3D array
    annotation_array = np.array(annotation_array)
    display_instances(img_array, annotation_array, class_list, caption_coords , title = os.path.join(SAVE_PATH, 'ground_truth_{}'.format(file)))
