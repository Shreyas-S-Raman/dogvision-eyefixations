import argparse
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.relpath('..'))


from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

import random
import colorsys
from mrcnn import utils
import mrcnn.model as MRCNN_model
from mrcnn import visualize
from mrcnn.config import Config
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib
import json
import pdb
import numpy as np
import cv2
import functools
from PIL import Image
import math
from skimage.draw import disk

class DogVideoMaskHelper:

    def get_polygon_mask(self, polygon_coord, img_size):

        '''Generates a boolean mask array from a list of 2D coordinates defining the coordinates of the mask area'''

        height, width = img_size

        boolean_mask = np.zeros((height, width))

        cv2.fillPoly(img = boolean_mask, pts = [polygon_coord], color = 1) #fill the polygon coordinates with '1's

        return boolean_mask


    def convert_polygon_percentage(self, polygon_coord, img_size):

        '''converts polygon coord in % to pixel values'''

        height, width = img_size

        polygon_coord[:,0] = (polygon_coord[:,0]/100)*width
        polygon_coord[:,1] = (polygon_coord[:,1]/100)*height

        return polygon_coord

def computeIntersection(pred_mask, fixation_mask):
    '''computes IoU for 2 binary masks'''

    return np.sum(pred_mask*fixation_mask)

def accumulateMasks(pred_mask, cumulative_mask):
    return np.logical_or(pred_mask, cumulative_mask)

def computeMaskArea(pred_mask):
    return np.sum(pred_mask)/(pred_mask.shape[0]*pred_mask.shape[1])

def apply_mask(image, mask, color, alpha=0.3):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_instances(image, boxes, masks, class_ids, class_names, fixation_coord, save_path, radius,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.5, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)


    x_cor, y_cor = fixation_coord
    fixation_circle = plt.Circle((x_cor,y_cor), radius, color = 'red', alpha = 0.6)
    ax.add_patch(fixation_circle)

    ax.imshow(masked_image.astype(np.uint8))

    plt.savefig(save_path)


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

def createFixationMask(rows, cols, y_cor, x_cor, radius):

    fixation_mask = np.zeros((rows, cols))

    rr, cc = disk((y_cor, x_cor), radius, shape=(rows, cols))

    fixation_mask[rr,cc] = 1

    return fixation_mask




def parse_annotations(file_annotation, dogvideo_mask_helper):

    annotations = file_annotation['annotations']

    class_predictions = []

    num_obj = len(annotations[0]['result'])
    if num_obj ==0:
        return None, None

    width = int(annotations[0]['result'][0]['original_width']); height = int(annotations[0]['result'][0]['original_height'])

    #num_obj = len(annotations[0]['result'])
    masks = np.zeros((height, width, num_obj))

    for a in annotations:

        for i,r in enumerate(a['result']):

            width = int(r['original_width']); height = int(r['original_height'])


            label = r['value']['polygonlabels'][0].lower()
            mask_polygon = np.array(r['value']['points'])

            mask_polygon = dogvideo_mask_helper.convert_polygon_percentage(mask_polygon, (height, width))
            mask_array = dogvideo_mask_helper.get_polygon_mask(mask_polygon.astype(np.int32), (height,width))

            class_predictions.append(label)
            masks[:,:,i] = mask_array

    prediction = {'masks':masks}

    return prediction, class_predictions


#list of target classes to detect
CLASSES_LIST = ['person', 'plant_horizontal', 'plant_vertical','building','sky','bench_chair','pavement','pole','sign','construction','bicycle','scooter','car','bus','sculpture']
TARGET_CLASSES = set(CLASSES_LIST)


#create arg parser for experiment no
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', choices = ['1','2','3','4','5','6','7','8','9'])
parser.add_argument('--weights_file', required = False)
parser.add_argument('--num_classes', required = False, default = None)
parser.add_argument('--default', required = True, choices = ["True", "False"])
parser.add_argument('--ground_truth',required=True, choices = ["True","False"])
parser.add_argument('--dog_name', required = True, choices = ['daisy','suna','kermit','goose'])
parser.add_argument('--save_fail', required = True, choices = ["True","False"])


args = parser.parse_args()
args.default = True if args.default == 'True' else False
args.save_fail = True if args.save_fail == 'True' else False
args.ground_truth = True if args.ground_truth=='True' else False


radius_error = {'daisy': math.ceil(58.9352381), 'suna': math.ceil(16.7494231), 'kermit':math.ceil(34.3156), 'goose': math.ceil(27.66145)}
args.pixel_radius = radius_error[args.dog_name]


class InferenceConfig(Config):

    #custom MaskRCNN configuration for Dog_Fixaiton dataset
    NAME = "Custom_Inference_Configuration"

    #count of GPUs + images with GPU
    GPU_COUNT = 1; IMAGES_PER_GPU = 1
    #Note: batch size is GPU_COUNT*IMAGES_PER_GPU = 30 [1600*0.5/30 = 25 weight updates]

    #number of custom classes (42 total new classes +background )

    '''
    Custom Classes:

    Human
    1. person, 2. hand, 3. legs, 4. foot, 5. eye, 6. head

    Animal
    7. dog, 8. bird, 9. cat, 10. squirrel

    Natural Static
    11. plant_horizontal (grass), 12. plant_vertical (trees + plants), 13. sky

    Man-Made Static
    14. building: 15. window, 16. door
    17. bench_chair, 18. sculpture, 19. construction
    20. pavement, 21. road, 22. pole, 23. sign, 24. fire_hydrant, 25. fence

    Vehicle
    26. bicycle, 27. scooter, 28. car, 29. bus, 30. motorcycle

    Miscellaneous
    31. ball, 32. dog_toy, 33. frisbee
    '''

    NUM_CLASSES = int(args.num_classes)+1 if not bool(args.default) else 81

    MAX_GT_INSTANCES = 300

    #setting minimum confidence prediction to 80% i.e. ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.8


pdb.set_trace()
#check default v.s. experiment from args + assign to global configs below
WEIGHTS_FILE = None; CLASS_NAMES = None; CONFIG = None
IMAGE_DIR = os.path.relpath('../../../Datasets/Dog_Fixations/{}/fixation_frames'.format(args.dog_name))
FIXATION_DATA_DIR = os.path.relpath('../../../Datasets/Dog_Fixations/{}/{}.csv'.format(args.dog_name,args.dog_name))

extension = 'default' if bool(args.default) else 'experiment_{}'.format(int(args.experiment))
extension = 'ground_truth' if bool(args.ground_truth) else extension

SAVE_DIR = os.path.relpath('./fixation_predictions/{}'.format(extension))
MISSING_PRED_DIR = os.path.relpath('./fixation_predictions/{}/missing_predictions'.format(extension))

#creating save dir if not present
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

if not os.path.isdir(MISSING_PRED_DIR):
    os.mkdir(MISSING_PRED_DIR)

#load csv fixation data file
fixation_dataframe = pd.read_csv(FIXATION_DATA_DIR)



if bool(args.default) and not bool(args.ground_truth):

    #MS-COCO configuration and weights
    sys.path.append(os.path.relpath("../samples/coco/"))
    import coco as configuration_file
    WEIGHTS_FILE = os.path.relpath("../samples/coco/mask_rcnn_coco_weights.h5")

    #store class names in order
    CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench_chair', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'bench_chair', 'couch', 'plant_vertical', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']

elif not args.ground_truth:

    #extract weights filename (.h5) from list of weights files
    trained_weights = os.path.relpath('logs/experiment_{}/weights'.format(int(args.experiment)))
    weight_filename = sorted(os.listdir(trained_weights))[int(args.weights_file)]

    WEIGHTS_FILE = os.path.relpath('logs/experiment_{}/weights/{}'.format(int(args.experiment), weight_filename))


    #store class names in order based on number of classes
    class_names_dict = {48: ['BG','dog','cat','squirrel','rabbit','bird','duck','raccoon','hamster','mouse','lizard','insect','bear','deer','horse',
                    'car','truck','bicycle','van','motorcycle','skateboard','bus','airplane','boat','train',
                    'person','human_hand','human_eye','hman_head','human_foot','rider',
                    'plant_vertical','plant_vertical','flower','plant_horizontal','sky',
                    'sidewalk','road','pole',
                    'building','wall','fence','house','skyscraper','building_entrance', 'fire_hydrant',
                    'ball','dog_toy','frisbee'],
                     33: ["BG","person","hand","legs","foot","head","eye","dog","bird","cat","squirrel",\
                                "plant_horizontal","plant_vertical","sky","building","window","door", "bench_chair","sculpture","construction",\
                                "pavement","road","pole","sign","fire_hydrant","fence",\
                                "bicycle","scooter","car","bus","motorcycle",\
                                "ball","dog_toy","frisbee"]}

    CLASS_NAMES = class_names_dict[int(args.num_classes)]




'''Step 1: create model in inference mode'''
if not args.ground_truth:
    CONFIG = InferenceConfig()
    CONFIG.IMAGES_PER_GPU = 1
    CONFIG.BATCH_SIZE = 1
    CONFIG.display()

    Mask_RCNN_model = MRCNN_model.MaskRCNN(mode="inference", model_dir ='./' , config = CONFIG)
    Mask_RCNN_model.load_weights(WEIGHTS_FILE, by_name = True)

else:
    DATASET_PATH = os.path.relpath('../../../Datasets/Dog_Video_Frames/split_dataset.csv')
    dataset = pd.read_csv(DATASET_PATH)[['dog','frame','filename']]
    dataset = dataset[dataset['dog']==args.dog_name]

    ANNOTATION_PATH = os.path.relpath('../../../Datasets/Dog_Video_Frames/annotations.json')

    with open(ANNOTATION_PATH) as f:
        annotation_content = json.load(f)

    dogvideo_mask_helper = DogVideoMaskHelper()


'''Step 2: iterate over fixation frames + make predictions'''
all_classes = []
fixation_classes = []
fixation_classes_no_background = []
union_areas = []
area_distribution = []

for i,row in enumerate(fixation_dataframe.iterrows()):

    print('Image: {} of {}'.format(i+1, len(fixation_dataframe)))

    #extract row data
    row = row[1]

    #extract frame number + x-cor and y-cor for fixation
    frame = row.FrameNumber
    x_cor = int(row.PORX); y_cor = int(row.PORY)


    #convert image to array
    image_file = os.path.join(IMAGE_DIR, '{}.jpg'.format(frame))

    try:
        image_array = matplotlib.image.imread(image_file)
    except:
        all_classes.append('')
        fixation_classes.append('')
        fixation_classes_no_background.append('')
        union_areas.append('')
        area_distribution.append('')
        continue
    #pdb.set_trace()
    fixation_mask = createFixationMask(image_array.shape[0], image_array.shape[1], y_cor, x_cor, int(args.pixel_radius))

    #generate predictions for image
    if args.ground_truth:
        filename = dataset[dataset['frame']==str(frame)]['filename']

        if len(filename)>1:
            print('Duplicate!')
            pdb.set_trace()
        try:
            filename = filename.values[0]
            file_annotation = annotation_content[filename]
        except:
            all_classes.append('')
            fixation_classes.append('')
            fixation_classes_no_background.append('')
            union_areas.append('')
            area_distribution.append('')
            continue


        prediction, class_predictions = parse_annotations(file_annotation, dogvideo_mask_helper)

        if prediction is None and class_predictions is None:
            all_classes.append('')
            fixation_classes.append('')
            fixation_classes_no_background.append('')
            union_areas.append('')
            area_distribution.append('')
            continue

    else:
        prediction = Mask_RCNN_model.detect([image_array], verbose=1)[0]

        #convert class ids to class name predictions
        class_predictions = [CLASS_NAMES[class_id] for class_id in prediction['class_ids']]


    class_string = ' '.join(sorted(class_predictions))


    #find class name where fixation overlaps
    overlap_classes = []; prediction_areas = {}
    cumulative_mask = np.zeros((image_array.shape[0], image_array.shape[1]))

    for c, class_name in enumerate(class_predictions):

        pred_mask = prediction['masks'][:,:,c]

        iou = computeIntersection(pred_mask, fixation_mask)
        cumulative_mask = accumulateMasks(pred_mask, cumulative_mask)
        mask_area = computeMaskArea(pred_mask)

        if class_name not in prediction_areas:
            prediction_areas[class_name] = 0

        prediction_areas[class_name] += mask_area


        if iou > 0:
            overlap_classes.append([class_name , iou])


    #adding background class IoU
    union_area = np.sum(cumulative_mask)/(cumulative_mask.shape[0]*cumulative_mask.shape[1])
    cumulative_mask = np.invert(cumulative_mask.astype(np.bool_)).astype(np.int8)

    overlap_classes_no_background = overlap_classes.copy()
    bg_iou = computeIntersection(cumulative_mask, fixation_mask)
    overlap_classes.append(['background',bg_iou])

    #pdb.set_trace()
    total_area = sum([p[1] for p in overlap_classes])
    overlap_classes = list(map(lambda x: x[0] + ': {}'.format(x[1]/total_area),overlap_classes))

    total_area = sum([p[1] for p in overlap_classes_no_background])
    overlap_classes_no_background = list(map(lambda x: x[0] + ': {}'.format(x[1]/total_area), overlap_classes_no_background))

    #saving examples where no classes overlap
    if len(overlap_classes)==1 and args.save_fail and not args.ground_truth:
        fixation_coord = (x_cor, y_cor)
        display_instances(image_array, prediction['rois'], prediction['masks'], prediction['class_ids'], CLASS_NAMES,
                              fixation_coord, os.path.join(MISSING_PRED_DIR,'{}.jpg'.format(frame)), args.pixel_radius, scores = prediction['scores'])

    overlap_classes = ' '.join(overlap_classes)
    prediction_areas = ' '.join(['{}:{}'.format(k,v) for k,v in prediction_areas.items()])
    overlap_classes_no_background = ' '.join(overlap_classes_no_background)

    #add prediction data for current image into dataframe
    all_classes.append(class_string)
    fixation_classes.append(overlap_classes)
    fixation_classes_no_background.append(overlap_classes_no_background)
    union_areas.append(union_area)
    area_distribution.append(prediction_areas)

'''Step 3: add new columns to dataframe'''
print('Saving CSV file!')
#pdb.set_trace()
fixation_dataframe['all_classes'] = all_classes
fixation_dataframe['fixation_classes'] = fixation_classes
fixation_dataframe['fixation_classes_no_background'] = fixation_classes_no_background
fixation_dataframe['union_areas'] = union_areas
fixation_dataframe['percentage_area'] = area_distribution

fixation_dataframe.to_csv(os.path.join(SAVE_DIR, '{}.csv'.format(args.dog_name)))
