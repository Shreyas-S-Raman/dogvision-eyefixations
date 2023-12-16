import argparse
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.relpath('..'))

from mrcnn import utils
import mrcnn.model as MRCNN_model
from mrcnn import visualize
from mrcnn.visualize import display_instances
from mrcnn.config import Config
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib
import json
import pdb
import numpy as np
import cv2
from PIL import Image
import random

#global CLASS_COLORS
#CLASS_COLORS = None

def computeIoU(mask1, mask2):
    '''computes IoU for 2 binary masks'''
    return np.sum(mask1*mask2)/np.sum(np.logical_or(mask1,mask2))


def increment_iou_predictions(pred_mask, gt_masks, class_name):
    '''increment IoU prediction dict'''

    #track gt mask with highest IoU
    max_IoU = float('-inf')


    for gt_mask in gt_masks:
        max_IoU = max(computeIoU(pred_mask, gt_mask), max_IoU)

    iou_predictions[class_name] += max_IoU

def increment_class_confidences(class_confidences, class_name, conf):
    '''increment prediction counts'''

    class_confidences[class_name] += conf

def increment_class_counts(test_dataset, class_counts):
    '''increment target class counts dictionary'''

    #set of classes per image
    test_dataset_classes = []
    test_dataset_unique_classes = []

    for row in test_dataset.iterrows():
        
        curr_class_counts = {}	

        class_string = row[1]['class_list']

        if int(row[1]['class_count'])==0:
            continue
        
        gt_classes = class_string.split(' ')
        #gt_classes = class_string.split(' ') if int(row[1]['class_count']) > 0 else []

        test_dataset_classes.append(set(gt_classes))
        test_dataset_unique_classes.append(gt_classes)

        for class_name in gt_classes:
            class_counts[class_name] += 1
	    

    return test_dataset_classes, test_dataset_unique_classes

def populate_confusion_matrix(gt_class_masks, pred_classes, pred_masks, confusion_matrix):
    #pdb.set_trace()
    gt_classes = []
    gt_masks = []
    for (k,v) in gt_class_masks.items():
        gt_classes += [k]*len(v)
        gt_masks += v
    

   
    temp_confusion_matrix = {class_name: {class_name: 0 for class_name in list(TARGET_CLASSES) + ['background']} for class_name in list(TARGET_CLASSES) + ['background']}

    pred_matching_gt = []

    for i in range(pred_masks.shape[-1]):
        max_IoU = float('-inf'); max_IoU_idx = None
        for j, gt_mask in enumerate(gt_masks):

            if computeIoU(pred_masks[:,:,i], gt_mask) > max_IoU and computeIoU(pred_masks[:,:,i], gt_mask)>0:
                max_IoU = computeIoU(pred_masks[:,:,i], gt_mask)
                max_IoU_idx = j

        pred_matching_gt.append(max_IoU_idx)
    #pdb.set_trace()
    #check in case of false negatives
    missing_gt_idxs = list( set(range(len(gt_masks))) - set(pred_matching_gt)  )
    missing_gt_matching_pred = []

    for idx in missing_gt_idxs:
        max_IoU = float('-inf'); max_IoU_idx = None
        for i in range(pred_masks.shape[-1]):

            if computeIoU(pred_masks[:,:,i], gt_masks[idx]) > max_IoU and computeIoU(pred_masks[:,:,i], gt_masks[idx])>0:
                max_IoU = computeIoU(pred_masks[:,:,i], gt_mask[idx])
                max_IoU_idx = i

        missing_gt_matching_pred.append(max_IoU_idx)


    #populate counts to confusion matrix

    for i, idx in enumerate(pred_matching_gt):
	
        if idx is None:
            #confusion_matrix['background'][pred_classes[i]] += 1
            temp_confusion_matrix['background'][pred_classes[i]] += 1
            continue
        #confusion_matrix[gt_classes[idx]][pred_classes[i]] += 1
        temp_confusion_matrix[gt_classes[idx]][pred_classes[i]] += 1

    for i, idx in zip(missing_gt_idxs, missing_gt_matching_pred):
        if idx is None:
            temp_confusion_matrix[gt_classes[i]]['background']+=1
            #confusion_matrix[gt_classes[i]]['background']+=1
            continue
        #confusion_matrix[gt_classes[i]][pred_classes[idx]] += 1
        temp_confusion_matrix[gt_classes[i]][pred_classes[idx]] += 1

    for c in set(gt_classes):
        assert(gt_classes.count(c) <= sum(temp_confusion_matrix[c].values())), 'ERROR: sum for {} was {}, expected {}'.format(c, sum(temp_confusion_matrix[c].values()), gt_classes.count(c))
        #assert(gt_classes.count(c) <= sum(temp_confusion_matrix[c].values()))
    #pdb.set_trace()
    return temp_confusion_matrix

def convert_to_class_counts(examples_class_set, example_classes):
    
    class_counts = {c: 0 for c in examples_class_set}
    
    for e in example_classes:
            class_counts[e] += 1
    return class_counts

def get_gt_masks(test_dataset, annotation_content):
    '''store dictionary of boolean masks for each test image'''

    #dictionary of masks per image
    test_dataset_masks = []
    all_class_counts = {}

    for row in test_dataset.iterrows():

        image_masks = {}

        filename = row[1]['filename']

        file_info = annotation_content[filename]['annotations'][0]['result']

        #iterate through annotations for 1 image
        for i,annotation in enumerate(file_info):

            img_size = (annotation['original_height'], annotation['original_width'])

            #extract each annotation mask
            polygon_coord = np.array(annotation['value']['points'])
            polygon_coord = convert_polygon_percentage(polygon_coord, img_size)


            annot_mask = get_polygon_mask(polygon_coord.astype(np.int32), img_size)

            #extract class name for annotation
            class_name = annotation['value']['polygonlabels'][0].lower()


            if class_name not in image_masks:
                image_masks[class_name] = []

            image_masks[class_name].append(annot_mask)
            
            if class_name not in all_class_counts:
                all_class_counts[class_name] = 0
            all_class_counts[class_name] += 1
        
        #add all image masks to dataset list
        if len(list(image_masks.keys())) > 0:
            test_dataset_masks.append(image_masks)
        else:
            print('NO GT MASKS FOUND')
            #test_dataset_masks.append(image_masks)
    print('gt mask: class counts: ', all_class_counts)
    return test_dataset_masks

#list of target classes to detect
TARGET_CLASSES = set(['person', 'plant_horizontal', 'plant_vertical','building','sky','bench_chair','pavement','pole','sign','construction','bicycle','scooter','car','bus','sculpture'])
TARGET_CLASSES_LIST = list(TARGET_CLASSES)

#create arg parser for experiment no
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', choices = ['1','2','3','4','5','6','7','8','9'])
parser.add_argument('--weights_file', required = False)
parser.add_argument('--num_classes', required = False, default = None)
parser.add_argument('--default', required = True, choices = ["True", "False"])
parser.add_argument('--randomize', required=True, choices=["True", "False"])
parser.add_argument('--visualize', required=False, action='store_true')

parser.add_argument('--add_labels', required=True, choices=["True", "False"])

args = parser.parse_args()
args.default = True if args.default == 'True' else False
args.randomize = True if args.randomize == 'True' else False
args.add_labels = True if args.add_labels == 'True' else False

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

#check default v.s. experiment from args + assign to global configs below
WEIGHTS_FILE = None; CLASS_NAMES = None; CONFIG = None
IMAGE_DIR = os.path.relpath('../../Datasets/dog_video_dataset/images')

extension = 'default' if bool(args.default) else 'experiment_{}'.format(int(args.experiment))
SAVE_DIR = os.path.relpath('./evaluation_output/{}'.format(extension))

#creating save dir if not present
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

DATASET_FILE = os.path.join('../../Datasets/dog_video_dataset/split_dataset.csv')
ANNOTATION_FILE = os.path.join('../../Datasets/dog_video_dataset/annotations.json')

#load json annotations file
with open(ANNOTATION_FILE,'r') as f:
    annotation_content = json.loads(f.read())

f.close()

#pdb.set_trace()

if bool(args.default):

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

else:

    #extract weights filename (.h5) from list of weights files
    trained_weights = os.path.relpath('logs/experiment_{}/weights'.format(int(args.experiment)))
    weight_filename = sorted(os.listdir(trained_weights))[int(args.weights_file)]

    WEIGHTS_FILE = os.path.relpath('logs/experiment_{}/weights/{}'.format(int(args.experiment), weight_filename))
    print(WEIGHTS_FILE)

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





'''Step 1: create subset of test set for visualization'''
visualized_test = set(['e5c117bb-26443.jpg','e8248c67-22692.jpg','5aea7414-19914.jpg','96c0f29e-10749.jpg','cb92eec9-17334.jpg','ef722188-14256.jpg','d542bd89-14557.jpg','9232b35b-21586.jpg'])

visualized_test = [ matplotlib.image.imread(os.path.join(IMAGE_DIR, filename)) for filename in visualized_test]


'''Step 2: create model in inference mode'''
CONFIG = InferenceConfig()
CONFIG.IMAGES_PER_GPU = len(visualized_test)
CONFIG.BATCH_SIZE = len(visualized_test)
CONFIG.display()


Mask_RCNN_model = MRCNN_model.MaskRCNN(mode="inference", model_dir ='./' , config = CONFIG)
#pdb.set_trace()
Mask_RCNN_model.load_weights(WEIGHTS_FILE, by_name = True)

'''Step 3: run inference predictions on all test set samples'''
print("Running inference on visual samples...")

# Returns a LIST of prediction DICTIONARIES i.e. one dictionary per image
# Returns a list of dicts, one dict per image. The dict contains:
# rois: [N, (y1, x1, y2, x2)] detection bounding boxes
# class_ids: [N] int class IDs
# scores: [N] float probability scores for the class IDs
# masks: [H, W, N] instance binary masks

#pdb.set_trace()

if args.visualize:
    predictions = Mask_RCNN_model.detect(visualized_test, verbose=1)

    #display_instances(image, boxes, masks, class_ids, class_names, scores=None, title="", figsize=(16, 16), ax=None, show_mask=True, show_bbox=True, colors=None, captions=None, class_labels=args.add_labels)

    for i, prediction in enumerate(predictions):
        test_pred = predictions[i]
        masked_image = display_instances(visualized_test[i], test_pred['rois'], test_pred['masks'], test_pred['class_ids'], CLASS_NAMES, scores=test_pred['scores'], class_labels = args.add_labels)
        plt.imshow(masked_image)
        plt.savefig(os.path.join(SAVE_DIR, 'sample_{}.png'.format(i)))
#     boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
#     masks: [height, width, num_instances]
#     class_ids: [num_instances]
#     class_names: list of class names of the dataset
#     scores: (optional) confidence scores for each box
#     title: (optional) Figure title
#     show_mask, show_bbox: To show masks and bounding boxes or not
#     figsize: (optional) the size of the image
#     colors: (optional) An array or colors to use with each object
#     captions: (optional) A list of strings to use as captions for each object


#for i,prediction in enumerate(predictions):
#
#    test_pred = predictions[i]
#    masked_image  = display_instances(visualized_test[i], test_pred['rois'], test_pred['masks'], test_pred['class_ids'], CLASS_NAMES, scores= test_pred['scores'])
#
#    plt.imshow(masked_image)
#    plt.savefig(os.path.join(SAVE_DIR, 'sample_{}.png'.format(i)) )



#pdb.set_trace()
'''Step 4: run overall prediction +  save final accuracies and IoUs'''
confusion_matrix = {class_name: {class_name: 0 for class_name in list(TARGET_CLASSES) + ['background']} for class_name in list(TARGET_CLASSES) + ['background']}

#class prediction + IoU prediction dictionaries
class_counts = {class_name:0 for class_name in TARGET_CLASSES}; class_pred_counts = {class_name: 0 for class_name in TARGET_CLASSES}
class_pred_counts_capped = {class_name:0 for class_name in TARGET_CLASSES}
class_confidences = {class_name:0 for class_name in TARGET_CLASSES}; iou_predictions = {class_name:0 for class_name in TARGET_CLASSES}

false_negatives = {class_name:0 for class_name in TARGET_CLASSES}; false_positives = {class_name:0 for class_name in TARGET_CLASSES}

print(TARGET_CLASSES)
print(iou_predictions)
#pdb.set_trace()
#read csv file into dataframe
test_dataset = pd.read_csv(DATASET_FILE)
test_dataset = test_dataset[test_dataset['subset']=='test']
#pdb.set_trace()
#populate target class_counts dict
gt_classes, gt_unique_classes = increment_class_counts(test_dataset, class_counts)

#generate list of target masks (for IoU)
#pdb.set_trace()
gt_masks = get_gt_masks(test_dataset, annotation_content)


test_data = []

for row in test_dataset.iterrows():
     try:
         img_array = matplotlib.image.imread(os.path.join(IMAGE_DIR, row[1]['filename'] ))
         img_array = img_array if img_array.shape[2]==3 else np.asarray(Image.fromarray(np.uint8(img_array*255)).convert('RGB'))
         test_data.append(img_array)
     except:
         print('failed to load image: ', row[1]['filename'])
test_dataset = test_data


#update configs for full test set eval
CONFIG.IMAGES_PER_GPU = 1
CONFIG.BATCH_SIZE = 1
Mask_RCNN_model = MRCNN_model.MaskRCNN(mode="inference", model_dir ='./' , config = CONFIG)
Mask_RCNN_model.load_weights(WEIGHTS_FILE, by_name = True)

#generate predictions
#pdb.set_trace()


print('\n \n Running on evaluation dataset')

for i, image in enumerate(test_dataset):

    print('Predicting image {} of {}'.format(i+1, len(test_dataset)))
    image_pred = Mask_RCNN_model.detect([image], verbose=1)[0]

    #pdb.set_trace()
    print('Done prediction!')
    pred_counts = {class_name:0 for class_name in TARGET_CLASSES}

    #convert class ids to class name predictions
    if not args.randomize:
        class_predictions = [CLASS_NAMES[class_id] for class_id in image_pred['class_ids']]
    else:
        class_predictions = [TARGET_CLASSES_LIST[random.sample(range(0,len(TARGET_CLASSES_LIST)), k=1)[0]] for class_id in image_pred['class_ids']]

    print('Convereted class id to names')

    image_gt_class_counts = convert_to_class_counts(gt_classes[i],gt_unique_classes[i])
    print('Ran convert_to_class_counts')
    #pdb.set_trace()
    for gt_class in gt_classes[i]:

        #false negative counts
        if gt_class not in set(class_predictions):
            false_negatives[gt_class] += 1

    print('Populated false negatives')
    
    for c, class_name in enumerate(class_predictions):


        #class prediction counts
        
        if (class_name in TARGET_CLASSES) and (class_name in gt_classes[i]) and (class_name in gt_masks[i]):

            increment_class_confidences(class_confidences, class_name, image_pred['scores'][c])

            '''TODO: update to find gt mask and pred mask with highest IoU + report/store that'''
            try:            
                increment_iou_predictions(image_pred['masks'][:,:,c], gt_masks[i][class_name], class_name)
            except:
                pdb.set_trace()
            #if pred_counts[class_name] < image_gt_class_counts[class_name]:
            #    pred_counts[class_name] += 1
            #    class_pred_counts_capped[class_name] += 1
            class_pred_counts[class_name] += 1

        print('Incremented Counts')

        #false positive counts
        if (class_name in TARGET_CLASSES) and (class_name not in gt_classes[i]):
            false_positives[class_name] += 1
        print('Increment FP')
    print('Incremented FP, IOU, Counts')
    temp_confusion_matrix = populate_confusion_matrix(gt_masks[i], class_predictions, image_pred['masks'], confusion_matrix)
    
    for c1 in temp_confusion_matrix.keys():
        for c2 in temp_confusion_matrix[c1].keys():
            confusion_matrix[c1][c2] += temp_confusion_matrix[c1][c2]


'''Step 5: Save final output: visualization images and csv files'''
print('Saving final logs ....')

FNR = {}; FPR = {}

for class_name in TARGET_CLASSES:

    class_confidences[class_name] = class_confidences[class_name]/class_pred_counts[class_name] if class_pred_counts[class_name] > 0 else 0
    iou_predictions[class_name] = iou_predictions[class_name]/class_pred_counts[class_name] if class_pred_counts[class_name] > 0 else 0

    FNR[class_name] = false_negatives[class_name]/class_counts[class_name]
    FPR[class_name] = false_positives[class_name]/class_counts[class_name]


metrics_dataframe = pd.DataFrame(columns = ["class","count","true_positives", "confidence","iou","false_negatives","false_positives", "FNR", "FPR"])

for class_name in TARGET_CLASSES:

    data_row = {"class":class_name, "count": class_counts[class_name], "true_positives": class_pred_counts_capped[class_name] , "confidence": class_confidences[class_name],"iou": iou_predictions[class_name],"false_negatives": false_negatives[class_name],"false_positives": false_positives[class_name], "FNR": FNR[class_name], "FPR":FPR[class_name]}

    metrics_dataframe = metrics_dataframe.append(data_row, ignore_index = True)


metrics_dataframe.to_csv(os.path.join(SAVE_DIR, 'metrics.csv'))

print('Confusion Matrix')
print(confusion_matrix)
