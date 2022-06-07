import pdb
import imgaug.augmenters as iaa
import random
import json
import sys
import datetime
import pandas as pd
import cv2
import numpy as np

import tensorflow as tf
sys.path.append('..')
from mrcnn.config import Config
import mrcnn.model as MaskRCNN_model
from mrcnn import visualize

from mrcnn.utils import Dataset #importing dataset class to configure custom dataset
from mrcnn.model import MaskRCNN

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

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

class CityscapeMaskHelper:

    '''CityScapes Classes: class name use (Cityscapes classname):
            1. car, 2. truck, 3. bus, 4. bicycle, 5. motorcycle, 6. train,
            7. person, 8. person(rider -person on a bike), 9. plant_vertical (vegetation), 10. plant_horizontal (terrain), 11. sky (sky),
            12. road (road), 13. pavement (sidewalk), 14. building (building), 15. pole (pole), 16. fence (fence), 17. wall (wall)'''

    relevant_classes_cityscapes = set(['car','truck','bus','bicycle','motorcycle','person', 'plant_vertical', 'sky', 'road',\
                                'pavement','building','pole','wall'])

    class_conversion_cityscapes = {'rider': 'person', 'vegetation': 'plant_vertical', 'terrain': 'plant_horizontal', 'sidewalk': 'pavement'}

    def get_polygon_mask(self,polygon_coord,img_height, img_width):

        '''Generates a boolean mask array from a list of 2D coordinates defining the coordinates of the mask area'''

        #generate single channel or intensity based array filled with 0s
        boolean_mask = np.zeros((img_height,img_width))
        cv2.fillPoly(boolean_mask,pts=[polygon_coord],color=1)

        return boolean_mask


class DatasetGenerator():

    def __init__(self):

        #array of dictionaries storing dataset
        self.dataset = []

        #json file of configs for dataset sampling
        CONFIG_PATH = os.path.relpath('dataset_configs.json')

        #store sequential augmentation operations
        self.seq_augmenter = None

        self.load_configs(CONFIG_PATH)

    def get_imgaug_function(self, key):
        '''returns img augmentation function given key'''

        imgaug_functions = {  "ElasticTransform": iaa.imgcorruptlike.ElasticTransform,
      "Pixelate": iaa.imgcorruptlike.Pixelate, "JpegCompression": iaa.imgcorruptlike.JpegCompression,
      "Brightness": iaa.imgcorruptlike.Brightness, "Contrast": iaa.imgcorruptlike.Contrast,
      "Fliplr": iaa.flip.Fliplr, "MotionBlur": iaa.MotionBlur, "Fog": iaa.imgcorruptlike.Fog,
      "CoarseDropout": iaa.CoarseDropout, "Affine": iaa.geometric.Affine,"Sharpen": iaa.convolutional.Sharpen }

        return imgaug_functions[key]

    def load_cityscapes_dataset(self, dataset, train):

        #important + relevant data directories
        PATH_TO_DATA = os.path.relpath('../../..Metadata/CityScapes/gtFine/')
        PATH_TO_IMAGES = os.path.relpath('../../../Datasets/leftImg8bit/')

        #extracting list of images and annotations
        data_path = os.path.join(PATH_TO_DATA,'train/CityScapes_train_required_images.txt') if train else  os.path.join(PATH_TO_DATA, 'val/CityScapes_val_required_images.txt')


        with open(data_path) as data_file:
            required_images = data_file.readlines() #reads all lines + appends/cllects each line in txt file
            '''Note : required_images contains list of paths to polygon annotations'''


            for annotation_path in required_images:


                image_id = annotation_path.split('/')[-1] #accesing the file name e.g. strasbourg_000001_018872_gtFine_polygons.json
                image_id = image_id.split('.')[0] #accessing the id e.g. strasbourg_000001_018872_gtFine_polygons
                image_id = "_".join(image_id.split('_')[0:4]) #refiing the id e.g. strasbourg_000001_018872_gtFine
                image_id = "cityscapes_"+image_id

                '''Example Annotation path: /home/ssunda11/Dog_Fixations/Datasets/gtFine/train/strasbourg/strasbourg_000001_018872_gtFine_polygons.json'''

                image_path = annotation_path.split('/')[6:] # e.g. [train,strasbourg,strasbourg_000001_018872_gtFine_polygons.jso
                image_path[-1] = image_path[-1].split('_')[0:3]
                image_path[-1].append('leftImg8bit.png') #e.g. [....,[strasbourg,000001,018872] + [leftImg8bit.png]]


                #rejoining to form correct image path
                image_path[-1] = "_".join(image_path[-1]) #e.g. [....,strasbourg_000001_018872_leftImg8bit.png]
                image_path = '/'.join(image_path) # e.g. train/strasbourg/strasbourg_000001_018872_leftImg8bit.png


                image_path = os.path.join(PATH_TO_IMAGES,image_path) #full finalpath to png image
                #e.g. /home/ssunda11/Dog_Fixations/Datasets/leftImg8bit/ + train/strasbourg/strasbourg_000001_018872_leftImg8bit.png

                annotation_path = annotation_path.split('\n')[0] #remove '\n' character from the path

                #adding images and their polygon annotations to matterport dataset class
                data_entry = { 'source_dataset': 'cityscapes', 'image_id' : image_id, 'image_path' : image_path, 'annot_path': annotation_path }

                dataset.append(data_entry)
        data_file.close()


    def load_dogvideo_dataset(self, dataset, train):
        '''loading all instances of dog video dataset into list'''

        IMAGE_DIR = os.path.relpath('../../../Datasets/Dog_Video_Frames/images')
        DATASET_FILE = os.path.join('../../../Datasets/Dog_Video_Frames/split_dataset.csv')
        ANNOTATION_FILE = os.path.join('../../../Datasets/Dog_Video_Frames/annotations.json')

        #load json annotations file
        with open(ANNOTATION_FILE,'r') as f:
            annotation_content = json.loads(f.read())

        f.close()

        #read csv file into dataframe
        subset = 'train' if train is True else 'test'
        dogvideo_dataset = pd.read_csv(DATASET_FILE)
        dogvideo_dataset = dogvideo_dataset[dogvideo_dataset['subset']== subset]

        skipped_files = set(['4c12fd0f-12705.jpg','e83f4469-36479.jpg'])

        for row in dogvideo_dataset.iterrows():

            filename = row[1]['filename']


            if filename in skipped_files:
                continue

            image_path = os.path.join(IMAGE_DIR, filename)


            file_info = annotation_content[filename]['annotations'][0]['result']

            data_entry = { 'source_dataset': 'dogvideo', 'image_id' : 'dogvideo_'+filename, 'image_path' : image_path, 'annot_path': file_info }

            dataset.append(data_entry)


    def load_configs(self, config_path):
        '''load configs for dataset loading and augmentation params '''

        with open(config_path, 'r') as f:
            content = f.read()
            content = json.loads(content, object_hook= self.hinted_tuple_hook)
        f.close()

        self.dataset_configs = content['dataset']
        self.augmentation_configs = content['augmentations']

        #to be used in matterport augmentation: if augmentation performed + datasets to augment
        self.augmentation = self.augmentation_configs['augmentation']
        self.sample_augmentation = self.augmentation_configs['sample_augmentation']

        assert len(self.dataset_configs['datasets'])==len(self.augmentation_configs['augmentation_datasets']), "Length Mismatch: lengths of dataset and augmentation_datasets don't match"
        assert len(self.dataset_configs['datasets']) == len(self.dataset_configs['dataset_weights']), "Length Mismatch: lengths of 'dataset' and 'dataset_weights'  don't match"

        self.augment_classes = { k : v for k,v in zip(self.dataset_configs['datasets'], self.augmentation_configs['augmentation_datasets'])}

        if self.augmentation:
            self.dataset_augmenter()

    def sample_dataset(self, train = True):

        '''weight_class       True                                                          False
        weight_dataset
        True               sample dataset for total*(dataset weight)*(class weight)     randomly sample each dataset for total*(dataset weight)
        False              sample total*(class weight) from entire dataset              fully random [sample dataset size]'''

        #empty list of dictionaries containing dataset
        dataset = []
        dataset_names = set(self.dataset_configs['datasets'])

        #pdb.set_trace()
        #case 1: completely random split
        if not self.dataset_configs['weighted_dataset_split'] and not self.dataset_configs['weighted_class_split']:

            if ('dogvideo' in dataset_names) or train is False:
                self.load_dogvideo_dataset(dataset, train)
            if ('cityscapes' in dataset_names) and train is True:
                self.load_cityscapes_dataset(dataset, train)

            if ('openimagesv6' in dataset_names) and train is True:
                raise not NotImplementedError()

        else:
            raise NotImplementedError()


        #sample + shuffle dataset and add to dataframe
        dataset_dataframe = pd.DataFrame(columns = ['source_dataset','image_id','image_path','annot_path'])

        random.shuffle(dataset)
        dataset  = random.sample(dataset, int(self.dataset_configs['total_samples'])) if len(dataset) > int(self.dataset_configs['total_samples']) else dataset

        for image in dataset:
            dataset_dataframe = dataset_dataframe.append(image, ignore_index = True)

        return dataset_dataframe

    def dataset_augmenter(self):
        '''collects augmentations listed in configs + creates sequential augmenter'''

        augmentation_list = []

        #sample subset of augmentations if needed
        num_augmentations = int(len(self.augmentation_configs['augmentation_list'])*random.random()) if self.sample_augmentation else len(self.augmentation_configs['augmentation_list'])


        #generate sequential augmentation function
        for aug in random.sample(self.augmentation_configs['augmentation_list'], num_augmentations):

            assert (aug in self.augmentation_configs['augmentation_params']) , "Augmentation Not Found: {} not in augmentation_params of dataset_configs.json".format(aug)

            augmentation = self.get_imgaug_function(aug)
            augmentation_params = self.augmentation_configs['augmentation_params'][aug]['args']

            for param in augmentation_params.keys():

                augmentation_params[param] = random.sample(augmentation_params[param], 1)[0] if type(augmentation_params) is not tuple else augmentation_params[param]

            augmentation_list.append(augmentation(**augmentation_params))


        #shuffle order of augmentations
        random.shuffle(augmentation_list)
        self.seq_augmenter = iaa.Sequential(augmentation_list)

    def hinted_tuple_hook(self, obj):
        if '__tuple__' in obj:
            return tuple(obj['values'])
        else:
            return obj

# test = DatasetGenerator()


class TrainingConfig(Config):

    NAME = "dog_fixations_config"

    #batch size = GPU_COUNT*IMAGES_PER_GPU
    GPU_COUNT = 3; IMAGES_PER_GPU = 3

    #number of custom classes (35 total new classes + background )
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

    NUM_CLASSES = 33 + 1

    #training steps per epoch i.e. before weights update
    STEPS_PER_EPOCH = 1000

    #setting learning rate
    LEARNING_RATE = 8e-4

    #setting weight decay [l2 regularization]
    WEIGHT_DECAY = 6e-4

    #Non-max suppression threshold i.e. removes all RPN proposals with IoU > 85% (increased to increase no of proposals during training)
    RPN_NMS_THRESHOLD = 0.85

    #setting minimum confidence prediction to 80% i.e. ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.85

    #percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.3

    #maximum number of GT boxes (objects) in 1 image
    MAX_GT_INSTANCES = 300

    #loss weights for more precise optimization.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.1,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.15,
        "mrcnn_mask_loss": 1.15
    }

    VALIDATION_STEPS = 100

class CustomDataset(Dataset):

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

    def load_all_classes(self):

        #iterate over class list and add classes
        class_list = ["person","hand","legs","foot","head","eye","dog","bird","cat","squirrel",\
                    "plant_horizontal","plant_vertical","sky","building","window","door", "bench_chair","sculpture","construction",\
                    "pavement","road","pole","sign","fire_hydrant","fence",\
                    "bicycle","scooter","car","bus","motorcycle",\
                    "ball","dog_toy","frisbee"]


        for i in range(1, len(class_list)+1):
            self.add_class("dataset",i,class_list[i-1])

    def load_dataset_images(self, train):

        #create custom dataset generator (across datasets) + sample dataset with specified configs
        generator = DatasetGenerator()
        image_data = generator.sample_dataset(train)
        #pdb.set_trace()
        for image in image_data.iterrows():
            image = image[1]
            self.add_image(source = image['source_dataset'], image_id = image['image_id'], path = image['image_path'], annotation = image['annot_path'])


    def load_mask(self, image_id):


        #extracting information of image from image_id
        info = self.image_info[image_id]

        source_name = info['source']
        annotation_info = info['annotation']

        #return mask using appropriate mask loader function

        if source_name == 'dogvideo':

            masks, class_ids = self.get_masks_dogvideo(annotation_info)

        elif source_name == 'cityscapes':

            masks, class_ids = self.get_masks_cityscapes(annotation_info)

        elif source_name == 'openimagesv6':

            masks, class_ids = self.get_masks_openimagesv6()
            raise NotImplementedError()


        return masks, np.asarray(class_ids, dtype = np.int32)


    def get_masks_dogvideo(self, file_info):
        '''support function to extract dog video masks'''

        dogvideo_mask_helper = DogVideoMaskHelper()
        img_size = (file_info[0]['original_height'], file_info[0]['original_width']); num_obj = len(file_info)



        masks = np.zeros((img_size[0], img_size[1], num_obj))
        class_ids = []

        #iterate through annotations for 1 image
        for i,annotation in enumerate(file_info):



            #extract each annotation mask
            polygon_coord = np.array(annotation['value']['points'])
            polygon_coord = dogvideo_mask_helper.convert_polygon_percentage(polygon_coord, img_size)


            annot_mask = dogvideo_mask_helper.get_polygon_mask(polygon_coord.astype(np.int32), img_size)

            #extract class name for annotation
            class_name = annotation['value']['polygonlabels'][0].lower()


            masks[:,:,i] = annot_mask
            class_ids.append(self.class_names.index(class_name))

        return masks, class_ids



    def get_masks_cityscapes(self, mask_path):
        '''support function to extract cityscapes masks'''

        cityscapes_mask_helper = CityscapeMaskHelper()

        json_file = open(annotation_filename)
        json_file = json.load(json_file)

        height = json_file['imgHeight']
        width = json_file['imgWidth']

        num_obj = 0 #count no. relevant objects in image
        class_ids = []
        relevant_object_indexes = []

        for i,obj in enumerate(json_file['objects']):

            label = obj['label']

            #converting label name to default set
            if label in class_conversion_cityscapes:
                label = class_conversion_cityscapes[label]

            #check if label is being used
            if label in relevant_classes_cityscapes:

                class_ids.append(self.class_names.index(label))
                relevant_object_indexes.append(i)
                num_obj +=1


        masks = np.zeros((height,width,num_obj), dtype='uint8') #boolean matrix to store masks of relevant objects

        for i,idx in enumerate(relevant_object_indexes):
            obj = json_file['objects'][idx]

            polygon_coord = np.array(obj['polygon'])
            polygon_coord = np.reshape(polygon_coord, (len(polygon_coord),2)) #list of [x,y] or [height,width] coordinates forming border polygon


            #use external function to contruct boolean mask (for the given object) + assign particular mask in collection of 'masks' to the objects
            boolean_mask = cityscapes_mask_helper.get_polygon_mask(polygon_coord,height,width)
            masks[:,:,i] = boolean_mask


        return masks, class_ids


    def get_masks_openimagesv6(self, mask_path):
        '''support function to extract OIV6 masks'''

        raise NotImplementedError()





    def image_reference(self,image_id):

        info = self.image_info[image_id] #accessing particular image's information (via image_id)

        return info['path'] #returning path to raw image




def train():
    pdb.set_trace()
    #constants e.g. logdir and weights dir
    LOGDIR = os.path.relpath('./logs')
    DATE_TIME_STRING = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    EXPERIMENT_NO = 10
    METRICS_DIR = os.path.join(LOGDIR, 'experiment_'+str(EXPERIMENT_NO),'metrics', DATE_TIME_STRING)

    if not os.path.isdir(METRICS_DIR):
        os.mkdir(METRICS_DIR)

    WEIGHTS_DIR = os.path.relpath("../samples/coco/mask_rcnn_coco_weights.h5")
    WEIGHTS_DIR = os.path.relpath("./logs/experiment_8/weights/epoch_3_20220420-034124.h5")
    NUM_EPOCHS = 20
    EXPERIMENT_NO = 10
    LAYERS = 'heads'
    INCREMENTAL = False

    #setup training config
    training_config = TrainingConfig()
    training_config.display()


    #load training dataset + training classes
    print('Creating training dataset ....')
    training_dataset = CustomDataset()
    training_dataset.load_all_classes()
    training_dataset.load_dataset_images(train = True)


    print('Preparing training dataset...')
    training_dataset.prepare()
    print('Loaded {} images'.format(len(training_dataset.image_ids)))


    #load validation dataset
    print('Creating validation dataset....')
    validation_dataset = CustomDataset()
    validation_dataset.load_all_classes()
    validation_dataset.load_dataset_images(train = False)

    print('Preparing validation dataset....')
    validation_dataset.prepare()
    print('Loaded {} images'.format(len(validation_dataset.image_ids)))


    #load Mask RCNN model for training
    pdb.set_trace()
    model = MaskRCNN_model.MaskRCNN(mode = 'training', config = training_config, model_dir = LOGDIR)

    print('Loading pretrained weights ...')
    model.load_weights(WEIGHTS_DIR, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])



    print('Starting training ....')
    pdb.set_trace()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(METRICS_DIR, update_freq=50) #tensorflow callback to log the losses every 'update_freq' batches
    augmentation_generator = DatasetGenerator().seq_augmenter

    if INCREMENTAL:
        for e in range(1, 6):

            print('Training sequence {}'.format(e))

            model.train(training_dataset, validation_dataset, learning_rate = training_config.LEARNING_RATE, epochs = NUM_EPOCHS//5, layers = LAYERS, custom_callbacks = [tensorboard_callback], augmentation = augmentation_generator, no_augmentation_sources = ['dogvideo'])

            print('Saving model weights....')
            weights_folder = os.path.join(LOGDIR, 'experiment_'+str(EXPERIMENT_NO),'weights')

            if not os.path.isdir(weights_folder):
                os.mkdir(weights_folder)

            weights_filename = 'epoch_'+str(e)+'_'+DATE_TIME_STRING+'.h5'
            weights_file = os.path.join(weights_folder,weights_filename)

            model.keras_model.save_weights(weights_file)

            print('Loading new model weights...')
            WEIGHTS_DIR = weights_file
            model.load_weights(WEIGHTS_DIR, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

    else:
        print('Training sequence')
        model.train(training_dataset, validation_dataset, learning_rate = training_config.LEARNING_RATE, epochs = NUM_EPOCHS, layers = LAYERS, custom_callbacks = [tensorboard_callback], augmentation = augmentation_generator, no_augmentation_sources = ['dogvideo'])

        weights_folder = os.path.join(LOGDIR, 'experiment_'+str(EXPERIMENT_NO),'weights')

        if not os.path.isdir(weights_folder):
            os.mkdir(weights_folder)

        weights_filename = 'epoch_'+str(NUM_EPOCHS)+'_'+DATE_TIME_STRING+'.h5'
        weights_file = os.path.join(weights_folder,weights_filename)

        model.keras_model.save_weights(weights_file)


    print('Finished training!')

train()
