import os 
import pandas as pd
import json
from matplotlib import pyplot as plt
from PIL import Image
import time
import pdb
import argparse

#setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--subset',required=True,choices=['train','val','test'])

args = parser.parse_args()

#opne iamges dataset directory
ROOT_DIR = '~/fiftyone/open-images-v6'
PATH_END = args.subset #train test or validation
METADATA_SAVE_DIR = './Open_Images_V6'

class_metadata_path = os.path.join(ROOT_DIR, PATH_END,'metadata')
labels_path = os.path.join(ROOT_DIR, PATH_END, 'labels')
masks_path = os.path.join(labels_path,'masks')
data_path = os.path.join(ROOT_DIR,PATH_END,'data')




labels_dataframe = pd.read_csv(os.path.join(labels_path,'segmentations.csv'))
class_metadata_dataframe = pd.read_csv(os.path.join(class_metadata_path,'classes.csv'), header=None, names=['LabelName','Label'])


'''OpenImages V6 Classes: class name used in custom dataset (OpenImages classname):
1. bird, 2. duck, 3. hamster, 4. mouse, 5. insect, 6. lizard, 7. rabbit, 8. raccoon, 9. squirrel, 10. dog, 11. cat, 12. deer
13. bicycle, 14. bus, 15. car, 16. skateboard, 17. train, 18. truck, 19. van, 20. motorcycle
21. person, 22. human_eye (human eye), 23. human_foot (human foot), 24. human_head (human head), 25. human_hand (human hand)
26. flower, 27. house plant, 28. plant, 29. tree
30. building, 31. house, 32. building (office building), 33. building_entrance (porch), 34. skyscraper, 35. fire hydrant
36. ball, 37. dog_toy (toy)'''

relevant_open_images_v6_classes = set(['Bird','Duck','Hamster','Mouse','Insect','Lizard','Rabbit','Raccoon','Squirrel','Dog','Cat','Deer',\
                                'Bicycle','Bus','Car','Skateboard','Train', 'Truck', 'Van', 'Motorcycle',\
                                'Person','Human eye','Human foot','Human head', 'Human hand',\
                                'Flower','House plant', 'Plant','Tree','Building','House','Office building','Porch','Skyscraper','Fire hydrant',\
                                'Ball','Toy'])
error_masks = 0

#dictionary to track the class counts + extract metdata from open images dataset 
class_count = {}

class_count['total_image_count'] = len(os.listdir(data_path))
class_count['total_instance_count'] = 0
class_count['maximum_instances'] = 0
class_count['total_counts'] = {}

#dictionary to collect data and transfer to json file (for entire dataset - not divided by class)
required_json_data = {}
'''Note: keys -> image path
    values -> dictionary of: image height [imgHeight], image width [imgWidth], mask paths [mask_files] which contains a list of full paths to mask file: [mask path , label]'''
#dictionary to collect data per mask/class (divided by class) to export to csv file
required_csv_data = {'image_filename':[],'imgHeight':[],'imgWidth':[], 'mask_path':[],'label':[]}
'''Note: contain the keys mentioned above '''

#iterating through the images obtained from fiftyone extractor
image_files = os.listdir(data_path)
not_relevant_image = 0
for i, image_filename in enumerate(image_files):

    print("Processing image ", i+1, " of ", len(image_files), "...")
    
    

    image_id = image_filename.split('.')[0]  #e.g. 9d8c7cff4cb1ddf4.jpg --> 98c7cff4cb1ddf4
    label_and_mask_paths = labels_dataframe[labels_dataframe['ImageID']==image_id]

    #extract the image hegiht and width
    image_full_path = os.path.join(data_path,image_filename)
    image_pil = Image.open(image_full_path)

    img_height = image_pil.height; img_width = image_pil.width

    #appending to total instance count 
    if len(label_and_mask_paths) > class_count['maximum_instances']:
        class_count['maximum_instances'] = len(label_and_mask_paths)
    
    class_count['total_instance_count'] = class_count['total_instance_count']+len(label_and_mask_paths)

    in_relevant = False
    
    
    #extracting label and mask paths
    for [mask_filename, label_code] in label_and_mask_paths[['MaskPath','LabelName']].values:

        #decode the label code
        try:
            label_name = class_metadata_dataframe[class_metadata_dataframe['LabelName']==label_code.strip()]['Label'].values[0]
        except:
            error_masks +=1
            print("ERROR: ", label_code)
            exit()
        #label_code: /m/01yrx
        #label_name: Cat 

        if label_name in relevant_open_images_v6_classes:
            
            in_relevant = True

            if label_name=='Human eye':
                label_name = 'human_eye'
            
            elif label_name=='Human head':
                label_name = 'human_head'
            
            elif label_name=='Human foot':
                label_name = 'human_foot'
            
            elif label_name=='Human hand':
                label_name == 'human_hand'
            
            elif label_name=='Porch':
                label_name = 'building_entrance'
            
            elif label_name=='Office building':
                label_name = 'building'
            
            elif label_name=='Toy':
                label_name='dog_toy'
            
            elif label_name=='Fire hydrant':
                label_name = 'fire_hydrant'
            
            elif label_name=='House plant':
                label_name = 'plant'

            label_name = label_name.lower()

            #reformat mask_filename to full path
            first_digit = mask_filename[0] #get the sub-index  for the mask file in the masks directory

            

            mask_filename = os.path.join(masks_path,first_digit,mask_filename)

            #append values to csv data dictionary
            required_csv_data['image_filename'].append(image_filename) #e.g. 9dbc7640b4a9a4e8.jpg
            required_csv_data['mask_path'].append(mask_filename) #e.g. 0000048549557964_m0k4j_0f7c02af.png
            required_csv_data['label'].append(label_name) #e.g. Cat
            required_csv_data['imgHeight'].append(img_height) #e.g. 225
            required_csv_data['imgWidth'].append(img_width) #e.g. 116

            #append values to json data dictionary
            if image_filename not in required_json_data.keys():
                required_json_data[image_filename]={'imgHeight':img_height, 'imgWidth':img_width, 'mask_files':[[mask_filename,label_name]]}
            else:
                required_json_data[image_filename]['mask_files'].append([mask_filename,label_name])
            
            #add details about class counts 
            
            if label_name not in class_count['total_counts'].keys():
                class_count['total_counts'][label_name] = 1
            else:
                class_count['total_counts'][label_name] += 1
    
    if not in_relevant:
        not_relevant_image +=1

print("Required Images: ", len(required_json_data.keys()))
print("Non Required Images: ", not_relevant_image)

#save all the json/csv files and metadata
print("COMPLETED PROCESSING, Saving files ...")
csv_dataframe = pd.DataFrame.from_dict(required_csv_data); csv_dataframe.to_csv(os.path.join(METADATA_SAVE_DIR,'open_images_v6_'+PATH_END+'.csv'),index=False)

with open(os.path.join(METADATA_SAVE_DIR,'open_images_v6_'+PATH_END+'_metadata.json'), 'w') as f:
    print(class_count['maximum_instances'])
    json.dump(class_count, f,  indent=4)

    f.close()

with open(os.path.join(METADATA_SAVE_DIR,'open_images_v6_'+PATH_END+'.json'), 'w') as f:
    json.dump(required_json_data, f,  indent=4)

    f.close()

print("ALL FILES SAVED!")


'''
Labels: 
    segmentations: 'MaskPath' [677c122b0eaa5d16_m04yx4_9a041d52.png], 'ImageID' [677c122b0eaa5d16], 'LabelName' [/m/04yx4], 'BoxID', 'BoxXMin', 'BoxXMax','BoxYMin', 'BoxYMax', 'PredictedIoU', 'Clicks'
    NOTE: the first digit of MaskPath represents the sub-folder (in train/labels/masks) that ithe mask is within
    NOTE: LabelName indexes into classes.csv file to give label name e.g. 'dog'
    NOTE: segmentations.csv file for images with both segmentation and image-level class

    classifications: 'ImageID' [000002b66c9c498e], 'Source'[verification], 'LabelName'[/m/014jlm], 'Confidence'
    NOTE: classifications.csv file only for images without segmentation data i.e. only image-level class

Metadata:
    segmentation_classes: subset of classes (codes) with segemtnattion data [350 rows or classes]
    
    classes: all classes (codes) [601 rows or classes]'''
