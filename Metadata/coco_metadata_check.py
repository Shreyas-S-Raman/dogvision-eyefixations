import json
import os 

'''COCO Classes: class name used in custom dataset (COCO classname):
'''

ROOT_DIR = '/home/ssunda11/Dog_Fixations/Datasets/COCO/annotations/annotations/'
DATASET = 'train'
JSON_ANNOT_FILE = os.path.join(ROOT_DIR, 'stuff_'+DATASET+'2017.json')

METADATA_SAVE_DIR = '/home/ssunda11/Dog_Fixations/Metadata'
METADATA_SAVE_DIR = os.path.join(METADATA_SAVE_DIR,'coco_2017_'+DATASET+'_metadata.json')

relevant_classes = ['bear', 'branch','building-other','bush', 'fence', 'flower', 'grass', 'house', 'pavement', 'plant-other', 'road', 'sky-other',\
                    'skyscraper', 'tree', 'wall-brick', 'wall-concrete', 'wall-other', 'wall-stone'  ]

class_count = {}
class_count['total_image_count'] = 0
class_count['total_annotation_count'] = 0
class_count['maximum_instances'] =  0
class_count['total_counts'] = {}

class_name_dict = {}

#set of image ids 
image_set = set([])

#image instances counts
image_instances = {}

with open(JSON_ANNOT_FILE,'r') as json_file:
    json_data = json.load(json_file)

    for category in json_data["categories"]:
        
        class_name_dict[category["id"]] = category["name"]
    
    for class_name in class_name_dict.values():
        if class_name in relevant_classes:
            class_count['total_counts'][class_name] = 0
json_file.close()


with open(JSON_ANNOT_FILE, 'r') as json_file:

    json_data = json.load(json_file)

    class_count['total_annotation_count'] = len(json_data['annotations'])

    for i, annotation in enumerate(json_data["annotations"]):

        print("Processing annotation ", i, " of ", class_count["total_annotation_count"], " annotations")

        class_id = annotation['category_id'] #extract the class id from the json file

        class_name = class_name_dict[class_id] #extract the class name for the particular annotation
        
        if class_name in relevant_classes: #only if we want the particular class_name

            #increment its count
            class_count['total_counts'][class_name]+=1

            #add the image to the image set if not already present
            if annotation['image_id'] not in image_set:
                image_set.add(annotation['image_id'])
                image_instances[annotation['image_id']] = 1 #there is only one instance in the image if we see image for 1st time
            
            else: #if we have seen the image before, increment instance count
                image_instances[annotation['image_id']] += 1

    
    #updating the total image count based on the overall max set of image id's gathered
    class_count['total_image_count'] = len(image_set)
    print("Finding Maximum Instances..")
    max_instances = 0
    for image_id in image_instances.keys():

        if image_instances[image_id] > max_instances:
            max_instances = image_instances[image_id]
    
    class_count['maximum_instances'] = max_instances

json_file.close()

print("Saving File...")
with open(METADATA_SAVE_DIR, 'w') as save_path:
    json.dump(class_count, save_path)
save_path.close()

