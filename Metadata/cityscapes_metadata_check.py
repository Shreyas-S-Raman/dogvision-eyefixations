import os
import json
import pandas as pd

'''CityScapes Classes: class name used in custom dataset (Cityscapes classname):
1. car, 2. truck, 3. bus, 4. bicycle, 5. motorcycle,
6. person, 7. rider (rider -person on a bike), 8. tree (vegetation), 9. grass (terrain), 10. sky (sky),
11. road (road), 12. sidewalk (sidewalk/pavement), 13. building (building), 14. pole (pole), 15. wall (wall)'''

#ROOT_DIR = '/home/ssunda11/Dog_Fixations/Datasets/gtFine/train'
#ROOT_DIR = '/home/ssunda11/Dog_Fixations/Datasets/gtFine/test'
ROOT_DIR = '/home/ssunda11/Dog_Fixations/Datasets/gtFine/val'
METADATA_SAVE_DIR = '/home/ssunda11/Dog_Fixations/Metadata'
city_names = os.listdir(ROOT_DIR)

class_count = {}
class_count['city_count'] = len(city_names)
class_count['total_image_count'] = 0
class_count['maximum_instances'] =  0
class_count['total_counts'] = {}

#tracking images with 'terrain' and 'vegetation' class 
terrain_images = []
vegetation_images = []
relevant_images = set([]) #images with the particular classes required 
relevant_classes = ['car','truck','bus','bicycle','motorcycle','person','road','sidewalk','sky','terrain','building','vegetation','pole','wall','rider']

relevant_images_dataframe = pd.DataFrame(columns=['annotation_path','label'])

for city in sorted(city_names):
    print("Processing city...", city, "\n")
    city_path = sorted(os.listdir(os.path.join(ROOT_DIR,city)))
    
    class_count[city] = {'image_count':len(city_path)}
    class_count['total_image_count'] += len(city_path)//4

    for i in range(3,len(city_path),4):
        
        json_file_path = os.path.join(ROOT_DIR,city,city_path[i])
        
        json_file = open(json_file_path)
        json_file = json.load(json_file)    

        if len(json_file['objects'])>class_count['maximum_instances']:
            class_count['maximum_instances'] = len(json_file['objects'])

        for obj in json_file['objects']:
            label = obj['label']

            #adding image files if they have ambiguous classes
            if label=='terrain':
                terrain_images.append(json_file_path)
            if label=='vegetation':
                vegetation_images.append(json_file_path)

             #accumulating counts of different labels by city
            if label not in class_count[city].keys():
                class_count[city][label] = 1
            else:
                class_count[city][label]+=1
            
            #accumulating counts of different labels overall
            if label not in class_count['total_counts'].keys():
                class_count['total_counts'][label] = 1
            else:
                class_count['total_counts'][label]+=1
            

            

            #adding image files if they contain any of the releveant classes
            if label in relevant_classes:
                relevant_images.add(json_file_path)


                #converting the mismatched classes for the csv file
                if label=='vegetation':
                    label= 'tree'
                elif label=='terrain':
                    label = 'grass'

                relevant_images_dataframe = relevant_images_dataframe.append({'annotation_path':json_file_path, 'label': label}, ignore_index=True)




           

print("DONE PROCESSING...SAVING FILES\n")
with open(os.path.join(METADATA_SAVE_DIR,'CityScapes_val_metadata.json'), 'w') as f:
    print(class_count['maximum_instances'])
    json.dump(class_count, f,  indent=4)

    f.close()

with open(os.path.join(METADATA_SAVE_DIR,'CityScapes_val_amb_class.txt'), "w") as textfile:
    textfile.write("TERRAIN IMAGES \n\n")
    for file_name in terrain_images:
        textfile.write(file_name + "\n")

    textfile.write("VEGETATION IMAGES \n\n")
    for file_name in vegetation_images:
        textfile.write(file_name + "\n")
    textfile.close()

with open(os.path.join(METADATA_SAVE_DIR, 'CityScapes_val_required_images.txt'),"w") as textfile:
    for file_name in relevant_images:

        textfile.write(file_name +"\n")
    textfile.close()

print(relevant_images_dataframe.head())
relevant_images_dataframe.to_csv(os.path.join(METADATA_SAVE_DIR, 'CityScapes_val_required_images.csv'), index=False)

print(len(relevant_images))
print("FINISHED!")