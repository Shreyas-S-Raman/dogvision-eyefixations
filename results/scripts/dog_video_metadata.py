import json
import os
import pdb


#step 1: find correct path  + iterate through json files

ANNOTATION_PATH = os.path.relpath('../../Datasets/dog_video_dataset/annotations.json')

def check_class_names(joint_filename):

    class_counts = {}


    with open(joint_filename, 'r') as f:
        content = json.loads(f.read())

    for filename in content.keys():

        image_data = content[filename]


        for annot in image_data['annotations']:
            for result in annot['result']:

                label = result['value']['polygonlabels'][0]

                if label not in class_counts:
                    class_counts[label] = 0

                class_counts[label] += 1

    print("Overall class counts: \n", class_counts)
    print("Total Count: ", sum(class_counts.values()))



check_class_names(ANNOTATION_PATH)
