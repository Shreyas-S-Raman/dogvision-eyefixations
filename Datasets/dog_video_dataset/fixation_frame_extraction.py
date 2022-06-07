#importing necessary libraries

'''
Instructions:


    Ensure to have the dog video (mp4) filename replaced in the format {dog name}.mp4 e.g. daisy.mp4
    Ensure to have the dog video (mp4) file stored in the same directory as this python file
    Ensure to store the fixation csv file in the same directoy as this python file
    Ensure to have the fixation csv filename in the format {dog name}.csv e.g. daisy.csv
    Add the new dog name to the set of set of names in 'dict' below
    To run this script, call python3 fixation_frame_extraction.py --dog_name {name} on the terminal (ensure to have the libraries below installed in a virtual environment)
'''

import pandas as pd
from matplotlib import pyplot as plt
import cv2
import os
import pdb
import numpy as np

import argparse

#creating path to the datasets
BASE_PATH = os.path.relpath('.')
VIDEO_PATH = os.path.relpath('.')

#names of all dog videos
dict = set(['goose', 'suna', 'kermit', 'daisy'])
frame_rate = 30


def extract_fixation_frames(video_path=None, save_base_path=None):

    if save_base_path==None:
        raise Exception('save path not defined')
    if video_path==None:
        raise Exception('video path not defined')

    #important filepaths
    BASE_PATH = save_base_path
    VIDEO_PATH = video_path

    #creating args parser
    parser = argparse.ArgumentParser(description='flags for visualizations', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dog_name', required=True, default=None)

    args = parser.parse_args()

    #get the name of the dog for which data is to be extracted and plotted
    dog_name = args.dog_name

    if not isinstance(dog_name, str):
        raise Exception('input dog_name is not a string')


    dog_name = dog_name.lower()

    if dog_name not in dict:
        raise Exception('dog name {} not defined in dictionary'.format(dog_name))

    #create path to video frames if it doesn't exist already
    #create folder to store graph output
    if not os.path.isdir(os.path.join(BASE_PATH, dog_name)):
        os.mkdir(os.path.join(BASE_PATH, dog_name))
    if not os.path.isdir(os.path.join(BASE_PATH, dog_name,'fixation_frames')):
        os.mkdir(os.path.join(BASE_PATH, dog_name, 'fixation_frames'))

    SAVE_DIR = os.path.join(BASE_PATH, dog_name, 'fixation_frames')



    #read the fixation data in from CSV file
    fixation_data = pd.read_csv(os.path.join(BASE_PATH, dog_name+'.csv'), encoding='utf-8')
    fixation_frames = fixation_data['FrameNumber'].values

    #track the current frame + index of target frame
    curr_frame = 0
    target_frame = fixation_frames[0]
    target_idx = 0


    #open the video on CV2 to extract frames
    VIDEO_PATH = os.path.join(VIDEO_PATH, dog_name+'.mp4')
    vid = cv2.VideoCapture(VIDEO_PATH)


    while True:

        #reading the frame
        ret, frame = vid.read()

        #convert data type of the frame

        curr_frame+=1

        if ret:

            #if more frames exist, continue saving to image at target_frame
            if curr_frame==int(target_frame):

                print("Saving frame {}: {} of {} frames".format(curr_frame, target_idx+1 , len(fixation_data)))

                #save frame as image
                filename = os.path.join(SAVE_DIR, str(curr_frame)+'.jpg')
                cv2.imwrite(filename,frame)

                target_idx +=1
                target_frame = fixation_frames[target_idx]


                if target_idx >= len(fixation_frames):
                    break

        #terminate when all frames are read from video
        else:
            break


    #release and remove all resources
    vid.release()
    cv2.destroyAllWindows()


extract_fixation_frames(VIDEO_PATH,BASE_PATH)
