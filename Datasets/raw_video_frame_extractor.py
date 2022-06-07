import cv2
import os
from numpy import random

#Reading the raw dog video from (mp4) file
RAW_VIDEO_PATH = "/home/ssunda11/Dog_Fixations/Datasets/Raw_Videos/"
VIDEO_DATASET_PATH = "/home/ssunda11/Dog_Fixations/Datasets/Video_Frames_Small/"
TEST_SPLIT = 0.2

'''PART 0: Frame calculation and max number of frames extracted'''
NO_MINS = 2
START_FRAME = NO_MINS*60*30 #start collecting 2minutes after start of video

MAX_FRAMES = 7000 #maximum number of frames to extract across train and test
#Note: at min (at 30fps) can accomodate 234 second video i.e. 4 min video

SAMPLING_RATE = 30 #sampling every 5 frames i.e. 6 frames in a second than 30

'''PART 1: Setting up video and user input'''
decision = False

while decision==False:

    access_choice = input("Choose by filename or index? Type 'name' or 'index': ").strip()

    if access_choice=='name':
        video_name = input("Enter Video Filename: ").strip() #getting user input to process particular raw video
        decision = True

    elif access_choice=='index':
        print("Video Directory: \n\n")
        for vid in os.listdir(RAW_VIDEO_PATH):
            print(vid)
        video_name = int(input("Enter Video Index: "))
        video_name = os.listdir(RAW_VIDEO_PATH)[video_name]
        decision = True

    else:
        print("ERROR: Type 'name' or 'index' specifically \n")
        pass



raw_video = cv2.VideoCapture(os.path.join(RAW_VIDEO_PATH,video_name))
video_name = video_name.split('.')[0] #removing the '.mp4' from the filename for easy indexing
frame_no_start = 1 #tracks when the script should start storing/saving frames from the video i.e. after 2 min


'''PART 2: Making all the folder structures to save frames from a particular video'''
try: #make path to video file
    if not os.path.exists(os.path.join(VIDEO_DATASET_PATH,video_name)):
        os.makedirs(os.path.join(VIDEO_DATASET_PATH,video_name))
except OSError:
    raise Exception("ERROR: could not create path: ",os.path.join(VIDEO_DATASET_PATH,video_name))

try: #make path to video file/frames
    if not os.path.exists(os.path.join(VIDEO_DATASET_PATH,video_name,'frames')):
        os.makedirs(os.path.join(VIDEO_DATASET_PATH,video_name,'frames'))
except OSError:
    raise Exception("ERROR: could not create path: ",os.path.join(VIDEO_DATASET_PATH,video_name,'frames'))

try: #make path to video file/frames/test
    if not os.path.exists(os.path.join(VIDEO_DATASET_PATH,video_name,'frames','test')):
        os.makedirs(os.path.join(VIDEO_DATASET_PATH,video_name,'frames','test'))
except OSError:
    raise Exception("ERROR: could not create path: ",os.path.join(VIDEO_DATASET_PATH,video_name,'frames','test'))

try: #make path to video file/frames/train
    if not os.path.exists(os.path.join(VIDEO_DATASET_PATH,video_name,'frames','train')):
        os.makedirs(os.path.join(VIDEO_DATASET_PATH,video_name,'frames','train'))
except OSError:
    raise Exception("ERROR: could not create path: ",os.path.join(VIDEO_DATASET_PATH,video_name,'frames','train'))


''' PART 3: Iterating through frames and saving to appropriate folders '''

frame_no = len(os.listdir(os.path.join(VIDEO_DATASET_PATH,video_name,'frames','train'))) + len(os.listdir(os.path.join(VIDEO_DATASET_PATH,video_name,'frames','test')))
selected_frame_no = frame_no

while True:

    success, frame = raw_video.read() #reading the frames from the mp4 file

    if frame_no_start>=START_FRAME: #start collecting/storing frames after first 2 minutes of video

        if success and selected_frame_no<=MAX_FRAMES: #if the read was successful then safe and the video has less than max number of frames, save the frame image

            if frame_no%SAMPLING_RATE!=0:
                frame_no +=1
                continue

            if random.rand()<=TEST_SPLIT: #save into test set
                train_test_path = os.path.join(VIDEO_DATASET_PATH,video_name,'frames','test')



            else: #save into training set
                train_test_path = os.path.join(VIDEO_DATASET_PATH,video_name,'frames','train')



            image_name = os.path.join(train_test_path,str(selected_frame_no)+'.jpg') #creating filename for video frame image

            print("Creating file ",str(selected_frame_no),": "+ image_name+ "\n")

            cv2.imwrite(image_name, frame) #writing frame data to image file

            print("Created file! \n\n")

            frame_no +=1
            selected_frame_no +=1
            #updating frame counter to track number of frames

        else:
            print("COMPLETED: finished processing video ", video_name)
            break

    frame_no_start +=1 #updating frame counter to track when to start collecting frames

raw_video.release()
cv2.destroyAllWindows()