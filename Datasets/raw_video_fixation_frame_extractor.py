import cv2
import os
from numpy import random
import pandas as pd
import chardet

#Reading the raw dog video from (mp4) file
RAW_VIDEO_PATH = "/home/ssunda11/Dog_Fixations/Final_Output/Raw_Videos"
FIXATION_FRAME_SAVE_PATH = "/home/ssunda11/Dog_Fixations/Final_Output/Fixation_Video_Frames"
FIXATION_DATASET_PATH = "/home/ssunda11/Dog_Fixations/Final_Output/Fixation_Video_Frames"


'''Note: need to iterate all frames in the video and extract only those that are desired'''

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
video_name = video_name.split('_')[:4] #final video name filename
video_name = '_'.join(video_name)

 

'''PART 2: Extracting the necessary frames numbers where fixations occur'''

fixation_dataset_file = os.path.join(FIXATION_DATASET_PATH,video_name,video_name+'.csv')

try:
    pd.read_csv(fixation_dataset_file, encoding='utf-8')

except:
    fixation_dataframe = pd.read_csv(fixation_dataset_file, encoding='utf-16')
    fixation_dataframe.to_csv(fixation_dataset_file, encoding='utf-8')



fixation_dataframe  = pd.read_csv(fixation_dataset_file, encoding='utf-8')['FrameNumber'].to_numpy()
print(fixation_dataframe)



'''PART 2: Making all the folder structures to save frames from a particular video'''
try: #make path to video file
    if not os.path.exists(os.path.join(FIXATION_FRAME_SAVE_PATH,video_name)):
        os.makedirs(os.path.join(FIXATION_FRAME_SAVE_PATH,video_name))
except OSError:
    raise Exception("ERROR: could not create path: ",os.path.join(FIXATION_FRAME_SAVE_PATH,video_name))

try: #make path to video file/frames
    if not os.path.exists(os.path.join(FIXATION_FRAME_SAVE_PATH,video_name,'frames')):
        os.makedirs(os.path.join(FIXATION_FRAME_SAVE_PATH,video_name,'frames'))
except OSError:
    raise Exception("ERROR: could not create path: ",os.path.join(FIXATION_FRAME_SAVE_PATH,video_name,'frames'))

try: #make path to video file/frames/test
    if not os.path.exists(os.path.join(FIXATION_FRAME_SAVE_PATH,video_name,'outputs')):
        os.makedirs(os.path.join(FIXATION_FRAME_SAVE_PATH,video_name,'outputs'))
except OSError:
    raise Exception("ERROR: could not create path: ",os.path.join(FIXATION_FRAME_SAVE_PATH,video_name, 'outputs'))




''' PART 3: Iterating through frames and saving to appropriate folders '''
curr_fixation_idx = 0
curr_fixation_frame_no = fixation_dataframe[curr_fixation_idx]
curr_frame_no = 0

#folder in which to save the frames in which fixations occur
fixation_frames_save_path = os.path.join(FIXATION_FRAME_SAVE_PATH,video_name,'frames')

while True:

    success, frame = raw_video.read() #reading the frames from the mp4 file

    curr_frame_no +=1 #incrementing frame number after extracting the next frame

    if success and curr_frame_no==curr_fixation_frame_no: #if the read was successful, and the frame number is when the next fixation occurs, save the frame

        
        image_name = os.path.join(fixation_frames_save_path,str(curr_frame_no)+'.jpg') #creating filename for video frame image

        print("Creating file ",str(curr_frame_no),": "+ image_name+ "\n")

        cv2.imwrite(image_name, frame) #writing frame data to image file

        print("Created file! \n\n")

        curr_fixation_idx+=1

        if curr_fixation_idx>=len(fixation_dataframe):
            print("COMPLETED: finished processing video ", video_name)
            break


        curr_fixation_frame_no = fixation_dataframe[curr_fixation_idx]
        #updating frame counter to track number of frames with fixations that have been saved
        #updating the fixation frame number to track + look for the next frame with a fixation
        print("Processed fixation ", curr_fixation_idx, " of ", len(fixation_dataframe))

   

    elif not success: #if there is an error in extracting the video frame
        print("ERROR: could not load frame ", curr_frame_no)
        break

    

raw_video.release()
cv2.destroyAllWindows()