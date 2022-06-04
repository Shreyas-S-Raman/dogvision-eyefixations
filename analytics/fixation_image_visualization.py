#importing necessary libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
import pdb
import random
import argparse
import scipy
from skimage.segmentation import felzenszwalb, watershed, mark_boundaries #for super-pixel segmentation
from skimage.filters import sobel_h, sobel_v
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage.color import label2rgb, rgb2gray
from colorblind import colorblind
from PIL import ImageEnhance, Image

'''
https://www.akc.org/expert-advice/health/can-dogs-see-color/
https://vcahospitals.com/know-your-pet/do-dogs-see-color#:~:text=The%20retina%20of%20the%20eye,and%20cones%2C%20which%20differentiate%20colors.&text=Dogs%20possess%20only%20two%20types,perception%20is%20called%20dichromatic%20vision.
'''

#creating path to the datasets
BASE_PATH = os.path.abspath('.')

#names of all dog videos
dict = set(['goose', 'suna', 'kermit', 'daisy'])


def visualize_fixations(base_path=None, num_samples=5):

    if base_path==None:
        raise Exception('save path not defined')


    #important filepaths
    BASE_PATH = base_path

    #creating args parser
    parser = argparse.ArgumentParser(description='flags for visualizations', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dog_name', required=True, default=None)

    args = parser.parse_args()

    #get the name of the dog for which data is to be extracted and plotted
    dog_name = args.dog_name

    if not isinstance(dog_name, str):
        raise Exception('input dog_name is not a string')

    if not isinstance(num_samples, int):
        raise Exception('input dog_name is not a string')


    dog_name = dog_name.lower()

    if dog_name not in dict:
        raise Exception('dog name {} not defined in dictionary'.format(dog_name))


    #create path to video frames if it doesn't exist already
    #create folder to store graph output
    if not os.path.isdir(os.path.join(BASE_PATH, dog_name,'samples')):
        os.mkdir(os.path.join(BASE_PATH, dog_name, 'samples'))


    #path for saving visualizations and the frames extracted from video
    SAVE_DIR = os.path.join(BASE_PATH, dog_name, 'samples')
    FRAMES_PATH = os.path.join(BASE_PATH, dog_name, 'frames')


    '''Step 1: get random sample of frames'''
    frames_list = sorted(os.listdir(FRAMES_PATH))

    frames_list = random.sample(frames_list, num_samples)

    #read the fixation data in from CSV file
    fixation_data = pd.read_csv(os.path.join(BASE_PATH, dog_name, dog_name+'.csv'), encoding='utf-8')
    fixation_data = fixation_data[['FrameNumber','PORX','PORY']].apply(np.floor).astype('int32')



    #filter the needed fixation frames
    frame_nos = list(map(lambda x: int(x.split('.')[0]), frames_list))
    boolean_filter = fixation_data.FrameNumber.isin(frame_nos)




    for frame_no,frame in zip(frame_nos,frames_list):

        print('Plotting frame {}'.format(frame_no))
        '''Step 2: display red dot for fixation coordiante'''

        full_frame_path = os.path.join(FRAMES_PATH, frame)

        #extract frame coordinates for the current sample fixation
        fixation_coords = fixation_data[fixation_data['FrameNumber']==frame_no][['PORX','PORY']]
        fixation_x_cor = fixation_coords['PORX'].values[0]; fixation_y_cor = fixation_coords['PORY'].values[0]


        #load frame image and display red dot for fixation
        frame_plot = plt.imread(full_frame_path)

        '''Figure 1: original image with fixation dot'''
        plt.imshow(frame_plot)

        #red dot for fixation point
        plt.scatter([fixation_x_cor],[fixation_y_cor], c='r', s=15)
        plt.title('original')
        plt.savefig(os.path.join(SAVE_DIR,'original_{}_{}.jpg'.format(dog_name, frame_no)))
        plt.show()
        plt.close()



        '''Figure 2: image transformed to relative perceived brightness => 0.299*R + 0.587*G + 0.114*B
           https://www.w3.org/TR/AERT/#color-contrast
        '''

        #convert red channel
        red_brightness = frame_plot[:,:,0]*0.299; green_brightness = frame_plot[:,:,1]*0.587; blue_brightness = frame_plot[:,:,2]*0.114

        #combining channles to get perceived brightness
        perceived_brightness = red_brightness + green_brightness + blue_brightness

        #normalizing perceived brightnes [0-1 scale]
        perceived_brightness = perceived_brightness/np.max(perceived_brightness)

        plt.imshow(perceived_brightness, cmap='plasma')
        plt.colorbar()
        #red dot for fixation point
        plt.scatter([fixation_x_cor],[fixation_y_cor], c='w', s=15)
        plt.title('normalized perceived brightness')

        plt.text(fixation_x_cor + 10, fixation_y_cor, str( np.round(perceived_brightness[fixation_y_cor,fixation_x_cor], decimals=2) ), c='w' )
        plt.savefig(os.path.join(SAVE_DIR,'brightness_{}_{}.jpg'.format(dog_name, frame_no)))
        plt.show()
        plt.close()

        '''Figure 3: HOG descriptor thermal map + cornerness detection => map of gradient in pixel values'''


        '''Figure 4: convolutional edge detection => gradient based on x-y sobel filters on image'''

        sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        #finding image gradient along x-axis i.e. to find vertical edges
        grad_x = fft_convolution(sobel_filter, frame_plot)

        #finding image gradient along y-axis i.e. to find horizontal edges
        grad_y = fft_convolution(np.rot90(sobel_filter, k=1), frame_plot)

        #finding the final gradient magnitudes
        grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
        grad_mag = grad_mag/np.max(grad_mag)


        plt.imshow(grad_mag, cmap='gray')
        plt.colorbar()
        #red dot for fixation point
        plt.scatter([fixation_x_cor],[fixation_y_cor], c='r', s=25)
        plt.title('pixel-gradient magnitude')
        plt.text(fixation_x_cor + 20,fixation_y_cor, str(  np.round(np.mean(grad_mag[fixation_y_cor,fixation_x_cor]) , decimals=2 )  ), c='r' )
        plt.savefig(os.path.join(SAVE_DIR,'pixel_gradient_rgb_{}_{}.jpg'.format(dog_name, frame_no)))
        plt.show()
        plt.close()

        '''Figure 4.5: conv edge detection => using sobel inbuilt on hsv'''
        @adapt_rgb(hsv_value)
        def sobel_h_hsv(image):
            return sobel_h(image)

        @adapt_rgb(hsv_value)
        def sobel_v_hsv(image):
            return sobel_v(image)


        #finding image gradient along x-axis i.e. to find vertical edges
        grad_x = sobel_v_hsv(frame_plot)

        #finding image gradient along y-axis i.e. to find horizontal edges
        grad_y = sobel_h_hsv(frame_plot)

        #finding the final gradient magnitudes
        grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
        grad_mag = grad_mag/np.max(grad_mag)

        # pdb.set_trace()

        plt.imshow(grad_mag, cmap='gray')
        plt.colorbar()
        #red dot for fixation point
        plt.scatter([fixation_x_cor],[fixation_y_cor], c='r', s=25)
        plt.title('hsv-based pixel-gradient magnitude')
        plt.text(fixation_x_cor+20,fixation_y_cor, str(  np.round(np.mean(grad_mag[fixation_y_cor,fixation_x_cor]) , decimals=2 )  ) , c='r' )
        plt.savefig(os.path.join(SAVE_DIR,'pixel_gradient_hsv_{}_{}.jpg'.format(dog_name,frame_no)))
        plt.show()
        plt.close()

        min_hysterisis_threshold = 60; max_hysterisis_threhsold = 110
        canny_edges = cv2.Canny(frame_plot, min_hysterisis_threshold, max_hysterisis_threhsold)

        plt.imshow(canny_edges, cmap='gray')

        #red dot for fixation point
        plt.scatter([fixation_x_cor],[fixation_y_cor], c='r', s=25)
        plt.title('canny edge detection')
        plt.savefig(os.path.join(SAVE_DIR,'canny_edges_{}_{}.jpg'.format(dog_name, frame_no)))
        plt.show()
        plt.close()


        '''Figure 5: super-pixel segmentation => middle or edges of objects'''

        #Part a: attempt super-pixel segmentation using felzenszwalb method
        fs_segments = felzenszwalb(frame_plot, scale=425, sigma=0.55, min_size = 1500)

        num_up, num_down, num_left, num_right = find_distance_to_edges(fs_segments, fixation_x_cor, fixation_y_cor)

        #generating random set of 5 colors to cycle through
        random_colors = gen_random_colors(len(fs_segments))
        colored_superpixels = label2rgb(fs_segments, frame_plot, colors=random_colors, alpha=0.5, bg_label = 0)

        #plt.imshow(mark_boundaries(frame_plot, fs_segments, color=(1,0,1), outline_color = (0,0,1)))
        plt.imshow(colored_superpixels)

        #red dot for fixation point
        plt.scatter([fixation_x_cor],[fixation_y_cor], c='r', s=25)
        plt.title('super-pixels: felzenszwalb method')
        plt.xlabel('up: {}, down: {}, left: {}, right: {} '.format(num_up, num_down, num_left, num_right), c = 'b' )
        plt.savefig(os.path.join(SAVE_DIR,'fekzebszwalb_superpixel_{}_{}.jpg'.format(dog_name, frame_no)))
        plt.show()
        plt.close()


        #Part b: attempt super-pixel segmentation using watershed method
        frame_plot_gray = rgb2gray(frame_plot)

        #finding image gradient along x-axis i.e. to find vertical edges
        grad_x = sobel_v_hsv(frame_plot_gray)

        #finding image gradient along y-axis i.e. to find horizontal edges
        grad_y = sobel_h_hsv(frame_plot_gray)

        #finding the final gradient magnitudes
        grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
        grad_mag = grad_mag/np.max(grad_mag)




        #markers = 17 or 18
        grad_segments = watershed(grad_mag, markers=18, compactness=1e-8)

        num_up, num_down, num_left, num_right = find_distance_to_edges(grad_segments, fixation_x_cor, fixation_y_cor)

        #generating random set of 5 colors to cycle through
        random_colors = gen_random_colors(len(grad_segments))
        colored_superpixels = label2rgb(grad_segments, frame_plot, colors=random_colors, alpha=0.5, bg_label = 0)

        #plt.imshow(mark_boundaries(frame_plot, grad_segments, color=(1,0,1), outline_color = (0,0,1), mode='thick'))
        plt.imshow(colored_superpixels)

        #red dot for fixation point
        plt.scatter([fixation_x_cor],[fixation_y_cor], c='r', s=15)
        plt.title('super-pixels: watershed method')
        plt.xlabel('up: {}, down: {}, left: {}, right: {} '.format(num_up, num_down, num_left, num_right), c = 'b' )
        plt.savefig(os.path.join(SAVE_DIR,'watershed_superpixel_{}_{}.jpg'.format(dog_name, frame_no)))
        plt.show()
        plt.close()

        '''Figure 6: dog dichromatic vision + magnitude of fixation point'''

        #converting BGR image to LAB space (to extract B-Y segment of color)
        dog_perception = cv2.imread(full_frame_path)
        dog_perception = colorblind.simulate_colorblindness(dog_perception, colorblind_type='protanopia')
        dog_perception = reduce_brightness_variation(dog_perception)



        plt.imshow(dog_perception)
        #red dot for fixation point
        plt.scatter([fixation_x_cor],[fixation_y_cor], c='r', s=15)
        plt.title('dichromatic: blue-yellow spectrum')
        plt.savefig(os.path.join(SAVE_DIR,'dogview_{}_{}.jpg'.format(dog_name,frame_no)))
        plt.show()
        plt.close()


        sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        #finding image gradient along x-axis i.e. to find vertical edges
        grad_x = fft_convolution(sobel_filter, dog_perception)

        #finding image gradient along y-axis i.e. to find horizontal edges
        grad_y = fft_convolution(np.rot90(sobel_filter, k=1), dog_perception)

        #finding the final gradient magnitudes
        grad_mag_dog = np.sqrt(np.square(grad_x) + np.square(grad_y))
        grad_mag_dog = grad_mag_dog/np.max(grad_mag_dog)



        plt.imshow(grad_mag_dog, cmap='gray')
        plt.colorbar()
        #red dot for fixation point
        plt.scatter([fixation_x_cor],[fixation_y_cor], c='r', s=25)
        plt.title('dichromatic (blue-yellow) pixel-gradient magnitude')
        plt.text(fixation_x_cor+20,fixation_y_cor, str(  np.round(np.mean(grad_mag[fixation_y_cor,fixation_x_cor]) , decimals=2 )  ) , c='r' )
        plt.savefig(os.path.join(SAVE_DIR,'pixel_gradient_dogview_{}_{}.jpg'.format(dog_name,frame_no)))
        plt.show()
        plt.close()





def reduce_brightness_variation(img, factor=0.9):

    converter = ImageEnhance.Contrast(Image.fromarray(img))
    img = converter.enhance(factor)
    img = np.asarray(img)

    return img




def find_distance_to_edges(segments, fixation_x_cor, fixation_y_cor):
    '''Finds the distance between the fixation point and the edge of the current super-pixel'''

    #track the current segmentation value
    curr_value = segments[fixation_y_cor, fixation_x_cor]


    #compare segmentation values above, below, right and left of the current position
    num_pixels_up = 0

    while fixation_y_cor-num_pixels_up-1 >= 0 and segments[fixation_y_cor-num_pixels_up-1, fixation_x_cor]==curr_value:
        num_pixels_up+=1

    num_pixels_down = 0
    while fixation_y_cor + num_pixels_down+1 < segments.shape[0] and segments[fixation_y_cor+num_pixels_down+1, fixation_x_cor]==curr_value :
        num_pixels_down+=1


    num_pixels_left = 0
    while fixation_x_cor - num_pixels_left -1 >=0 and segments[fixation_y_cor, fixation_x_cor-num_pixels_left-1]==curr_value:
        num_pixels_left+=1

    num_pixels_right = 0
    while fixation_x_cor + num_pixels_right + 1 < segments.shape[1] and segments[fixation_y_cor, fixation_x_cor+num_pixels_right+1]==curr_value:
        num_pixels_right+=1

    return num_pixels_up, num_pixels_down, num_pixels_left, num_pixels_right

def gen_random_colors(num_segments):

    return np.random.rand(num_segments,3)

def fft_convolution(kernel, image):

    '''Function to perform fast fourier transform convolution in freq. space'''
    kernel = np.rot90(kernel, k=2)
    filtered_image = np.zeros(image.shape)

    #performing 2D fourier transform function: convert from intensity domain to freq domain
    kernel_freq = np.fft.fft2(kernel, s=[image.shape[0], image.shape[1]])

    for channel in range(image.shape[-1]):

        #performing 2D fourier transform for intensities over a given channel of the image
        image_freq = np.fft.fft2(image[:,:,channel])

        convolution_output = np.abs(np.fft.ifft2(image_freq*kernel_freq))


        filtered_image[:,:,channel] = convolution_output

    return filtered_image

def extra():
    sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    #finding image gradient along x-axis i.e. to find vertical edges
    grad_x = fft_convolution(sobel_filter, frame_plot)

    #finding image gradient along y-axis i.e. to find horizontal edges
    grad_y = fft_convolution(np.flip(sobel_filter.T, axis=0), frame_plot)

    #finding the final gradient magnitudes
    grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
    grad_mag = grad_mag/np.max(grad_mag)

    #computing gradient directions
    grad_dir = np.arctan2(grad_y, grad_x)
    grad_dir = np.rad2deg(grad_dir) + 180 #convert direction to degrees in 0-360 range






visualize_fixations(BASE_PATH)
