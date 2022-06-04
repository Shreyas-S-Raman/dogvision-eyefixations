#importing necessary libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import sklearn
import os
import pdb

import argparse

#creating path to the datasets
BASE_PATH = os.path.abspath('.')

#video durations in minutes
dict = {'goose':20.5, 'suna':16, 'kermit': 17.5, 'daisy': 9}
frame_rate = 30

'''Note: dog video captured at 30fps: every 0.5 minutes is 900 frames'''

def run_visualizer(path=None, box_whisker = False):

    if path==None:
        raise Exception('base dataset file path not defined')

    BASE_PATH = path

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

    #create folder to store graph output
    if not os.path.isdir(os.path.join(BASE_PATH, dog_name)):
        os.mkdir(os.path.join(BASE_PATH, dog_name))

    SAVE_DIR = os.path.join(BASE_PATH, dog_name)

    #read the fixation data in from CSV file
    fixation_data = pd.read_csv(os.path.join(BASE_PATH, dog_name, dog_name+'.csv'), encoding='utf-8')


    '''Plot 1: distribution plot : % of video/walk v.s. length of fixations, datapoint color by pupil size (height*width)'''

    #extracting frame numbers as percentage of total walk: x-axis
    total_frames = dict[dog_name]*60*frame_rate
    walk_percentage = fixation_data['FrameNumber'].values/total_frames

    #extracting fixation durations: y-axis
    fixation_duration = fixation_data['Duration(ms)'].values/10e3

    small_fixation_duration = fixation_duration[np.where(fixation_duration<=0.1)]
    small_walk_percentage = walk_percentage[np.where(fixation_duration<=0.1)]

    #extracting pupil size information: approx. height*width
    #ellipse area: A = π x ((w ÷ 2) x (h ÷ 2))

    pupil_size = np.pi*(fixation_data['PupilHeight'].values/2)*(fixation_data['PupilWidth'].values/2)
    pupil_size = pupil_size/max(pupil_size) #max scaling the pupil size

    small_pupil_size = pupil_size[np.where(fixation_duration<=0.1)]

    #plotting scatter plot of fixation patterns
    # cbar = plt.colorbar()
    # cbar.ax.get_yaxis().labelpad = 15
    # cbar.ax.set_ylabel('pupil size', rotation=270)
    plt.scatter(walk_percentage, fixation_duration, s=12, c=pupil_size, cmap='inferno', alpha=0.5)
    plt.xlabel('% of walk')
    plt.ylabel('fixation duration (ms | $10^3$)')

    plt.savefig(os.path.join(SAVE_DIR, 'fixation_scatter_{}.png'.format(dog_name)))
    plt.show()
    plt.close()

    plt.scatter(small_walk_percentage, small_fixation_duration, s=12, c=small_pupil_size, cmap='inferno',alpha=0.5)
    plt.xlabel('% of walk')
    plt.ylabel('fixation duration (ms | $10^3$) - small fixations')

    plt.savefig(os.path.join(SAVE_DIR, 'small_fixation_scatter_{}.png'.format(dog_name)))
    plt.show()
    plt.close()


    '''Plot 2: bar plot: number of fixations every 30s of video'''
    fixations_per_period = []

    all_frames = fixation_data['FrameNumber'].values

    start = 0; end = start + frame_rate*30

    total_frames = dict[dog_name]*60*frame_rate

    while True:

        #extract number of total fixations between start and end
        num_fixations = np.where(np.logical_and(all_frames>=start, all_frames<end))
        num_fixations = len(all_frames[num_fixations])


        #append fixations to list
        fixations_per_period.append(num_fixations)

        #increment start and end counters
        start += frame_rate*30
        end += frame_rate*30

        if start > total_frames:
            break

    #plot the bar graph using the collected fixations data
    x_values = np.array(range(0, int(total_frames+frame_rate*30), frame_rate*30))
    x_values = (x_values/(frame_rate)).astype(np.int32)



    plt.bar(x_values, fixations_per_period, color='grey', width=8)
    plt.plot(x_values, fixations_per_period, 'r-', linewidth=0.8)
    plt.xlabel('time (seconds)')
    plt.ylabel('number of fixations')

    plt.savefig(os.path.join(SAVE_DIR, 'number_fixations_{}.png'.format(dog_name)))
    plt.show()
    plt.close()


    '''Plot 3: multiple box and whisker plots : for each dog, Q1 median Q3 of length of fixations '''
    if box_whisker:


        boxplot_data = []
        means = []
        medians = []

        for dog_name in dict.keys():
            fixation_duration = pd.read_csv(os.path.join(BASE_PATH, dog_name+'.csv'), encoding='utf-8')
            fixation_duration = fixation_duration['Duration(ms)'].values/10e3

            #extracting the periods where fixation is beyond a threshold
            long_fixation_duration = fixation_duration[np.where(fixation_duration>0.1)]

            boxplot_data.append(long_fixation_duration)
            means.append(np.round(np.mean(long_fixation_duration), decimals=2))
            medians.append(np.round(np.median(long_fixation_duration),decimals=2))


        flierprops = {'marker':'o', 'markerfacecolor':'grey', 'markersize':5,'linestyle':'none', 'alpha':0.5}


        box_plot = plt.boxplot(boxplot_data, labels=dict.keys(), whis=1.5, showmeans=True, meanline=True, flierprops=flierprops, patch_artist=True)
        pdb.set_trace()
        #color in the box plot boxes
        for bplot in box_plot['boxes']:
            bplot.set_facecolor('lightblue')

        print(medians, means)
        plt.ylabel('fixation duration (ms | $10^3$)')
        plt.xlabel('dogs')
        plt.savefig(os.path.join(BASE_PATH, 'dog_fixation_distribution.png'))
        plt.show()
        plt.close()



run_visualizer(BASE_PATH, False)
