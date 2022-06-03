import pandas as pd
import argparse
import os
import sys
sys.append('..')
import pdb
from matplotlib import pyplot as plt
import numpy as np

#create parser to access the dog name to compute distribution for
parser = argparse.ArgumentParser()
parser.add_argument('--dog_name', required = True, choices = ['daisy','suna','kermit','goose','all'])
parser.add_argument('--experiment', required = True, choices = ['1','2','3','4','5','6','7','8','9'])
parser.add_argument('--include_bg', required = True, choices = ['True','False'])
args = parser.parse_args()

args.include_bg = True if args.include_bg == 'True' else False

CLASSES_LIST = ['person', 'plant_horizontal', 'plant_vertical','building','sky','bench_chair','pavement','pole','sign','construction','bicycle','scooter','car','bus','sculpture']

if args.include_bg:
    CLASSES_LIST.append('background')

DOGNAME_LIST = ['daisy','kermit','suna','goose']

SAVE_PATH = os.path.relpath('../experiment_{}/experiment_{}/{}'.format(args.experiment, args.experiment, args.dog_name))
if not os.path.isdir(SAVE_PATH):
    os.makedir(SAVE_PATH)

#collect occurences of classes in frames + number of fixations on classes
num_occurences = {}
num_fixations = {}


#collect most overlap as fixation pred
max_overlap_count = {}


#collect overlap scores for classes
overlap_count = {}

#conditional fixation distribution
conditional_fixations = {}; conditional_fixations_count = {}

for c1 in (CLASSES_LIST[:-1] if args.include_bg else CLASSES_LIST):
    conditional_fixations[c1] = {}
    conditional_fixations_count[c1] = 0

    for c2 in CLASSES_LIST:
        conditional_fixations[c1][c2] = 0

#conditional occurence distribution
conditional_occurences = {}; conditional_occurences_count = {}

if args.dog_name != 'all':
    dataframe = pd.read_csv(os.path.join('../experiment_{}/experiment_{}/{}.csv'.format(args.experiment, args.experiment, args.dog_name) ))

else:
    dataframe = pd.read_csv(os.path.join('../experiment_{}/experiment_{}/{}.csv'.format(args.experiment, args.experiment, DOGNAME_LIST[0])))

    for d in DOGNAME_LIST[1:]:
        dataframe = dataframe.append(pd.read_csv(os.path.join('../experiment_{}/experiment_{}/{}.csv'.format(args.experiment, args.experiment, d))))




fixation_col = 'fixation_classes' if args.include_bg else 'fixation_classes_no_background'

for row in dataframe.iterrows():

    row = row[1]

    all_classes = row['all_classes'].split() if type(row['all_classes'])== str else []

    fixation_classes = row[fixation_col].split() if type(row[fixation_col])==str else []

    if type(row[fixation_col])!=str:
        print('ERROR: fixation')
    if type(row['all_classes'])!=str:
        print('ERROR: all class')


    for c in all_classes:

        if c not in num_occurences:
            num_occurences[c] = 0


        num_occurences[c] += 1


    for c in set(all_classes):
        conditional_fixations_count[c] += 1


    for i in range(0,len(fixation_classes),2):

        cl = fixation_classes[i][:-1]
        score = float(fixation_classes[i+1])

        if cl not in num_fixations:
            num_fixations[cl] = 0

        num_fixations[cl] += 1

        if cl not in overlap_count:
            overlap_count[cl] = 0

        overlap_count[cl] += score



        for c in set(all_classes):
            conditional_fixations[c][cl] += score


    max_overlap = float('-inf'); max_overlap_class = None

    for i in range(0, len(fixation_classes), 2):

        cl = fixation_classes[i][:-1]
        score = float(fixation_classes[i+1])


        if score > max_overlap:
            max_overlap = score
            max_overlap_class = cl


    if (max_overlap_class is not None):
        if max_overlap_class not in max_overlap_count:
            max_overlap_count[max_overlap_class] = 0

        max_overlap_count[max_overlap_class] += 1


max_overlap_total = sum(max_overlap_count.values())
num_fixations_total = sum(num_fixations.values())
num_occurences_total = sum(num_occurences.values())

#pdb.set_trace()

#generate distribution plots after processing
for k in CLASSES_LIST:

    overlap_count[k] = overlap_count[k]/num_fixations[k] if k in overlap_count else 0.0

    max_overlap_count[k] = (max_overlap_count[k]/max_overlap_total)*100 if k in max_overlap_count else 0.0


    num_fixations[k] = (num_fixations[k]/num_fixations_total)*100 if k in num_fixations else 0.0

    if k!='background':
        num_occurences[k] = (num_occurences[k]/num_occurences_total)*100 if k in num_occurences else 0.0


'''Creating Final Heatmap'''

conditional_heatmap = np.zeros((len(CLASSES_LIST[:-1]), len(CLASSES_LIST)))


for i, c1 in enumerate(CLASSES_LIST[:-1]):

    for j, c2 in enumerate(CLASSES_LIST):

        try:
            val = conditional_fixations[c1][c2]/conditional_fixations_count[c1] if conditional_fixations_count[c1] > 0 else 0
        except Exception as e:
            print(e)
            pdb.set_trace()

        conditional_heatmap[i][j] = conditional_fixations[c1][c2]/conditional_fixations_count[c1] if conditional_fixations_count[c1] > 0 else 0




'''Generating plots based on computed data'''

#Figure 1: conditional fixation heatmap i.e. given class A is present in the scene, probability distribution for fixations over all classes
plt.rcParams["figure.figsize"] = (10, 10)
fig, ax = plt.subplots()
im = ax.imshow(conditional_heatmap, cmap = 'coolwarm')
fig.colorbar(im, orientation='vertical')
ax.set_title('Conditional Fixations [Radius of error overlap]')
ax.set_xlabel('fixation class')
ax.set_ylabel('given/conditional class')
ax.set_xticks(np.arange(len(CLASSES_LIST)))
ax.set_xticklabels(CLASSES_LIST, rotation = 40)
ax.set_yticks(np.arange(len(CLASSES_LIST[:-1])))
ax.set_yticklabels(CLASSES_LIST[:-1])

for i in range(conditional_heatmap.shape[0]):
    for j in range(conditional_heatmap.shape[1]):
        text = ax.text(j, i, round(conditional_heatmap[i, j],2),
                       ha="center", va="center", color="w")

plt.savefig('../experiment_{}/experiment_{}/conditional_fixation_matrix_{}_{}.png'.format(args.experiment,args.experiment,args.dog_name, 'withbg' if args.include_bg else 'nobg'))
plt.show()


#Figure 2: number of occurences
plt.rcParams["figure.figsize"] = (20, 10)
keys = list(sorted(num_occurences.keys()))
values = [num_occurences[k] for k in keys]
plt.bar(keys, values, 0.5, color='b', align='edge')
plt.title('Number of occurences')
plt.xticks(rotation=40)
plt.savefig('../experiment_{}/experiment_{}/num_occurence_{}_{}.png'.format(args.experiment,args.experiment,args.dog_name, 'withbg' if args.include_bg else 'nobg'))
plt.show()

#Figure 3: total number of fixations
keys = list(sorted(num_fixations.keys()))
values = [num_fixations[k] for k in keys]
plt.bar(keys, values, 0.5, color='orange', align='edge')
plt.title('Number of fixations')
plt.xticks(rotation=40)
plt.savefig('../experiment_{}/experiment_{}/num_fixations_{}_{}.png'.format(args.experiment,args.experiment,args.dog_name, 'withbg' if args.include_bg else 'nobg'))
plt.show()

#Figure 4: average overlap with radius of error
keys = list(sorted(overlap_count.keys()))
values = [overlap_count[k] for k in keys]
plt.bar(keys, values, 0.5, color='red', align = 'edge')
plt.title('Average overlap with radius of error')
plt.xticks(rotation=40)
plt.savefig('../experiment_{}/experiment_{}/avg_overlap_{}_{}.png'.format(args.experiment,args.experiment,args.dog_name, 'withbg' if args.include_bg else 'nobg'))
plt.show()

#Figure 5: number of fixations where class (e.g. class A) had highest overlap with radius of error
keys = list(sorted(max_overlap_count.keys()))
values = [max_overlap_count[k] for k in keys]
plt.bar(keys, values, 0.5, color='purple', align='edge')
plt.title('Number of fixations with max overlap')
plt.xticks(rotation=40)
plt.savefig('../experiment_{}/experiment_{}/max_overlap_{}_{}.png'.format(args.experiment,args.experiment,args.dog_name,'withbg' if args.include_bg else 'nobg'))
plt.show()
