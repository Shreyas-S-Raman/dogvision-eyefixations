import pandas as pd
import argparse
import os
import sys
sys.append('..')
import pdb
import numpy as np
from matplotlib import pyplot as plt


#create parser to access the dog name to compute distribution for
parser = argparse.ArgumentParser()
parser.add_argument('--dog_name', required = True, choices = ['daisy','suna','kermit','goose','all'])
parser.add_argument('--experiment', required = True, choices = ['1','2','3','4','5','6','7','8','9'])
#parser.add_argument('--only_test', required = True, choices = ['True','False'])
args = parser.parse_args()



CLASSES_LIST = ['person', 'plant_horizontal', 'plant_vertical','building','sky','bench_chair','pavement','pole','sign','construction','bicycle','scooter','car','bus','sculpture','background']
DOGNAME_LIST = ['daisy','kermit','suna','goose']


SAVE_PATH = os.path.relpath('../experiment_{}/experiment_{}/{}'.format(args.experiment, args.experiment, args.dog_name))

if not os.path.isdir(SAVE_PATH):
    os.makedir(SAVE_PATH)


#getting subset of ground truth csv file with predictions
if args.dog_name == 'all':

    ground_truth = pd.read_csv('ground_truth/ground_truth/{}.csv'.format(DOGNAME_LIST[0]))

    for name in DOGNAME_LIST[1:]:
        ground_truth = ground_truth.append( pd.read_csv('ground_truth/ground_truth/{}.csv'.format(name)), ignore_index = True)
else:
    ground_truth = pd.read_csv('../ground_truth/ground_truth/{}.csv'.format(args.dog_name))

ground_truth = ground_truth[ground_truth['union_areas'].notnull()]

#collecting + filtering subset of matching fixations from predictions
if args.dog_name == 'all':
    fixation_predictions = pd.read_csv('../experiment_{}/experiment_{}/{}.csv'.format(args.experiment, args.experiment, DOGNAME_LIST[0]))

    for name in DOGNAME_LIST[1:]:
        fixation_predictions = fixation_predictions.append( pd.read_csv('../experiment_{}/experiment_{}/{}.csv'.format(args.experiment, args.experiment, name)), ignore_index = True)
else:
    fixation_predictions = pd.read_csv('../experiment_{}/experiment_{}/{}.csv'.format(args.experiment, args.experiment, args.dog_name))

fixation_predictions = fixation_predictions.loc[ground_truth.index.values]


#metric 1: collect mean area + variation in area covered by each
mean_areas = {'ground_truth': ground_truth.mean()['union_areas'], 'predicted':fixation_predictions.mean()['union_areas']}
var_areas = {'ground_truth': ground_truth.var()['union_areas'], 'predicted':fixation_predictions.var()['union_areas']}


#metric 2: mean difference in fixation overlap for each class
fixation_area_diff = {}
fixation_class_counts = {}

#metric 3: max difference in fixation overlap for each class
max_fixation_area_diff = {}

#metric 4: mean difference in percentage area for each class
percentage_area_diff = {}
percentage_class_counts = {}

#metric 5: max difference in percentage area for each class
max_percentage_area_diff = {}

#metric 6: chi-square distance distribution
chi_square_distance = []


for gt_row, fixation_row in zip(ground_truth.iterrows(), fixation_predictions.iterrows()):

    gt_row = gt_row[1]; fixation_row = fixation_row[1]


    #extract fixation data for difference metric
    gt_fixation = gt_row['fixation_classes_no_background'].split(' ') if type(gt_row['fixation_classes_no_background']) is str else []

    predicted_fixation = fixation_row['fixation_classes_no_background'].split(' ') if type(fixation_row['fixation_classes_no_background']) is str else []

    gt_fixations = {}; predicted_fixations = {}

    for i in range(0, len(gt_fixation),2):

        gt_fixations[gt_fixation[i][:-1]] = float(gt_fixation[i+1])

    for i in range(0, len(predicted_fixation),2):

        predicted_fixations[predicted_fixation[i][:-1]] = float(predicted_fixation[i+1])


    all_fixation_classes = set(list(gt_fixations.keys()) + list(predicted_fixations.keys()))
    total_chi_dist = 0

    for c in all_fixation_classes:

        gt_overlap = gt_fixations[c] if c in gt_fixations.keys() else 0
        pred_overlap = predicted_fixations[c] if c in predicted_fixations.keys() else 0

        total_chi_dist += ((pred_overlap-gt_overlap)**2)/(pred_overlap + gt_overlap)

        fixation_area_diff[c] = pred_overlap - gt_overlap

        if c not in max_fixation_area_diff:
            max_fixation_area_diff[c] = 0


        max_fixation_area_diff[c] = pred_overlap - gt_overlap if abs(pred_overlap-gt_overlap) > max_fixation_area_diff[c] else max_fixation_area_diff[c]

        if c not in fixation_class_counts:
            fixation_class_counts[c] = 0
        fixation_class_counts[c] += 1

    total_chi_dist  = 0.5*total_chi_dist/len(all_fixation_classes) if len(all_fixation_classes)>0 else 0
    chi_square_distance.append(total_chi_dist)


    #extract percentage area for difference metric
    gt_area = gt_row['percentage_area'].split(' '); predicted_area = fixation_row['percentage_area'].split(' ')
    gt_areas = {}; predicted_areas = {}

    for i in range(0, len(gt_area)):

        [c, a] = gt_area[i].split(':')
        gt_areas[c] = float(a)

    for i in range(0, len(predicted_area)):

        [c, a] = predicted_area[i].split(':')
        predicted_areas[c] = float(a)


    all_classes = set(list(gt_areas.keys()) + list(predicted_areas.keys()))


    for c in all_classes:

        gt_a = gt_areas[c] if c in gt_areas.keys() else 0
        pred_a = predicted_areas[c] if c in predicted_areas.keys() else 0

        percentage_area_diff[c] = pred_a - gt_a

        if c not in max_percentage_area_diff:
            max_percentage_area_diff[c] = 0
        max_percentage_area_diff[c] = pred_a-gt_a if abs(pred_a-gt_a) > max_percentage_area_diff[c] else max_percentage_area_diff[c]

        if c not in percentage_class_counts:
            percentage_class_counts[c] = 0
        percentage_class_counts[c] += 1






for c in fixation_area_diff.keys():
    fixation_area_diff[c] = fixation_area_diff[c]/fixation_class_counts[c]

for c in percentage_area_diff.keys():
    percentage_area_diff[c] = percentage_area_diff[c]/percentage_class_counts[c]

#plot histogram distribution of chi square distance
bins = np.linspace(0, 0.6, 20)
fig, ax = plt.subplots()
values, bins, __  = ax.hist(chi_square_distance, bins = bins)
val_sum = sum(values)

val_percentage = list(map(lambda x: x/val_sum, values))

for i, v in enumerate(values):
    ax.text(bins[i]+1e-3,v+1, str(round(val_percentage[i],2)), color='blue', fontweight='bold', fontsize='x-small')

plt.xlabel('chi-square distance [between predicted and ground truth distributions]')
plt.ylabel('frequency')
plt.savefig('../experiment_{}/experiment_{}/chi_square_dist_{}.png'.format(args.experiment, args.experiment, args.dog_name))
plt.show()

#printing final statistical results
print('Mean Chi-Square Dist: ', np.mean(chi_square_distance))
print('Variance Chi-Square Dist: ', np.var(chi_square_distance))
print('95th Percentile Chi-Square: ', np.percentile(chi_square_distance, 95))
print('Union Area Mean: ', mean_areas)
print('Union Area Variance: ', var_areas)

print('Fixation Area Difference: ', fixation_area_diff)
print('--------------------------')
print('Percentage Area Difference: ', percentage_area_diff)
print('--------------------------')
print('Max Fixation Area Difference: ', max_fixation_area_diff)
print('--------------------------')
print('Max Percentage Area Difference: ', max_percentage_area_diff)
