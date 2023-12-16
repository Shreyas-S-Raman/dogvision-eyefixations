import pandas as pd
import argparse
import os
import sys
sys.path.append('..')
import pdb
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import copy
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
from scipy import stats as stats
from scipy.stats import chisquare


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    global n

    s = str(round(100 * y / n, 3))
    

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

#create parser to access the dog name to compute distribution for
parser = argparse.ArgumentParser()
parser.add_argument('--dog_name', required = True, choices = ['daisy','suna','kermit','goose','all'])
parser.add_argument('--experiment', required = True, choices = ['1','2','3','4','5','6','7','8','9'])
#parser.add_argument('--only_test', required = True, choices = ['True','False'])
args = parser.parse_args()



CLASSES_LIST = ['person', 'plant_horizontal', 'plant_vertical','building','sky','bench_chair','pavement','pole','sign','construction','bicycle','scooter','car','bus','sculpture','background']
DOGNAME_LIST = ['daisy','kermit','suna','goose']


SAVE_PATH = os.path.relpath('../experiment_{}/experiment_visualizations/{}'.format(args.experiment, args.dog_name))

if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)


#getting subset of ground truth csv file with predictions
if args.dog_name == 'all':

    ground_truth = pd.read_csv('../ground_truth/fixation_predictions/{}.csv'.format(DOGNAME_LIST[0]))

    for name in DOGNAME_LIST[1:]:
        ground_truth = ground_truth.append( pd.read_csv('../ground_truth/fixation_predictions/{}.csv'.format(name)), ignore_index = True)
else:
    ground_truth = pd.read_csv('../ground_truth/fixation_predictions/{}.csv'.format(args.dog_name))

ground_truth = ground_truth[ground_truth['union_areas'].notnull()]

#collecting + filtering subset of matching fixations from predictions
if args.dog_name == 'all':
    fixation_predictions = pd.read_csv('../experiment_{}/fixation_predictions/{}.csv'.format(args.experiment, DOGNAME_LIST[0]))

    for name in DOGNAME_LIST[1:]:
        fixation_predictions = fixation_predictions.append( pd.read_csv('../experiment_{}/fixation_predictions/{}.csv'.format(args.experiment, name)), ignore_index = True)
else:
    fixation_predictions = pd.read_csv('../experiment_{}/fixation_predictions/{}.csv'.format(args.experiment, args.experiment, args.dog_name))

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

#metric 7: confusion matrix based on max probability class
confusion_matrix = {}

for c in CLASSES_LIST:
    confusion_matrix[c] = {}
    for c1 in CLASSES_LIST:
        confusion_matrix[c][c1] = 0

for gt_row, fixation_row in zip(ground_truth.iterrows(), fixation_predictions.iterrows()):

    gt_row = gt_row[1]; fixation_row = fixation_row[1]


    #extract fixation data for difference metric
    gt_fixation = gt_row['fixation_classes_no_background'].split(' ') if type(gt_row['fixation_classes_no_background']) is str else []

    predicted_fixation = fixation_row['fixation_classes_no_background'].split(' ') if type(fixation_row['fixation_classes_no_background']) is str else []

    gt_fixations = {}; predicted_fixations = {}
    gt_pred = None; predicted_pred = None
    gt_max_score = float('-inf'); predicted_max_score = float('-inf')

    
    for i in range(0, len(gt_fixation),2):

        gt_fixations[gt_fixation[i][:-1]] = float(gt_fixation[i+1])
        
        if float(gt_fixation[i+1]) > gt_max_score:
            gt_max_score = float(gt_fixation[i+1])
            gt_pred = gt_fixation[i][:-1]

    for i in range(0, len(predicted_fixation),2):

        predicted_fixations[predicted_fixation[i][:-1]] = float(predicted_fixation[i+1])

        
        if float(predicted_fixation[i+1]) > predicted_max_score:
            predicted_max_score = float(predicted_fixation[i+1])
            predicted_pred = predicted_fixation[i][:-1]

    if gt_pred is None:
        gt_pred = 'background'
    if predicted_pred is None:
        predicted_pred = 'background'
    confusion_matrix[gt_pred][predicted_pred]+=1
    

    all_fixation_classes = set(list(gt_fixations.keys()) + list(predicted_fixations.keys()))
    total_chi_dist = 0

    for c in all_fixation_classes:

        gt_overlap = gt_fixations[c] if c in gt_fixations.keys() else 0
        pred_overlap = predicted_fixations[c] if c in predicted_fixations.keys() else 0

        total_chi_dist += ((pred_overlap-gt_overlap)**2)/(gt_overlap) if gt_overlap >0 else ((pred_overlap-gt_overlap)**2)


        fixation_area_diff[c] = pred_overlap - gt_overlap

        if c not in max_fixation_area_diff:
            max_fixation_area_diff[c] = 0


        max_fixation_area_diff[c] = pred_overlap - gt_overlap if abs(pred_overlap-gt_overlap) > max_fixation_area_diff[c] else max_fixation_area_diff[c]

        if c not in fixation_class_counts:
            fixation_class_counts[c] = 0
        fixation_class_counts[c] += 1

    #total_chi_dist += ((pred_overlap-gt_overlap)**2)/(gt_overlap) if gt_overlap >0 else ((pred_overlap-gt_overlap)**2)
    chi_square_distance.append(total_chi_dist)
    print(chi_square_distance)


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



global n
n = len(chi_square_distance)

# pdb.set_trace()
for c in fixation_area_diff.keys():
    fixation_area_diff[c] = fixation_area_diff[c]/fixation_class_counts[c]

for c in percentage_area_diff.keys():
    percentage_area_diff[c] = percentage_area_diff[c]/percentage_class_counts[c]

#plot histogram distribution of chi square distance

# bins = np.linspace(0, max(chi_square_distance), 400)
# fig, ax = plt.subplots()

# values, bins, __  = ax.hist(chi_square_distance, bins = bins)

# func_formatter =FuncFormatter(to_percent)
# plt.gca().yaxis.set_major_formatter(func_formatter)

# val_sum = sum(values)

# val_percentage = list(map(lambda x: x/val_sum, values))

# for i, v in enumerate(values):

#     if round(val_percentage[i],2) >= 0.01:
#         ax.text(bins[i]+1e-3,v+1, str(round(val_percentage[i],2)), color='blue', fontweight='bold', fontsize='x-small')

# mu, std = norm.fit(chi_square_distance)
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# plt.xlabel('chi-square distance [between predicted and ground truth distributions]')
# plt.ylabel('frequency (as  %)')
# plt.savefig(os.path.join(SAVE_PATH, 'chi_square_dist_full_{}.png'.format(args.dog_name)))
# plt.show()

#plot histogram distribution of chi square distance
fig, ax = plt.subplots()

mu, std = norm.fit(chi_square_distance)

x=np.linspace(min(chi_square_distance), max(chi_square_distance),1000)
iq=stats.norm(mu,std)
ax.plot(x,iq.pdf(x),'k', alpha=0.7)

px=np.arange(min(chi_square_distance),max(chi_square_distance),10)
ax.fill_between(px,iq.pdf(px),color='b', alpha=0.3)

ax.axvline(x = np.percentile(chi_square_distance, 25), color = 'r', linestyle='dashed', label = 'Q1')
ax.text(np.percentile(chi_square_distance, 25)+1,0.01,'Q1: $\chi^{} = {}$'.format(2, np.percentile(chi_square_distance, 25)))
ax.axvline(x = np.percentile(chi_square_distance, 50), color = 'g', linestyle='dashed', label = 'Q2')
ax.text(np.percentile(chi_square_distance, 50)+1,0.005,'Q2: $\chi^{} = {}$'.format(2, np.percentile(chi_square_distance, 50)))
ax.axvline(x = np.percentile(chi_square_distance, 75), color = 'purple', linestyle='dashed', label = 'Q3')
ax.text(np.percentile(chi_square_distance, 75)+1,0.002,'Q3: $\chi^{} = {}$'.format(2, np.percentile(chi_square_distance, 75)))

ax.axvline(x = 24.996, color = 'r', linestyle='dashed', label = 'chi-square threshold (dof: 15)')
ax.text(24.996+1, 0.001, 'threshold: $\chi^{2} = 24.996$')

plt.xlabel('chi-square ($\chi^{2}$) distance [between predicted and ground truth class distributions]')
plt.ylabel('probability density')
plt.savefig(os.path.join(SAVE_PATH, 'chi_square_dist_all_{}.png'.format(args.dog_name)))
plt.show()

# bins = np.linspace(min(chi_square_distance), max(chi_square_distance), 100)

# values, bins, __  = ax.hist(chi_square_distance, bins = bins)

# func_formatter =FuncFormatter(to_percent)
# plt.gca().yaxis.set_major_formatter(func_formatter)

# val_sum = sum(values)

# val_percentage = list(map(lambda x: x/val_sum, values))

# for i, v in enumerate(values):

#     if round(val_percentage[i],2) >= 0.01:
#         ax.text(bins[i]+1e-3,v+1, str(round(val_percentage[i],2)), color='blue', fontweight='bold', fontsize='x-small')

bins = np.linspace(0, 25, 200)
fig, (ax, ax2) = plt.subplots(1,2, sharey=True)

values, bins, __  = ax.hist(chi_square_distance, bins = bins)

func_formatter =FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(func_formatter)

val_sum = sum(values)

val_percentage = list(map(lambda x: x/val_sum, values))

for i, v in enumerate(values):

    if round(val_percentage[i],2)*100 > 0:
        ax.text(bins[i]+1e-3,v+1, str(int(round(val_percentage[i],2)*100)), color='blue', fontweight='bold', fontsize=10)

ax2.axvline(x = 24.996, color = 'r', linestyle='dashed', label = 'chi-square threshold (dof: 15)')
ax2.text(24.99-1.5, 0.001, 'threshold: $\chi^{2} = 24.996$')

ax.set_xlim(0, 4)
ax2.set_xlim(23, 26)

ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

# ax2.tick_params(labelleft=False)  # don't put tick labels at the top
# ax.yaxis.tick_right()
ax2.tick_params(axis='x', which='both',
                right=False)

d = .015
kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((-d, +d), (-d, +d), **kwargs)  
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)      # top-left diagonal
ax2.tick_params(axis='y', colors='white') 
# ax2.plot((1-d, +d), (-d, +d), **kwargs)  # top-right diagonal
kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
  # bottom-left diagonal
ax.plot((1-d, 1+d), (1 - d, 1 + d), **kwargs)    # bottom-right diagonal
ax.plot((1-d, 1+d), (- d, + d), **kwargs)

plt.xlabel('chi-square ($\chi^{2}$) distance [between predicted and ground truth class distributions]')
ax.set_ylabel('frequency (as %)')
plt.savefig(os.path.join(SAVE_PATH, 'chi_square_dist_thresh_{}.png'.format(args.dog_name)))
plt.show()

confusion_matrix_heatmap = np.zeros((len(CLASSES_LIST), len(CLASSES_LIST)))

for i,r in enumerate(confusion_matrix.keys()):
    for j, c in enumerate(confusion_matrix[r].keys()):
        confusion_matrix_heatmap[i][j] = confusion_matrix[r][c]

confusion_matrix_heatmap_vis = copy.deepcopy(confusion_matrix_heatmap)
total_counts = np.sum(confusion_matrix_heatmap, axis=1)

for i in range(len(confusion_matrix_heatmap_vis)):

    confusion_matrix_heatmap_vis[i] = confusion_matrix_heatmap_vis[i]/total_counts[i] if total_counts[i]>0 else confusion_matrix_heatmap_vis[i]



plt.rcParams["figure.figsize"] = (10, 10)
fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix_heatmap_vis, cmap = 'coolwarm')
fig.colorbar(im, orientation='vertical')
ax.set_title('Confusion Matrix')
ax.set_xlabel('predicted class')
ax.set_ylabel('ground truth class')
ax.set_xticks(np.arange(len(CLASSES_LIST)))
ax.set_xticklabels(CLASSES_LIST, rotation = 40)
ax.set_yticks(np.arange(len(CLASSES_LIST)))
ax.set_yticklabels(CLASSES_LIST)

for i in range(confusion_matrix_heatmap.shape[0]):
    for j in range(confusion_matrix_heatmap.shape[1]):
        text = ax.text(j, i, int(confusion_matrix_heatmap[i, j]),
                       ha="center", va="center", color="w")

plt.savefig(os.path.join(SAVE_PATH, 'confusion_matrix_{}_{}.png'.format(args.experiment, args.dog_name)))
plt.show()

#printing final statistical results
print('Mean Chi-Square Dist: ', np.mean(chi_square_distance))
print('Median Chi-Square: ', np.median(chi_square_distance))
print('Variance Chi-Square Dist: ', np.var(chi_square_distance))
print('% Below Critical: ', len(np.where(np.array(chi_square_distance) <= 24.996)[0])/len(chi_square_distance))
print('95th Percentile Chi-Square: ', np.percentile(chi_square_distance, 95))
print('90th Percentile Chi-Square: ', np.percentile(chi_square_distance, 90))
print('Union Area Mean: ', mean_areas)
print('Union Area Variance: ', var_areas)

print('Fixation Area Difference: ', fixation_area_diff)
print('--------------------------')
print('Percentage Area Difference: ', percentage_area_diff)
print('--------------------------')
print('Max Fixation Area Difference: ', max_fixation_area_diff)
print('--------------------------')
print('Max Percentage Area Difference: ', max_percentage_area_diff)
