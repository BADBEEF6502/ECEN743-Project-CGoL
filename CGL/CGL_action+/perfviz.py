import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

# Ripped from CGL on 2 MAY 2024.
# Compute the possible maximum density for still life.
# This is periodic therefore the use of special tables can be used.
# https://www.sciencedirect.com/science/article/pii/S0004370212000124?ref=pdf_download&fr=RR-2&rr=87d2acb2c840ea2a
def get_max_density(side):
    size = side ** 2
    density = 0
    table7 = [0, 0, 4, 6, 8, 16, 18, 28, 36, 43, 
            54, 64, 76, 90, 104, 119, 136, 152, 171, 190,
            210, 232, 253, 276, 302, 326, 353, 379, 407, 437, 
            467, 497, 531, 563, 598, 633, 668, 706, 744, 782, 
            824, 864, 907, 949, 993, 1039, 1085, 1132, 1181, 1229,
            1280, 1331, 1382, 1436, 1490, 1545, 1602, 1658, 1717, 1776, 1835]
    thrm6 = [0, 1, 3, 8, 9, 11, 16, 17, 19, 25, 27, 31, 33, 39, 41, 47, 49]

    if side <= 60:
        density = table7[side]
    elif side % 54 in thrm6:
        density = np.floor((size / 2) + (17 / 27) * side - 2)
    else:
        density = np.floor((size / 2) + (17 / 27) * side - 1)

    return density

def get_breakdown_sum(breakdowns):
    # Find the maximum number from all breakdowns to make a density list.
    max_val = 0
    for breakdown in breakdowns:
        max_row = np.max(breakdown[:, 0])
        if max_row > max_val:
            max_val = max_row

    # Create breakdown sum list.
    breakdown_sum = {}
    for i in range(max_val + 1):
        breakdown_sum[i] = np.uint64()

    # Start summing all breakdowns.
    for breakdown in breakdowns:
        for row in breakdown:
            breakdown_sum[row[0]] += np.uint64(row[1])

    return breakdown_sum

print('This program converts data_<side_lenght>.pkl files into human readble data plots.')
print('Usage: perfviz.py <side_length (int)>.pkl')

# Load in the data.
side = int(sys.argv[1])
max_density = get_max_density(side)
breakdowns = []
evals      = []
rewards    = []
heatmaps   = []
with open(f'data_{side}.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

    breakdowns   = loaded_data['breakdown']
    evals        = loaded_data['evals']
    rewards      = loaded_data['rewards']
    heatmaps = loaded_data['data_heatmaps']
    eval_period  = loaded_data['eval_period']

# Build the x-axis 
x_axis = np.asarray(range(0, len(rewards))) * eval_period
breakdown_sum = get_breakdown_sum(breakdowns)
density_percentage = 100 * np.asarray(evals) / len(rewards)

print(x_axis)
print(breakdown_sum)
print(density_percentage)
print(rewards)
print(heatmaps)