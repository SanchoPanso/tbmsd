# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np

classes = {
    0: 'plane',
    1: 'ship',
    2: 'storage tank',
    3: 'baseball diamond',
    4: 'tennis court',
    5: 'basketball court',
    6: 'ground track field',
    7: 'harbor',
    8: 'bridge',
    9: 'large vehicle',
    10: 'small vehicle',
    11: 'helicopter',
    12: 'roundabout',
    13: 'soccer ball field',
    14: 'swimming pool',
}

classes_list = list(classes.values())
map_values = {
    'Baseline': (18.35, 18.43, 14.98, 18.35, 18.43, 14.98, 18.35, 18.43, 14.98, 18.35, 18.43, 14.98, 18.35, 18.43, 14.98),
    'Proposed Method': (38.79, 48.83, 47.50, 38.79, 48.83, 47.50, 38.79, 48.83, 47.50, 38.79, 48.83, 47.50, 38.79, 48.83, 47.50),
}

x = np.arange(len(classes_list))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in map_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Length (mm)')
ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, classes_list)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 250)

plt.savefig('map_diff.jpg')