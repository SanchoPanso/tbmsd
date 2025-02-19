import matplotlib.pyplot as plt

map = [0.689, 0.562,  0.472]
tick_label = ['small', 'medium', 'large']
plt.bar([1, 2, 3], map, width=0.8, tick_label=tick_label)
plt.xlabel('Size')
plt.ylabel('mAP@0.5')
plt.ylim([0, 1])
plt.grid()
plt.savefig('map-size.jpg')


mar = [0.828, 0.669,  0.526]
tick_label = ['small', 'medium', 'large']
plt.cla()
plt.bar([1, 2, 3], mar, width=0.8, tick_label=tick_label)
plt.xlabel('Size')
plt.ylabel('mAR@0.5')
plt.ylim([0, 1])
plt.grid()
plt.savefig('mar-size.jpg')