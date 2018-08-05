import sys
import numpy as np
import matplotlib.pyplot as plt

file1 = open('dso_rpe.txt')
data1 = file1.read()
values1 = [float(i) for i in data1.split("\n") if i.strip() != ""]

file2 = open('godso_rpe.txt')
data2 = file2.read()
values2 = [float(i) for i in data2.split("\n") if i.strip() != ""]

file3 = open('orb_rpe.txt')
data3 = file3.read()
values3 = [float(i) for i in data3.split("\n") if i.strip() != ""]

fig = plt.figure()
ax = fig.add_subplot(111)

# values from second file
# evaluate the histogram
hvalues2, hbase2 = np.histogram(values2, bins=100)
#evaluate the cumulative
cumulative2 = np.cumsum(hvalues2)
# plot the cumulative function
ax.plot(hbase2[:-1], cumulative2, c='green',label='globally optimized DSO estimates')

# values from first file
# evaluate the histogram
hvalues1, hbase1 = np.histogram(values1, bins=100)
#evaluate the cumulative
cumulative1 = np.cumsum(hvalues1)
# plot the cumulative function
ax.plot(hbase1[:-1], cumulative1, c='blue',label='DSO estimates')

# values from third file
# evaluate the histogram
hvalues3, hbase3 = np.histogram(values3, bins=100)
#evaluate the cumulative
cumulative3 = np.cumsum(hvalues3)
# plot the cumulative function
ax.plot(hbase3[:-1], cumulative3, c='orange',label='ORB estimates with disabled explicit \n loop closure detection and relocalization')

ax.legend(loc=4)
ax.set_xlabel('Relative Pose Error')
ax.set_ylabel('Number of Sequences')

ax2 = ax.twinx()
ax2.set_ylim((0, 100))
ax2.set_yticklabels(['{:3.0f}%'.format(x*2) for x in ax.get_yticks()])
ax2.set_ylabel('Percentage out of all sequences')

plt.savefig('plot.png',dpi=300)
