import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":

	values = []
	for s in xrange(1,90):
		file1 = open('orb_rpe_' + str(s) + '.txt') # 'orb_ate_'
		data1 = file1.read()
		values.append([float(i) for i in data1.split("\n") if i.strip() != ""])
	medians = np.median(values, axis=0)

	with open('orb_rpe.txt', 'w') as results: # 'orb_ate.txt'
		for m in medians:
	    		results.write("{:.15f}".format(m) + '\n')
