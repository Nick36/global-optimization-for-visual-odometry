import evaluate_ate as ate
import evaluate_rpe as rpe

if __name__=="__main__":

	ate_rmses = []
	for s in xrange(1,10):
		x = ate.rmse('gt/sequence_0' + str(s) + '.txt', 'dso/sequence_0' + str(s) + '.txt')
		#print "{} : {}".format(s, x)
		ate_rmses.append(x)

	for s in xrange(10,51):
		y = ate.rmse('gt/sequence_' + str(s) + '.txt', 'dso/sequence_' + str(s) + '.txt')
		#print "{} : {}".format(s, y)
		ate_rmses.append(y)

	with open('dso_ate.txt', 'w') as results:
		for ate_rmse in ate_rmses:
	    		results.write("{:.15f}".format(ate_rmse) + '\n')


	ate_rmses = []
	for s in xrange(1,10):
		x = ate.rmse('gt/sequence_0' + str(s) + '.txt', 'godso/sequence_0' + str(s) + '.txt')
		#print "{} : {}".format(s, x)
		ate_rmses.append(x)

	for s in xrange(10,51):
		y = ate.rmse('gt/sequence_' + str(s) + '.txt', 'godso/sequence_' + str(s) + '.txt')
		#print "{} : {}".format(s, y)
		ate_rmses.append(y)

	with open('godso_ate.txt', 'w') as results:
		for ate_rmse in ate_rmses:
	    		results.write("{:.15f}".format(ate_rmse) + '\n')


	rpe_means = []
	for s in xrange(1,10):
		x = rpe.mean_trans('gt/sequence_0' + str(s) + '.txt', 'dso/sequence_0' + str(s) + '.txt')
		#print "{} : {}".format(s, x)
		rpe_means.append(x)

	for s in xrange(10,51):
		y = rpe.mean_trans('gt/sequence_' + str(s) + '.txt', 'dso/sequence_' + str(s) + '.txt')
		#print "{} : {}".format(s, y)
		rpe_means.append(y)

	with open('dso_rpe.txt', 'w') as results:
		for rpe_mean in rpe_means:
	    		results.write("{:.15f}".format(rpe_mean) + '\n')


	rpe_means = []
	for s in xrange(1,10):
		x = rpe.mean_trans('gt/sequence_0' + str(s) + '.txt', 'godso/sequence_0' + str(s) + '.txt')
		#print "{} : {}".format(s, x)
		rpe_means.append(x)

	for s in xrange(10,51):
		y = rpe.mean_trans('gt/sequence_' + str(s) + '.txt', 'godso/sequence_' + str(s) + '.txt')
		#print "{} : {}".format(s, y)
		rpe_means.append(y)

	with open('godso_rpe.txt', 'w') as results:
		for rpe_mean in rpe_means:
	    		results.write("{:.15f}".format(rpe_mean) + '\n')



