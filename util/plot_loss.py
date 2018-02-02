#!/usr/bin/python
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
import scipy.signal as sig



def parse_line(line):
	# (epoch: 1, iters: 2, time: 1.453) D_A_real: -1.416 D_A_fake: -0.580 
	# D_B_real: 1.567 D_B_fake: 2.359 G_A: 0.000 G_B: 0.000 Cyc_A: 0.000 Cyc_B: 0.000 featL: 0.000
	elmts = line.split()

	(it, da_real, da_fake, ga, cyca, db_real, db_fake, gb, cycb, feat) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

	if len(elmts) < 18:
		return (it, da_real, da_fake, ga, cyca, db_real, db_fake, gb, cycb, feat)
	elif len(elmts) < 22: # no cycle loss
		it = elmts[3]
		da_real = elmts[7]
		da_fake = elmts[9]
		db_real = elmts[11]
		db_fake = elmts[13]
		ga = elmts[15]
		gb = elmts[17]
	elif (len(elmts) < 24): # no featL
		cyca = elmts[13]
		cycb = elmts[21]
	else: # all values are present
		feat = elmts[23]
	
	return (it, da_real, da_fake, ga, cyca, db_real, db_fake, gb, cycb, feat)


def main():
	# open file and read it line by line
	parser = argparse.ArgumentParser()
	parser.add_argument("path")
	parser.add_argument("--median", action="store_true")
	parser.add_argument("--hideDisc", action="store_true")
	parser.add_argument("--hideGen", action="store_true")
	parser.add_argument("--hideCyc", action="store_true")
	parser.add_argument("--hideFeat", action="store_true")
	parser.add_argument("--save", action="store_true")


	args = parser.parse_args()



	filepath = args.path
	f = open(filepath, 'r')

	da_reals, da_fakes, gas, cycas, db_reals, db_fakes, gbs, cycbs, feats = [], [], [], [], [], [], [], [], []
	da_diff, db_diff = [], []


	for line in f:
		it, da_real, da_fake, ga, cyca, db_real, db_fake, gb, cycb, feat = parse_line(line)
		if it < 0:
			continue

		if not args.hideDisc:
			da_reals.append(da_real)
			da_fakes.append(da_fake)
			db_reals.append(db_real)
			db_fakes.append(db_fake)
		
		if not args.hideGen:
			gbs.append(gb)
			gas.append(ga)
		
		if not args.hideCyc:
			cycas.append(cyca)	
			cycbs.append(cycb)
		
		if not args.hideFeat:
			feats.append(feat)

	print("Parsing completed.")
	plt.figure()

	


#med_filtered_loss = scipy.signal.medfilt((-Loss_D, dtype='float64'), 101)

	if not args.hideDisc:
		if args.median:
			# import pdb; pdb.set_trace()
			plt.plot(sig.medfilt(np.asarray(da_reals, dtype='float64'), 101), linewidth=2.0)
			plt.plot(sig.medfilt(np.asarray(da_fakes, dtype='float64'), 101), linewidth=2.0)
			plt.plot(sig.medfilt(np.asarray(db_reals, dtype='float64'), 101), linewidth=2.0)
			plt.plot(sig.medfilt(np.asarray(db_fakes, dtype='float64'), 101), linewidth=2.0)
		else:
			plt.plot(np.asarray(da_reals))
			plt.plot(np.asarray(da_fakes))
			plt.plot(np.asarray(db_reals))
			plt.plot(np.asarray(db_fakes))
	
	if not args.hideGen:
		if args.median:
			plt.plot(sig.medfilt(np.asarray(gas, dtype='float64'), 101), linewidth=2.0)
			plt.plot(sig.medfilt(np.asarray(gbs, dtype='float64'), 101), linewidth=2.0)
		else:
			plt.plot(np.asarray(gas))
			plt.plot(np.asarray(gbs))

	if not args.hideCyc:
		if args.median:
			plt.plot(sig.medfilt(np.asarray(cycas, dtype='float64'), 101), linewidth=2.0)
			plt.plot(sig.medfilt(np.asarray(cycbs, dtype='float64'), 101), linewidth=2.0)
		else:
			plt.plot(np.asarray(cycas))
			plt.plot(np.asarray(cycbs))

	if not args.hideFeat:
		if args.median:
			plt.plot(sig.medfilt(np.asarray(feats, dtype='float64'), 101), linewidth=2.0)
		else:
			plt.plot(np.asarray(feats))


	plt.legend(['D_A_real', 'D_A_fake', 'D_B_real', 'D_B_fake', 
		'G_A', 'G_B', 'Cyc_A', 'Cyc_B', 'Feat'], ncol=9, mode='expand', loc='lower center')


	if args.median:
		plt.title('Median filtered Losses over iterations')
	else:
		plt.title('Losses over iterations')
	plt.ylabel('Loss')
	plt.xlabel('Iteration')
	# plt.legend(['G_A', 'G_B'], loc='lower left')
	if args.save:
		plt.savefig('CycWGAN_losses.png')
		plt.close()
	else:
		plt.show()


if __name__ == "__main__":
	main()



