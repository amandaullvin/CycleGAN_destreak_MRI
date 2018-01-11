#!/usr/bin/python
import sys
import matplotlib.pyplot as plt
import numpy as np


def parse_line(line):
	# (epoch: 3, iters: 148, time: 0.104) D_A_real: 26.258 D_A_fake: 24.458 G_A: 25.814 Cyc_A: 5.322 D_B_real: 23.165 D_B_fake: 25.765 G_B: 24.903 Cyc_B: 5.746 featL: 19.404
	elmts = line.split()

	if (len(elmts) != 24):
		return (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
	it = elmts[3]
	da_real = elmts[7]
	da_fake = elmts[9]
	ga = elmts[11]
	cyca = elmts[13]
	db_real = elmts[15]
	db_fake = elmts[17]
	gb = elmts[19]
	cycb = elmts[21]
	feat = elmts[23]
	return (it, da_real, da_fake, ga, cyca, db_real, db_fake, gb, cycb, feat)


def main():
	# open file and read it line by line
	if len(sys.argv) != 2:
		raise ValueError('Requires only one argument: path to log file.')

	filepath = sys.argv[1]
	f = open(filepath, 'r')

	its, da_reals, da_fakes, db_reals, db_fakes = [], [], [], [], []

	for line in f:
		it, da_real, da_fake, _, _, db_real, db_fake, _, _, _ = parse_line(line)
		if it < 0:
			continue


		its.append(it)
		da_reals.append(da_real)
		db_reals.append(db_real)
		da_fakes.append(da_fake)
		db_fakes.append(db_fake)

	print("Parsing completed.")
	plt.figure()
	plt.plot(np.asarray(da_reals))
	plt.plot(np.asarray(da_fakes))
	plt.plot(np.asarray(db_reals))
	plt.plot(np.asarray(db_fakes))
	plt.title('Discriminator losses over iterations')
	plt.ylabel('Loss')
	plt.xlabel('Iteration')
	plt.legend(['D_A Real', 'D_A Fake', 'D_B Real', 'D_B Fake'], loc='lower right')
	plt.savefig('d_losses.png')
	plt.close()


if __name__ == "__main__":
	main()



