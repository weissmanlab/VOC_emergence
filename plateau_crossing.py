#/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools as it

def mendel_prob(x, y, z):
	'''
	Calculate the probability for x and y recombine to form z. 

	Assuming recombination rate is 1/2. 

	x, y, z -- Genotypes
	'''
	p = 1
	for locus in range(len(x)):
		if x[locus] == y[locus] == z[locus]:
			continue
		elif x[locus] == z[locus] or y[locus] == z[locus]:
			p *= 1/2
		else:
			return 0
	return p

def sampling(N, p, eps=1e-8):
	'''
	Sampling N based on probabilities p.

	eps prevents small float number caused by python, which should be 0. 
	'''
	if np.all(p > eps): # No small probabilities, ordinary multinomial is safe.
		return np.random.multinomial(N, p)
	else: # Handle the small probabilities with Poisson draws, then do multinomial for the rest.
		n = np.zeros(len(p), dtype=np.uint64)
		ismall = np.nonzero(p <= eps)[0]
		ilarge = np.nonzero(p > eps)[0]
		for i in ismall:
			n[i] = np.random.poisson(N * p[i])
		n_large = np.random.multinomial(N - np.sum(n), p[ilarge] / np.sum(p[ilarge]))
		for i, nl_i in enumerate(n_large):
			n[ilarge[i]] = nl_i
		return n 

class population:
	def __init__(self):
		# Get the parameters from the command line.
		parser = argparse.ArgumentParser()
		parser.add_argument("--N", type=int, default=100, help="Population size")
		parser.add_argument("--mut", type=float,default=1e-2, help="mutation rate")
		parser.add_argument("--rec", type=float, default=1e-4, help="frequency of sex")
		parser.add_argument("--s", type=float, default=0.1, help="advantage of triple mutant")
		parser.add_argument("--k", type=int, default=1, help="mutations to valley crossing")
		parser.add_argument("--poptype", choices=['C', 'E', 'F'], default='C',\
			  help="population type: C (Constant), E (Exponential), or F (Read from file n.csv)")
		parser.add_argument("--lineage", type=int, default=0,\
			  help="keep track of lineage (up to a number). Recomend run = 1 and plot out.")
		parser.add_argument("--runs", type=int, default=100, help="repeat time for simulation")
		parser.add_argument("--out", default='./', help="prefix for output files")
		parser.add_argument("--g", type=float, default=1, help="growth rate for exponential population")
		parser.add_argument("--tmax", type=int, default=500, help="max number of generations to run for")
		parser.add_argument("--tstep", type=int, default=10,\
			 help="record the population state every tstep generations")
		parser.add_argument("--seed", type=int, default=None,  help="seed for random number generator")

		parser.add_argument('--plot', dest='plot', action='store_true',\
			  help="plot trajectories (and lineages)")
		parser.add_argument('--no-plot', dest='plot', action='store_false',  help="no plots")
		parser.set_defaults(plot=False)

		# Pack all params into self.args.
		self.args = parser.parse_args()
		# Record important params
		with open(self.args.out + 'params.txt','w') as outfile:
			print("\n".join(["N = {:.3g}", "poptype = {}", "mu = {}", "r = {}", "s = {}", "k = {}", "g = {}", "tstep = {}", "seed = {}"])\
				.format(self.args.N, self.args.poptype, self.args.mut, self.args.rec, self.args.s, self.args.k, self.args.g, self.args.tstep, self.args.seed), file=outfile)

		# Form the population list nlist by population type. (constant, exponential or from file).
		if self.args.poptype == 'F': 
			self.nlist = np.loadtxt('cases.csv', delimiter = ',').astype(int)
			self.args.tmax = len(self.nlist)
		if self.args.poptype == 'C':
			self.nlist = np.ones(self.args.tmax).astype(int) * self.args.N
		if self.args.poptype == 'E':
			# There is a growth rate as g.
			self.nlist = np.ones(self.args.tmax).astype(int) * self.args.N
			for i in range(self.args.tmax):
				self.nlist[i] *= pow(self.args.g, i)
			self.nlist = self.nlist.astype(int) # force it to be int.

		# Generate genotypes: -1 as wildtype, 1 as mutant in a site.
		self.genotypes = np.asarray(list(it.product((-1,1), repeat=self.args.k)))
		# Sort the genotypes, so that the last one is full mutant
		self.genotypes = self.genotypes[np.argsort(self.genotypes.sum(axis=1))]
		# Genotype dimension
		self.dms = pow(2, self.args.k) 
		
		# form mutation matrix: m_i,j as mutation from i to j.
		self.mutmat = np.zeros((self.dms,self.dms))
		for i in range(self.dms):
			for j in range(self.dms):
				# If the hamming disntance of two genotype is 1, mutation probability is mut.
				if np.count_nonzero(self.genotypes[i] - self.genotypes[j]) == 1:
					self.mutmat[i,j] = self.args.mut

		# Form recombination matrix: r_i,j as rec from i to j.
		self.recmat = np.array([[[mendel_prob(x, y, z) for z in self.genotypes] for y in self.genotypes] for x in self.genotypes])

		# Form fitness array: s is the breeding advantage.
		self.fit_genotype = np.ones(self.dms)
		self.fit_genotype[-1] += self.args.s

		# Initial random generator.
		np.random.seed(self.args.seed)
		# Open files.
		self.outfile = open(self.args.out + "out.txt", 'w')
		self.trafile = open(self.args.out + "trajectory.txt", 'w')
		self.logfile = open(self.args.out + "log.txt", 'w')

	def mutation(self):
		'''
		Calculate changes in frequency based on mutation rate.

		Lineage may be recorded.
		'''
		if self.args.mut > 0:
			# The first part is mutating to certain genotype, the latter one is mutating out.
			self.freq_genotype += self.freq_genotype @ self.mutmat - self.args.mut * self.args.k * self.freq_genotype
			# If need to record lineage.
			if self.args.lineage > 0:
				self.lineagecount()

		
	def lineagecount(self):
		'''
		Record lineage. Consider a new lineage occurs when mutates a full mutant.
		'''
		self.prvs_gen = self.num_genotype
		totpop = np.sum(self.num_genotype)
		self.num_genotype = sampling(totpop, self.freq_genotype, eps=1/self.args.N)
		self.freq_genotype = self.num_genotype / totpop
		# Compare prev and new mutant.
		if self.num_genotype[-1] > self.prvs_gen[-1]:
			self.lincount += 1
			emptypos = np.where(self.lineage == 0)[0]
			# Only record when there is a space
			if len(emptypos) != 0:
				self.lineage[emptypos[0]] = self.num_genotype[-1] - self.prvs_gen[-1]
			
	def recombination(self):
		'''
		Calculate changes in frequency based on recombination rate.
		'''
		if self.args.rec > 0:
			self.freq_genotype += -self.args.rec/2 * self.freq_genotype + self.args.rec/2 * self.freq_genotype @ (self.freq_genotype @ self.recmat)

	def selection(self):
		'''
		Calculate changes in frequency based on selection.
		'''
		self.freq_genotype *= self.fit_genotype / (self.freq_genotype @ self.fit_genotype)

	def evolve(self):
		for run in np.arange(self.args.runs):
			# Initial population
			self.initpop()
			for t in np.arange(self.args.tmax):
				self.mutation()
				self.recombination()
				self.selection()
				self.prvs_gen = self.num_genotype
				self.num_genotype = sampling(self.nlist[t], self.freq_genotype)
				self.freq_genotype = self.num_genotype / self.nlist[t]
				# Sample lineage population based on their freq. Add up to full mutant.
				if self.args.lineage > 0:
					totpop = np.sum(self.lineage)
					if totpop > 0:
						self.lineage = sampling(self.num_genotype[-1], self.lineage/totpop, eps=1/self.args.N)
				self.check(run, t)
				
				if t%self.args.tstep == 0:
					self.output(run, t, self.num_genotype, self.trafile)
					if self.args.lineage > 0:
						self.output(run, t, self.lineage, self.lineagefile)
				# Consider full mutant is occupying the population
				if self.freq_genotype[-1] > 1/5:
					print(t, file = self.outfile)
					self.output(run, t, self.num_genotype, self.trafile)
					if self.args.lineage == True:
						self.output(run, t, self.lineage, self.lineagefile)
					break

		self.trafile.flush()
		self.trafile.close()
		self.outfile.flush()
		self.outfile.close()
		self.logfile.flush()
		self.logfile.close()
		
		if self.args.plot == True:
			self.plot("trajectory.txt")
			if self.args.lineage > 0:
				self.lineagefile.flush()
				self.lineagefile.close()
				print("Mutant occurance: ", self.lincount)
				print("Size of lineage = ", np.size(np.where(self.lineage > 0)))
				self.plot("lineage.txt", pos = 0)
			plt.show()

	def initpop(self):
		'''
		Initial/Reset frequencies, number of genotypes adn lineages.
		'''
		# Number of individuals of each genotype. Initial population: all wildtype.
		self.num_genotype = np.zeros(self.dms, dtype=np.uint64)
		self.num_genotype[1] = self.nlist[0]
		self.prvs_gen = self.num_genotype
		# Frequency of genotyes. This is also the selection probability of genotypes in sampling.
		self.freq_genotype = np.zeros(self.dms)
		self.freq_genotype[0] = 1.0
		# Initial lineage tracking.
		self.lincount = 0
		if self.args.lineage > 0:
			self.lineage = np.zeros(self.args.lineage, dtype=int)
			self.lineagefile = open(self.args.out + "lineage.txt", 'w')
		
	def output(self, run, t, data, filepath):
		'''
		Output data.
		'''
		if t == 0:
			print('run = {}'.format(run), file = filepath)
		print(t, file = filepath, end=",")
		print(','.join(str(n) for n in data), file = filepath)
	

	def plot(self, filename, pos = -1):
		'''
		Plot mutant/lineages.
		pos -- Index of plot data. Output full line when equals 0.
		'''
		plotfile = open(filename, 'r')
		x = np.array([])
		y = np.array([])
		while True:
			line = plotfile.readline()
			# If starts with 'run' (new run), or line is empty (end of file), plot.
			if len(line) == 0 or line[0] == "r":
				# If it is not the first line.
				if len(y) != 0:
					# y is 1d array.
					if pos != 0:
						plt.plot(x,y,'o-')
					# y is 2d array. Plot each lines.
					else:
						y = np.stack(y, axis=-1)
						for doty in y:
							# Only plot VOCs.
							if np.any(doty > 100):
								plt.plot(x,doty,linestyle = "-")
				# If reaches the end of file
				if len(line) == 0:
					break
				x = np.array([])
				y = np.array([])
				continue
			# Omit the newline and commas.
			lineitems = line.split('\n')[0].split(',')
			x = np.append(x, int(lineitems[0]))
			# Only extract a column
			if pos != 0:
				y = np.append(y, int(lineitems[pos]))
			# All columns
			else:
				lineint = np.array([int(i) for i in lineitems])
				if len(y) == 0:
					y = np.append(y, lineint[2:] + 1)
				else:
					y = np.vstack((y, lineint[2:] + 1))

		plt.yscale("log",base=10)
		plt.xlabel("T")
		plt.ylabel("Mutant")
		# Theoretical exp line
		expmut = np.exp(self.args.s*x)
		plt.plot(x,expmut, color = 'black', markersize=0, linestyle = '-')
		
	def check(self, run, t):
		if t == 0:
			return True
		status = 0
		if self.num_genotype[-1] > 0 and np.sum(self.prvs_gen[-self.args.k-1:]) == 0:
			print("Warning: full mutant appear out of nothing at run = {}, t={}.".format(run, t), file=self.logfile)
			status = 1
		if np.abs(np.sum(self.freq_genotype)-1) > 1e-6 or np.any(self.freq_genotype < 0) or np.any(self.freq_genotype > 1):
			print("Warning: improper genotype frequencies run = {}, t={}.".format(run, t), file=self.logfile)
			status = 2
		if np.any(np.logical_and(0 < self.freq_genotype, self.freq_genotype < 1/self.args.N)):
			print("Warning: small probability occurs run = {}, t={}.".format(run, t), file=self.logfile)
			status = 3
		if status > 0:
			print("Previous generation: ", self.prvs_gen, \
					"\nPre-sampling number of genotypes: ", ' '.join(str(x) for x in self.prvs_gen), \
						"\nPre-sampling frequency: ", ' '.join(str(x) for x in self.freq_genotype),\
							"\nNew generation: ", self.num_genotype, "\n", file=self.logfile)
			return False
		return True
		
if __name__ == "__main__":
	pop = population()
	pop.evolve()