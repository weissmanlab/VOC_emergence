#/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools as it

def mendel_prob(x, y, z):
# x and y recombine to form z:
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
	if np.all(p > eps): # no small probabilities, ordinary multinomial is safe
		return np.random.multinomial(N, p)
	else: # handle the small probabilities with Poisson draws, then do multinomial for the rest
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
		# get the parameters from the command line:
		parser = argparse.ArgumentParser()
		parser.add_argument("--N", type=np.int, default=100, help="Population size")
		parser.add_argument("--mut", type=np.float,default=1e-2, help="mutation rate")
		parser.add_argument("--rec", type=np.float, default=1e-4, help="frequency of sex")
		parser.add_argument("--s", type=np.float, default=0.1, help="advantage of triple mutant")
		parser.add_argument("--k", type=np.int, default=1, help="mutations to valley crossing")
		parser.add_argument("--poptype", choices=['C', 'E', 'F'], default='C',  help="population type: C (Constant), E (Exponential), or F (Read from file n.csv)")
		parser.add_argument("--runs", type=np.int, default=100, help="repeat time for simulation")
		parser.add_argument("--out", default='./', help="prefix for output files")
		parser.add_argument("--g", type=np.float, default=1, help="growth rate for exponential population")
		parser.add_argument("--tmax", type=np.int, default=100, help="max number of generations to run for")
		parser.add_argument("--tstep", type=np.int, default=10, help="record the population state every tstep generations")
		parser.add_argument("--seed", type=np.int, default=None,  help="seed for random number generator")
		self.args = parser.parse_args()
		with open(self.args.out + 'params.txt','w') as outfile:
			print("\n".join(["N = {:.3g}", "poptype = {}", "mu = {}", "r = {}", "s = {}", "k = {}", "g = {}", "tstep = {}", "seed = {}"])\
				.format(self.args.N, self.args.poptype, self.args.mut, self.args.rec, self.args.s, self.args.k, self.args.g, self.args.tstep, self.args.seed), file=outfile)

		# Form the population list by population type.(Constant, exponential or from file)
		if self.args.poptype == 'F': 
			self.nlist = np.loadtxt('cases.csv', delimiter = ',').astype(int)
			self.args.tmax = len(self.nlist)
		if self.args.poptype == 'C':
			self.nlist = np.ones(self.args.tmax).astype(int) * self.args.N
		if self.args.poptype == 'E': # There is a growth rate as g.
			self.nlist = np.ones(self.args.tmax).astype(int) * self.args.N
			for i in range(self.args.tmax):
				self.nlist[i] *= pow(self.args.g, i)
			self.nlist = self.nlist.astype(int) # force it to be int

		# generate genotypes: -1 as wildtype, 1 as mutant
		self.genotypes = np.asarray(list(it.product((-1,1), repeat=self.args.k)))
		self.genotypes = self.genotypes[np.argsort(self.genotypes.sum(axis=1))]
		dms = pow(2, self.args.k) # combination (dimension) of genotypes

		# number of individuals of each genotype
		self.num_genotype = np.zeros(dms, dtype=np.uint64)
		self.num_genotype[1] = self.nlist[0]

		# initial population: all wildtype
		self.freq_genotype = np.zeros(dms) # this is also the selective advantage of a genotype
		self.freq_genotype[0] = 1.0
		
		# form mutation matrix: m_i,j as mutation from i to j 
		self.mutmat = np.zeros((dms,dms))
		for i in range(dms):
			for j in range(dms):
				if np.count_nonzero(self.genotypes[i] - self.genotypes[j]) == 1:
					self.mutmat[i,j] = self.args.mut

		# form recombination matrix: r_i,j as rec from i to j 
		self.recmat = np.array([[[mendel_prob(x, y, z) for z in self.genotypes] for y in self.genotypes] for x in self.genotypes])

		# form fitness array: s_i is the breeding advantage
		self.fit_genotype = np.ones(dms)
		self.fit_genotype[-1] += self.args.s

		#initial random generator
		np.random.seed(self.args.seed)
		# open files
		self.outfile = open(self.args.out + "out.txt", 'w')
		self.trafile = open(self.args.out + "trajectory.txt", 'w')
		self.logfile = open(self.args.out + "log.txt", 'w')

	def mutation(self):
		if self.args.mut > 0:
			self.freq_genotype += self.freq_genotype @ self.mutmat - self.args.mut * 3 * self.freq_genotype

	def recombination(self):
		if self.args.rec > 0:
			self.freq_genotype += -self.args.rec/2 * self.freq_genotype + self.args.rec/2 * self.freq_genotype @ (self.freq_genotype @ self.recmat)

	def selection(self):
		self.freq_genotype *= self.fit_genotype / (self.freq_genotype @ self.fit_genotype)

	def evolve(self):
		for run in np.arange(self.args.runs):
			self.reset()
			for t in np.arange(self.args.tmax):
				self.mutation()
				self.recombination()
				self.selection()
				self.prvs_gen = self.num_genotype
				self.num_genotype = sampling(self.nlist[t], self.freq_genotype)
				self.freq_genotype = self.num_genotype / self.nlist[t]
				self.check(t)

				if t%self.args.tstep == 0:
					self.output(run,t)
				if self.freq_genotype[-1] > 1/2:
					print(t, '\t', file = self.outfile)
					self.output(run, t)
					break

	def reset(self):
		self.freq_genotype = np.zeros(pow(2,self.args.k))
		self.freq_genotype[0] = 1.0
		
	def output(self, run, t):
		if t == 0:
			print('run = {}'.format(run), file = self.trafile)
		print(t, ' '.join(str(n) for n in self.num_genotype), file = self.trafile)
		
	def check(self, t):
		if t == 0:
			return True
		status = 0
		if self.num_genotype[-1] > 0 and np.sum(self.prvs_gen[-self.args.k-1:]) == 0:
			print("Warning: full mutant appear out of nothing at t={}.".format(t), file=self.logfile)
			status = 1
		if np.abs(np.sum(self.freq_genotype)-1) > 1e-6 or np.any(self.freq_genotype < 0) or np.any(self.freq_genotype > 1):
			print("Warning: improper genotype frequencies t={}.".format(t), file=self.logfile)
			status = 2
		if np.any(self.freq_genotype < 1e-8):
			print("Warning: small probability occurs t={}.".format(t), file=self.logfile)
			status = 3
		if status > 0:
			print("Previous generation: ", self.prvs_gen, \
					"\nPre-sampling frequencies: ", ' '.join(str(x) for x in self.freq_genotype), \
						"\nNew generation: ", self.num_genotype, "\n", file=self.logfile)
			return False
		return True
		
if __name__ == "__main__":
	pop = population()
	pop.evolve()