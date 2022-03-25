# /usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import itertools as it
import time
import os
from scipy.stats import expon
from scipy.special import comb


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


def dirichlet_multinomial(N, p, scalar, eps=1e-12):
    '''
    Dirichlet multinomial sampling N based on probabilities p and scalar.
    Alpha = p*scalar
    '''
    alphas = p*scalar
    # alpha = 0 would raise an error
    if len(np.where(alphas == 0)[0]) == 0:
        new_p = np.random.dirichlet(alphas)
        return multinomial(N, new_p, scalar, eps)
    # temperory replace zeros
    else:
        pos_nonzeros = np.where(alphas != 0)[0]
        new_p = np.random.dirichlet(alphas[pos_nonzeros])
        sample_short = multinomial(N, new_p, scalar, eps)
        sample_full = np.zeros(len(p)).astype(np.uint64)
        sample_full[pos_nonzeros] = sample_short
        return sample_full


def multinomial(N, p, scalar=1, eps=1e-12):
    '''
    Multinomial sampling N based on probabilities p.
    scalar is to match dirichlet multinomial format.
    eps prevents error introduced by small float numbers. 
    '''

    # No small probabilities, ordinary multinomial is safe.
    if np.all(p > eps):
        return np.random.multinomial(N, p).astype(np.uint64)
    else:  # Handle the small probabilities with Poisson draws, then do multinomial for the rest.
        n = np.zeros(len(p)).astype(np.uint64)
        ismall = np.nonzero(p <= eps)[0]
        ilarge = np.nonzero(p > eps)[0]
        for i in ismall:
            n[i] = np.random.poisson(N * p[i])
        n_large = np.random.multinomial(
            N - np.sum(n), p[ilarge] / np.sum(p[ilarge]))
        for i, nl_i in enumerate(n_large):
            n[ilarge[i]] = nl_i
        return n


class Population:
    def __init__(self):
        # Get the parameters from the command line.
        parser = argparse.ArgumentParser()
        parser.add_argument("--N", type=int, default=10000000,
                            help="Population size")
        parser.add_argument("--mut", type=float,
                            default=0, help="mutation rate")
        parser.add_argument("--rec", type=float,
                            default=0, help="frequency of sex")
        parser.add_argument("--s", type=float, default=1,
                            help="advantage of triple mutant")
        parser.add_argument("--K", type=int, default=10,
                            help="mutations to valley crossing")
        parser.add_argument("--poptype", choices=['C', 'E', 'F'], default='C',
                            help="population type: C (Constant), E (Exponential), or F (Read from file covid.csv)")
        parser.add_argument("--runs", type=int, default=2,
                            help="repeat time for simulation")
        parser.add_argument("--outpath", default='./data/',
                            help="prefix for output files")
        parser.add_argument("--g", type=float, default=1,
                            help="growth rate (1+g) for exponential population. ")
        parser.add_argument("--tmax", type=int, default=100,
                            help="max number of generations to run for")
        parser.add_argument("--tstep", type=int, default=1,
                            help="record the population state every tstep generations")
        parser.add_argument("--seed", type=int, default=None,
                            help="seed for random number generator")
        parser.add_argument("--sampling_method", choices=['multinomial', 'dirichlet'], default='dirichlet',
                            help="sampling method: multinomial or dirichlet multinomial sampling")
        parser.add_argument("--diri_k", type=float, default=0.1,
                            help="overdispersion parameter. used in dirichlet multinomial.")
        # Options
        parser.add_argument("--lineage", action='store_true', default=True,
                            help="keep track of lineage. Recomend run = 1 and plot out.")
        parser.add_argument("--ici", action='store_true', default=True,
                            help="simulation of the immunocompromised patients.")
        parser.add_argument("--pf_ici", type=float, default=1e-7,
                            help="probability of immunocompromised patients (who produce VOCs).")
        parser.add_argument("--mu_ici", type=float, default=0.01,
                            help="rate of within-host fixation.")
        parser.add_argument("--accumulative", action='store_true', default=False,
                            help="Each mutation has the same selection effect: s/k.")
        parser.add_argument('--plot', choices=['none', 'genotype', 'lineage', 'voc'], default='none',
                            help="plot trajectories of all genotypes (and lineages).\
                                    option 'voc' for only plot vocs instead of all lineages.")
        parser.add_argument('--log', action='store_true', default=False,
                            help="check for valid frequencies and number of genotypes. output to .log file.")

        # Pack all params into self.args.
        self.args = parser.parse_args()
        # initial the outpath and filename
        self.init_outpath()
        # Form the population list nlist by population type. (constant, exponential or from file).
        # default constant
        self.nlist = np.ones(self.args.tmax).astype(np.uint64) * self.args.N
        if self.args.poptype == 'F':
            self.nlist = np.loadtxt(
                'casesD.csv', delimiter=',').astype(np.uint64)
            self.args.tmax = len(self.nlist)
        elif self.args.poptype == 'E':
            # growth rate as g.
            for i in range(self.args.tmax):
                self.nlist[i] *= pow(1 + self.args.g, i)

        # Generate genotypes: -1 as wildtype, 1 as mutant in a site.
        self.genotypes = np.asarray(
            list(it.product((-1, 1), repeat=self.args.K)))
        # Sort the genotypes, so that the last one is full mutant
        self.genotypes = self.genotypes[np.argsort(self.genotypes.sum(axis=1))]
        # Genotype dimension
        self.dms = pow(2, self.args.K)

        # form mutation matrix: m_i,j as mutation from i to j.
        self.mutmat = np.zeros((self.dms, self.dms))
        for i in range(self.dms):
            for j in range(self.dms):
                # If the hamming disntance of two genotype is 1, mutation probability is mut.
                if np.count_nonzero(self.genotypes[i] - self.genotypes[j]) == 1:
                    self.mutmat[i, j] = self.args.mut
        # form recombination matrix: r_i,j as rec from i to j.
        self.recmat = np.array([[[mendel_prob(x, y, z) for z in self.genotypes]
                               for y in self.genotypes] for x in self.genotypes])
        # Form fitness array: s is the breeding advantage.
        self.fit_genotype = np.ones(self.dms)
        self.fit_genotype[-1] += self.args.s
        if self.args.accumulative:
            # Each mutations has s/k increase in fitness.
            for i in np.arange(1, self.args.K):
                self.fit_genotype[self.args.K *
                                  (i-1)+1:-1] += self.args.s/self.args.K

        # sampling methods
        if self.args.sampling_method == 'multinomial':
            self.sampling = multinomial
        elif self.args.sampling_method == 'dirichlet':
            self.sampling = dirichlet_multinomial
        # generation methods
        if self.args.lineage:
            self.generation = self.generation_lineage
        else:
            self.generation = self.generation_no_lineage

        # Initial random generator.
        np.random.seed(self.args.seed)

        # Open files.
        self.trajfile = open(self.args.outpath + ".traj", 'w')
        if self.args.lineage:
            self.lineagefile = open(self.args.outpath + ".lineage", 'w')
            self.vocfile = open(self.args.outpath + ".voc", 'w')
        if self.args.ici:
            self.icifile = open(self.args.outpath + ".ici", 'w')
        if self.args.log:
            self.logfile = open(self.args.outpath + ".log", 'w')
        # Record important params
        with open(self.args.outpath + '.params', 'w') as outfile:
            print("\n".join(["N = {:.3g}", "poptype = {}", "mu = {}", "r = {}", "s = {}",
                             "k = {}", "g = {}", "tstep = {}", "seed = {}", "fitness accumultive = {}", "sampling_method = {}", "diri_k = {}"]).format(self.args.N, self.args.poptype, self.args.mut, self.args.rec, self.args.s, self.args.K, self.args.g, self.args.tstep, self.args.seed, self.args.accumulative, self.args.sampling_method, self.args.diri_k), file=outfile)
            if self.args.lineage:
                print("Lineages: True", file=outfile)
            if self.args.ici:
                print("ICI: True", file=outfile)
                print("\n".join(["pf = {}", "mu_ici = {}"]).format(
                    self.args.pf_ici, self.args.mu_ici), file=outfile)

    def init_pop(self):
        '''
        Initial/Reset frequencies, number of genotypes and lineages.
        '''
        # Number of individuals of each genotype. Initial population: all wildtype.
        self.num_genotype = np.zeros(self.dms).astype(np.uint64)
        self.num_genotype[0] = self.nlist[0]
        self.prvs_gen = self.num_genotype
        # Frequency of genotyes. This is also the selection probability of genotypes in sampling.
        self.freq_genotype = np.zeros(self.dms).astype(np.float64)
        self.freq_genotype[0] = 1.0

        # Initial lineage tracking
        if self.args.lineage:
            # max voc and voc threshold
            self.voc_max = 5
            self.voc_th = 1/self.args.s
            # number and frequency of lineages
            self.num_genotype_lineage = np.zeros(self.dms).astype(np.uint64)
            self.num_genotype_lineage[0] = self.nlist[0]
            self.freq_genotype_lineage = np.zeros(self.dms).astype(np.float64)
            self.freq_genotype_lineage[0] = 1.0
            # number of lineage
            self.num_voc = 0
            self.num_lineage = 0
            # lineage array and time of occurance
            self.num_lineage_array = np.empty((0, 1)).astype(np.uint64)
            self.lineage_t = np.array([]).astype(np.uint16)
        # initial ici tracking
        if self.args.ici:
            # number ici of the mutants. NOT including WT
            self.num_genotype_ici = np.zeros(self.args.K).astype(np.uint64)
            # Time to accumulate one more muatation. NOT including full mutant.
            self.ici_t = [np.empty([1, 0], dtype=np.uint16)
                          for i in range(self.args.K)]
            # index to split genotypes into single mutant, double mutant...
            self.ici_split_pos = np.cumsum(
                [comb(self.args.K, i).astype(np.int16) for i in np.arange(self.args.K)])
            # fitness of each ici genotype
            self.fit_ici = self.fit_genotype[self.ici_split_pos]
            # ici position in lineage array
            self.lineage_array_ici_pos = np.empty([0, 0], dtype=np.uint16)

    def init_outpath(self):
        '''
        initial output path and filename
        '''
        # Create data folder
        if not os.path.exists(self.args.outpath):
            os.makedirs(self.args.outpath)
        # create file names
        # k: genome size
        self.args.outpath += "K{}".format(self.args.K)
        # population size and type
        self.args.outpath += self.args.poptype
        if self.args.poptype == 'C':
            self.args.outpath += "N" + "{:1.1E}".format(self.args.N)
        elif self.args.poptype == 'E':
            self.args.outpath += "N" + \
                "{:1.1E}".format(self.args.N) + "G" + \
                "{:1.1E}".format(self.args.g)
        # mutations rate
        self.args.outpath += "M{:1.0E}".format(self.args.mut)
        # sampling methods
        if self.args.sampling_method == 'multinomial':
            self.args.outpath += "MN"
        elif self.args.sampling_method == 'dirichlet':
            self.args.outpath += "DM" + 'k' + str(self.args.diri_k)
        # immunocompromised patients
        if self.args.ici:
            self.args.outpath += "ICI"
        # fitness landscape
        if self.args.accumulative:
            self.args.outpath += "A"

    def mutation(self):
        '''
        Calculate changes in frequency based on mutation rate.
        Return frequency changes for mutate in and out.
        '''
        if self.args.mut == 0:
            return [0, 0]
        # The first part is mutating to certain genotype, the latter one is mutating out.
        mutate_in = self.freq_genotype @ self.mutmat
        mutate_out = self.args.mut * self.args.K * self.freq_genotype
        self.freq_genotype += mutate_in - mutate_out
        return [mutate_in[-1], mutate_out[-1]]

    def recombination(self):
        '''
        Calculate changes in frequency based on recombination rate.
        Return frequency changes for recombine in and out.
        '''
        if self.args.rec == 0:
            return [0, 0]
        # The first part is recombine to certain genotype, the latter one is recombine out.
        rec_in = self.args.rec/2 * \
            self.freq_genotype @ (self.freq_genotype @ self.recmat)
        rec_out = self.args.rec/2 * self.freq_genotype
        self.freq_genotype[:] += rec_in - rec_out
        return [rec_in[-1], rec_out[-1]]

    def selection(self):
        '''
        Calculate changes in frequency based on selection.
        '''
        if self.args.s == 0:
            return 1
        selection = self.fit_genotype / \
            (self.freq_genotype @ self.fit_genotype)
        self.freq_genotype *= selection
        return selection[-1]

    def evolve(self):
        '''
        Evolution of designed population.
        '''
        for self.run in np.arange(self.args.runs):
            # Initial population
            self.init_pop()
            for self.t in np.arange(self.args.tmax):
                # update mutant counts
                self.generation()
                if self.output_stop():
                    break
            # record max number of lineages
            if self.args.lineage:
                self.record_lineages_max()
        self.output_add_header()
        # plots
        if self.args.plot != 'none':
            self.traj_plot()

    def output_stop(self):
        '''
        Output status during evolution.
        Stop when it reaches the criteria (after output).
        '''
        if self.args.log:
            self.check()
        if self.t % self.args.tstep == 0:
            self.output_traj()
        # Consider full mutant is occupying the population
        if self.freq_genotype[-1] > 1/2 or (self.args.lineage and self.num_voc >= self.voc_max):
            self.output_traj()
            return True

    def record_lineages_max(self):
        '''
        Record the emergence time and the numbers of lineages.
        '''
        try:
            # record the emerge time of lineages
            self.lineage_time_max = np.max(
                [self.lineage_time_max, self.num_lineage_array.shape[1]])
            # record max number of lineages
            self.num_voc_max = np.max([self.num_voc_max, self.num_voc])
            if self.args.ici:
                self.num_voc_ici_max = np.max(
                    [self.num_voc_ici_max, self.num_genotype_ici[-1]])
        except:
            self.lineage_time_max = self.num_lineage_array.shape[1]
            self.num_voc_max = self.num_voc
            if self.args.ici:
                self.num_voc_ici_max = self.num_genotype_ici[-1]

        # output the status
        if self.num_lineage > 0:
            self.output_lineage()
        if self.num_voc > 0:
            self.output_voc()
        if self.args.ici:
            self.output_ici()

    def generation_no_lineage(self):
        '''
        Update frequency and genotype counts.
        '''
        self.selection()
        self.mutation()
        self.recombination()
        self.prvs_gen = self.num_genotype
        if self.args.ici:
            self.frequency_ici()
        # new sampling
        scalar = 1
        if self.args.sampling_method == "dirichlet":
            # if mutant is present
            scalar = self.args.diri_k*self.nlist[self.t]
        self.num_genotype = self.sampling(
            self.nlist[self.t], self.freq_genotype, scalar)
        # update ici sampling
        if self.args.ici:
            self.sampling_ici()
        self.freq_genotype = self.num_genotype / self.nlist[self.t]

    def generation_lineage(self):
        '''
        Update frequency and genotype counts. 
        Update lineage array and VOC records.
        '''
        selection = self.selection()
        [mutate_in, mutate_out] = self.mutation()
        [rec_in, rec_out] = self.recombination()
        # for non-mutants
        self.freq_genotype_lineage[:self.dms -
                                   1] = self.freq_genotype[:self.dms-1]
        # if no mutant in current generation (excluding ici)
        if len(self.freq_genotype_lineage) == self.dms or np.sum(self.freq_genotype_lineage[self.dms-1:]) == 0:
            self.freq_genotype_lineage[-1] = self.freq_genotype[-1]
        else:
            # previous exsistent mutants
            self.freq_genotype_lineage[self.dms-1:] *= selection
            lineage_portion = self.freq_genotype_lineage[self.dms-1:]/np.sum(
                self.freq_genotype_lineage[self.dms-1:])
            self.freq_genotype_lineage[self.dms -
                                       1:] -= (mutate_out + rec_out)*lineage_portion
            # probability of new mutant
            self.freq_genotype_lineage[-1] = mutate_in + rec_in
        # if consider ici, update frequencies
        if self.args.ici:
            self.frequency_ici()
        # new sampling
        scalar = 1
        if self.args.sampling_method == "dirichlet":
            # if mutant is present
            scalar = self.args.diri_k*self.nlist[self.t]
        self.prvs_gen = self.num_genotype
        # sampling process
        self.num_genotype_lineage = self.sampling(
            self.nlist[self.t], self.freq_genotype_lineage, scalar)
        self.num_genotype[:-1] = self.num_genotype_lineage[:self.dms-1]
        self.num_genotype[-1] = np.sum(self.num_genotype_lineage[self.dms-1:])

        if self.num_lineage > 0:
            # add the number of lineage of the new time point
            self.num_lineage_array = np.column_stack(
                (self.num_lineage_array, self.num_genotype_lineage[self.dms-1:-1]))
            # clear void data points (#=0)
            self.delete_dead_lineage()
        # change array shape to adapt new lineage
        new_lineage = self.num_genotype_lineage[-1]
        if new_lineage > 0:
            self.add_new_lineage(new_lineage)
        # update ici sampling
        if self.args.ici:
            self.sampling_ici()
        # new frequency after the sampling
        self.freq_genotype_lineage = (
            self.num_genotype_lineage / self.nlist[self.t]).astype(np.float64)
        self.freq_genotype = (self.num_genotype /
                              self.nlist[self.t]).astype(np.float64)
        # record voc
        self.num_voc = np.count_nonzero(
            self.num_genotype_lineage[self.dms-1:] > self.voc_th)

    def sampling_ici(self):
        '''
        Sample the ici cases of each genotype.
        '''
        def add_waiting_time(ici_new, index):
            new_ici_t = expon.rvs(size=ici_new, scale=1/self.args.mu_ici)
            self.ici_t[index] = np.append(self.ici_t[index], new_ici_t+self.t)

        # Number of indv of each genotype.
        num_genotype_split = np.split(self.num_genotype, self.ici_split_pos)
        # Get the number of single mutant and so on. Discard full mutants.
        num_mutants = [np.sum(i) for i in num_genotype_split[:-1]]
        # Sample the ici with different genotypes.
        ici_new = np.random.binomial(num_mutants, self.args.pf_ici)
        for i in np.arange(self.args.K):
            ici_new_i = ici_new[i]
            # add waiting time for this genotype
            if ici_new_i > 0:
                add_waiting_time(ici_new_i, i)
            # if there is any ici reaches its waiting time
            ici_t_cond = (self.ici_t[i] <= self.t)
            num_ici_new = np.count_nonzero(ici_t_cond)
            if np.any(ici_t_cond):
                self.ici_t[i] = np.delete(self.ici_t[i], ici_t_cond)
                # add to next genotype; no wt in num_genotype_ici
                self.num_genotype_ici[i] += num_ici_new
                # remove them from the previous genotypes
                if i > 0:
                    self.num_genotype_ici[i-1] -= num_ici_new
                # add waiting time to the accumulate one more mutation
                if i != self.args.K-1:
                    add_waiting_time(num_ici_new, i+1)

        # add array for new voc of ici
        if self.num_genotype_ici[-1] > 0:
            self.add_new_lineage(num_ici_new, scalar=0)
            new_pos = np.arange(num_ici_new) + self.num_lineage - num_ici_new
            self.lineage_array_ici_pos = np.append(
                self.lineage_array_ici_pos, new_pos)

    def frequency_ici(self):
        '''
        Update the frequencies from leakage. 
        Call this function after changing frequency before sampling genoytpe.
        '''
        # add ici leakage to the population
        if np.sum(self.num_genotype_ici) == 0:
            return
        # infection from ici
        num_ici_s = self.num_genotype_ici * self.fit_ici
        # sampling mean of normal pop
        mean_num = self.freq_genotype * self.nlist[self.t]
        # new sampling frequency
        mean_num[self.ici_split_pos] += num_ici_s
        self.freq_genotype = mean_num/np.sum(mean_num)
        # rescale lineage frequencies
        if self.args.lineage:
            # convert to mean number
            mean_num_lineage = self.freq_genotype_lineage * self.nlist[self.t]
            # for non-mutants
            mean_num_lineage[:self.dms-1] = mean_num[:-1]
            # lineage of ici
            if self.num_genotype_ici[-1] > 0:
                ici_pos = self.lineage_array_ici_pos + self.dms - 1
                # add ici lineage fitness
                mean_num_lineage[ici_pos] += self.fit_ici[-1]
            # rescale num to freq, to match overall freq of all lineages
            self.freq_genotype_lineage = mean_num_lineage / \
                np.sum(mean_num_lineage)

    def delete_dead_lineage(self):
        '''
        Delete lineages with 0 individuals.
        Keep ici lineages.
        '''
        # add ici mutants
        if self.args.ici:
            ici_pos = self.lineage_array_ici_pos + self.dms - 1
            self.num_genotype_lineage[ici_pos] += 1

        # pos of dead lineage
        delete_lin_pos = np.where(
            self.num_genotype_lineage[self.dms-1:-1] == 0)[0]
        self.num_lineage -= len(delete_lin_pos)
        self.num_genotype_lineage = np.delete(
            self.num_genotype_lineage, delete_lin_pos+self.dms-1)
        self.lineage_t = np.delete(self.lineage_t, delete_lin_pos)
        self.num_lineage_array = np.delete(
            self.num_lineage_array, delete_lin_pos, axis=0)

        # delete time (col) without lineages
        lineages_over_time = np.sum(self.num_lineage_array, axis=0)
        ori_len = len(lineages_over_time)
        delete_len = np.trim_zeros(lineages_over_time, 'f')
        delete_len = ori_len - len(delete_len)
        self.num_lineage_array = np.delete(
            self.num_lineage_array, np.arange(delete_len), axis=1)

        # update pos of ici cases and subtract ici cases from array
        if self.args.ici:
            self.update_ici_pos(delete_lin_pos)
            ici_pos = self.lineage_array_ici_pos + self.dms - 1
            self.num_genotype_lineage[ici_pos] -= 1

    def update_ici_pos(self, delete_lin_pos):
        '''
        Update index of ici mutants.
        '''
        if len(delete_lin_pos) == 0:
            return
        for ind, pos in enumerate(self.lineage_array_ici_pos):
            try:
                counts = np.count_nonzero(delete_lin_pos < pos)
                self.lineage_array_ici_pos[ind] -= counts
            except:
                continue

    def add_new_lineage(self, new_lineage, scalar=1):
        '''
        Add new lineages.
        Lineage source: from new ici and from sampling
        '''
        self.num_lineage += int(new_lineage)
        # add to genotype array
        new_genotypes = np.ones(int(new_lineage+1), dtype=np.uint64)*scalar
        new_genotypes[-1] = 0  # new mutant for next generations
        self.num_genotype_lineage = np.hstack(
            (self.num_genotype_lineage[:-1], new_genotypes))
        # add lineages to freq (for sampling)
        if self.args.ici:
            self.freq_genotype_lineage = np.hstack(
                (self.freq_genotype_lineage[:-1], new_genotypes*0))
        # add initial time of lineages
        new_lineage_t = (self.t*np.ones(new_lineage)).astype(np.uint16)
        self.lineage_t = np.hstack((self.lineage_t, new_lineage_t))
        # generate array for lineage time
        new_length = self.num_lineage_array.shape[1]
        # add new rows to lineage array
        new_lineage_array = np.zeros(
            (new_lineage, new_length)).astype(np.uint64)
        new_lineage_array[:, -1] = 1
        self.num_lineage_array = np.vstack(
            (self.num_lineage_array, new_lineage_array))

    def output_traj(self):
        '''
        Output genotypes counts. 
        Format: run, time, data
        '''
        num_geno = np.copy(self.num_genotype)
        if self.args.ici:
            num_geno[self.ici_split_pos] += self.num_genotype_ici
        print(self.run, file=self.trajfile, end=",")
        print(self.t, file=self.trajfile, end=",")
        print(','.join(str(n) for n in num_geno), file=self.trajfile)

    def output_lineage(self):
        '''
        Output the lineage trajectories.
        run,t0,#_in_t1,...,#_in_tn
        '''
        index = 0
        for ind, lineage in enumerate(self.num_lineage_array):
            print(self.run, file=self.lineagefile, end=",")
            print(index, file=self.lineagefile, end=",")
            index += 1
            print(lineage[-1] > self.voc_th, file=self.lineagefile, end=",")
            print(self.lineage_t[ind], file=self.lineagefile, end=",")
            # add ici patients
            if self.args.ici and np.isin(ind, self.lineage_array_ici_pos):
                # if only ici patients and no secondary cases
                if np.sum(lineage) == 0:
                    lineage[-1] = 1
                else:
                    # add ici itself
                    lineage = np.trim_zeros(lineage, trim='f')
                    lineage += 1
                    # first time point already adds 1
                    lineage[0] -= 1
            print(",".join(str(lin)
                  for lin in np.trim_zeros(lineage, trim='f')), file=self.lineagefile)

    def output_voc(self):
        '''
        Output the time of VOC occurance (freq > 1/s).
        run,t1(time to become a voc)...
        '''
        # calculate voc emerge time
        start_time = np.min(self.lineage_t)
        voc_index = np.where(
            self.num_genotype_lineage[self.dms-1:] > self.voc_th)[0]
        voc_t = []
        for i in voc_index:
            index = np.where(self.num_lineage_array[i] > self.voc_th)
            voc_t.append(start_time + index[0][0])
        # output voc time
        print(self.run, file=self.vocfile, end=",")
        voc_t = np.sort(voc_t)
        print(",".join(str(i) for i in voc_t), file=self.vocfile)

    def output_ici(self):
        '''
        Output the index of ici VOC.
        run,i1 (indexes of vocs are from ici)...
        '''
        print(self.run, file=self.icifile, end="")
        if self.num_genotype_ici[-1] == 0:
            print("", file=self.icifile)
            return
        print(",", file=self.icifile, end="")
        # output ici lineage index
        print(",".join(str(i)
              for i in self.lineage_array_ici_pos), file=self.icifile)

    def output_add_header(self):
        '''
        Add header to trajfile, lineagefile and vocfile (and icifile).
        Close files.
        '''
        self.trajfile.close()
        # header for trajectories file
        header = ["run", "t"]
        for genotype in self.genotypes:
            genotype = np.where(genotype == -1, 0, genotype)
            header.append("".join(str(i) for i in genotype))
        header = ",".join(i for i in header)
        self.file_add_header(self.args.outpath + ".traj", header, 0)
        # output log file
        if self.args.log:
            self.logfile.close()
        if self.args.lineage:
            # header for lineage and voc file
            self.lineagefile.close()
            header = "run,index,voc,t_emerge,"
            self.file_add_header(self.args.outpath + ".lineage",
                                 header, self.lineage_time_max, "t")
            self.vocfile.close()
            header = "run,"
            # at least 2 vocs, to calculate deltaT
            self.file_add_header(self.args.outpath + ".voc",
                                 header, np.max([self.num_voc_max, 2]), "t")
            if self.args.ici:
                self.icifile.close()
                header = "run,"
                self.file_add_header(
                    self.args.outpath + ".ici", header, np.max([self.num_voc_ici_max, 2]), "i")

    def file_add_header(self, file_name, header, count, pre="L"):
        '''
        Add header and lineage/voc names to file.
        count: lineage/voc count
        '''
        vocs = np.arange(count).astype(np.uint16) + 1
        header += ",".join(pre+str(i) for i in vocs)
        header += "\n"
        self.prepend_line(file_name, header)

    def prepend_line(self, file_name, line):
        '''
        insert file with line.
        '''
        with open(file_name, "r+") as file:
            contents = file.readlines()
            contents.insert(0, line)  # new_string should end in a newline
            # readlines consumes the iterator, so we need to start over
            file.seek(0)
            file.writelines(contents)

    def traj_plot(self, run_index=0):
        '''
        Plot out trajectories of all genotypes.
        For the full mutants, it would plot all lineages (with plot = 'lineage')
        or only vocs (with plot = 'voc')
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # plot traj for all genotypes
        trajs = pd.read_csv(self.args.outpath + ".traj", sep=",")
        traj_run = trajs[trajs.run == run_index]
        t = traj_run.t
        for index, col in traj_run.iloc[:, 2:].iteritems():
            ax.plot(t, col)
        if self.args.lineage and self.args.plot != 'genotype':
            self.lineage_plot(ax, run_index)
        # legend for genotypes
        legend = []
        for genotype in self.genotypes:
            genotype = np.where(genotype == -1, 0, genotype)
            legend.append("".join(str(i) for i in genotype))
        ax.legend(legend, loc=0)
        ymax = np.max(self.nlist)
        plt.yscale("log", base=10)
        plt.ylim([1, 2.5*ymax])
        plt.yscale("log", base=10)
        plt.xlabel("T")
        plt.ylabel("Cases")
        plt.show()

    def lineage_plot(self, ax, run_index=0):
        '''
        Plot trajectories of lineages or vocs.
        '''
        lineage_trajs = pd.read_csv(self.args.outpath + ".lineage", sep=",")
        voc_t = pd.read_csv(self.args.outpath + ".voc",
                            sep=",").replace(np.nan, -1, regex=True)
        # select row to plot
        if self.args.plot == 'voc':
            lineages_to_plot = lineage_trajs[(lineage_trajs.voc == True) & (
                lineage_trajs.run == run_index)]
        if self.args.plot == 'lineage':
            lineages_to_plot = lineage_trajs[lineage_trajs.run == run_index]
        # quit when no lineage to plot
        if lineages_to_plot.shape[0] == 0:
            return
        # plot each row
        for index, row in lineages_to_plot.iloc[:, 2:].iterrows():
            t0 = row[0]
            lineage = row[1:]
            t = np.arange(len(lineage)) + t0
            ax.plot(t, lineage)
        # voc time
        if voc_t.shape[0] == 0:
            return
        voc_text = voc_t.to_numpy()[run_index, 1:]
        voc_text = voc_text[voc_text != -1]
        voc_text = "VOC time: " + ",".join(str(int(i)) for i in voc_text)
        ymax = np.max(self.nlist)
        plt.text(0, 1.1*ymax, voc_text, fontsize='large')

    def check(self, eps=1e-15):
        '''
        Check if the arrays are reasonable. Output results to log file.
        '''
        if self.t == 0:
            return True
        status = 0
        frequency = self.freq_genotype
        if self.args.lineage:
            frequency = self.freq_genotype_lineage
        if self.num_genotype[-1] > 0 and np.sum(self.prvs_gen[-self.args.K-1:]) == 0:
            print("Warning: full mutant appear out of nothing at run = {}, t={}.".format(
                self.run, self.t), file=self.logfile)
            status = 1
        if np.abs(np.sum(frequency)-1) > eps or np.any(frequency < 0) or np.any(frequency > 1):
            print("Warning: improper genotype frequencies run = {}, t={}.".format(
                self.run, self.t), file=self.logfile)
            status = 2
        if np.any(np.logical_and(0 < frequency, frequency < 1/self.nlist[self.t])):
            print("Warning: small probability occurs run = {}, t={}.".format(
                self.run, self.t), file=self.logfile)
            status = 3
        if self.args.lineage:
            lineage_freq = np.sum(self.freq_genotype_lineage[self.dms-1:])
            freq = self.freq_genotype[-1]
            diff = lineage_freq - freq
            if np.abs(diff) > eps:
                print("Warning: improper lineage genotype frequencies {} vs {} diff {} at run = {}, t={}.".format(
                    lineage_freq, freq, diff, self.run, self.t), file=self.logfile)
                status = 4
            if int(self.num_lineage) != self.num_lineage_array.shape[0]:
                print("Warning: number of lineage is not correct ({} vs {}) at run = {}, t={}.".format(
                    int(self.num_lineage), self.num_lineage_array.shape[0], self.run, self.t), file=self.logfile)
                status = 5
        if self.args.ici:
            if self.num_genotype_ici[-1] != len(self.lineage_array_ici_pos):
                print("Warning: numbers of ici voc ar not correct (pos:{} vs num:{}) at run = {}, t={}.".format(
                    len(self.lineage_array_ici_pos), self.num_genotype_ici[-1], self.run, self.t), file=self.logfile)
                status = 6
        if status > 0:
            print("Previous generation: ", self.prvs_gen,
                  "\nSampling frequency: ", frequency,
                  "\nNew generation: ", self.num_genotype,
                  "\n", file=self.logfile)
            return False
        return True


if __name__ == "__main__":
    tik = time.time()
    pop = Population()
    pop.evolve()
    tok = time.time()
    print("Finished {}! Used time: {}s".format(pop.args.outpath[2:], tok-tik))
