# /usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import itertools as it
import time
import os


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
    if np.all(p > eps):  # No small probabilities, ordinary multinomial is safe.
        return np.random.multinomial(N, p)
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

def prepend_line(file_name, line):
    '''
    insert file with line.
    '''
    with open(file_name, "r+") as file:
        contents = file.readlines()
        contents.insert(0, line)  # new_string should end in a newline
        file.seek(0)  # readlines consumes the iterator, so we need to start over
        file.writelines(contents)

class Population:
    def __init__(self):
        # Get the parameters from the command line.
        parser = argparse.ArgumentParser()
        parser.add_argument("--N", type=int, default=10000,
                            help="Population size")
        parser.add_argument("--mut", type=float,
                            default=5e-5, help="mutation rate")
        parser.add_argument("--rec", type=float,
                            default=0, help="frequency of sex")
        parser.add_argument("--s", type=float, default=0.24,
                            help="advantage of triple mutant")
        parser.add_argument("--k", type=int, default=2,
                            help="mutations to valley crossing")
        parser.add_argument("--poptype", choices=['C', 'E', 'F'], default='F',
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
        parser.add_argument("--sampling_method", choices=['multinomial', 'dirichlet'], default='multinomial',
                            help="sampling method: multinomial or dirichlet multinomial sampling")
        # Options
        parser.add_argument("--lineage", action='store_true', default=True,
                            help="keep track of lineage. Recomend run = 1 and plot out.")
        parser.add_argument("--accumulative", action='store_true', default=False,
                            help="Each mutation has the same selection effect: s/k.")
        parser.add_argument('--plot', choices=['none', 'lineage', 'voc'], default='none',
                            help="plot trajectories of all genotypes (and lineages).\
                                    option 'voc' for only plot vocs instead of all lineages ")
        parser.add_argument('--log', action='store_true', default=False,
                            help="check for valid frequencies and number of genotypes. output to .log file.")

        # Pack all params into self.args.
        self.args = parser.parse_args()
        # Create data folder
        if not os.path.exists(self.args.outpath):
            os.makedirs(self.args.outpath)
        # create file names
        # k: genome size
        self.args.outpath += "K{}".format(self.args.k)
        # population size and type
        self.args.outpath += self.args.poptype
        if self.args.poptype == 'C':
            self.args.outpath += "N" + "{:1.1E}".format(self.args.N)
        elif self.args.poptype == 'E':
            self.args.outpath += "N" + "{:1.1E}".format(self.args.N) + "G" + "{:1.1E}".format(self.args.g)
        # mutations rate
        self.args.outpath += "M{:1.0E}".format(self.args.mut)
        if self.args.sampling_method == 'multinomial':
            self.args.outpath += "MN"
        elif self.args.sampling_method == 'dirichlet':
            self.args.outpath += "DM"
        if self.args.accumulative:
            self.args.outpath += "A"
        # Frequently use params

        # Form the population list nlist by population type. (constant, exponential or from file).
        # default constant
        self.nlist = np.ones(self.args.tmax).astype(int) * self.args.N 
        if self.args.poptype == 'F':
            self.nlist = np.loadtxt('casesD.csv', delimiter=',').astype(int)
            self.args.tmax = len(self.nlist)
        elif self.args.poptype == 'E':
            # growth rate as g.
            for i in range(self.args.tmax):
                self.nlist[i] *= pow(1 + self.args.g, i)

        # Generate genotypes: -1 as wildtype, 1 as mutant in a site.
        self.genotypes = np.asarray(
            list(it.product((-1, 1), repeat=self.args.k)))
        # Sort the genotypes, so that the last one is full mutant
        self.genotypes = self.genotypes[np.argsort(self.genotypes.sum(axis=1))]
        # Genotype dimension
        self.dms = pow(2, self.args.k)
        # add slot for new lineage.

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
            for i in np.arange(1, self.args.k):
                self.fit_genotype[self.args.k*(i-1)+1:self.args.k*i+1] += self.args.s/self.args.k

        # sampling methods
        if self.args.sampling_method == 'multinomial':
            self.sampling = multinomial
        elif self.args.sampling_method == 'dirichlet':
            self.sampling = dirichlet_multinomial
        # generation methods
        if self.args.lineage:
            self.generation = self.generation_lineage
            self.max_lineage_count = 0
        else:
            self.generation = self.generation_no_lineage
        # Initial random generator.
        np.random.seed(self.args.seed)
        # Open files.
        self.trajfile = open(self.args.outpath + ".traj", 'w')
        if self.args.log:
            self.logfile = open(self.args.outpath + ".log", 'w')
        # Record important params
        with open(self.args.outpath + '.params', 'w') as outfile:
            print("\n".join(["N = {:.3g}", "poptype = {}", "mu = {}", "r = {}", "s = {}",
                             "k = {}", "g = {}", "tstep = {}", "seed = {}"])
                  .format(self.args.N, self.args.poptype, self.args.mut, self.args.rec, self.args.s,
                          self.args.k, self.args.g, self.args.tstep, self.args.seed), file=outfile)

    def init_pop(self):
        '''
        Initial/Reset frequencies, number of genotypes adn lineages.
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
            self.freq_genotype_lineage = np.zeros(self.dms).astype(np.float64)
            self.freq_genotype_lineage[0] = 1.0
            self.lineage_count = 0
            self.num_voc = 0

    def mutation(self):
        '''
        Calculate changes in frequency based on mutation rate.
        Return frequency changes for mutate in and out.
        '''
        if self.args.mut == 0:
            return [0,0]
        # The first part is mutating to certain genotype, the latter one is mutating out.
        mutate_in = self.freq_genotype @ self.mutmat
        mutate_out = self.args.mut * self.args.k * self.freq_genotype
        self.freq_genotype += mutate_in - mutate_out
        return [mutate_in[-1], mutate_out[-1]]

    def recombination(self):
        '''
        Calculate changes in frequency based on recombination rate.
        Return frequency changes for recombine in and out.
        '''
        if self.args.rec == 0:
            return [0,0]
        # The first part is recombine to certain genotype, the latter one is recombine out.
        rec_in = self.args.rec/2 * self.freq_genotype @ (self.freq_genotype @ self.recmat)
        rec_out = self.args.rec/2 * self.freq_genotype
        self.freq_genotype[:] += rec_in - rec_out
        return [rec_in[-1], rec_out[-1]]

    def selection(self):
        '''
        Calculate changes in frequency based on selection.
        '''
        if self.args.s == 0:
            return 1
        selection = self.fit_genotype / (self.freq_genotype @ self.fit_genotype)
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
                if self.args.log:
                    self.check()
                if self.t % self.args.tstep == 0:
                    self.output(self.num_genotype, self.trajfile)
                # Consider full mutant is occupying the population
                # print(self.t, self.num_voc)
                if self.freq_genotype[-1] > 1/5 or (self.args.lineage and self.num_voc >= 10):
                    self.output(self.num_genotype, self.trajfile)
                    break
            # record max number of lineages
            if self.args.lineage:
                self.max_lineage_count = np.max([self.lineage_count, self.max_lineage_count])

        self.trajfile.flush()
        self.trajfile.close()
        # header for trajectories file
        header = "run,t,"
        for genotype in self.genotypes[:-1]:
            genotype = np.where(genotype == -1, 0, genotype)
            header += "".join(str(i) for i in genotype)
            header += ","
        self.file_add_header(self.args.outpath + ".traj", header, self.max_lineage_count)
        self.output_voc()
        # output log file
        if self.args.log:
            self.logfile.flush()
            self.logfile.close()
        # plots
        if self.args.plot != 'none':
            self.traj_plot()

    def generation_no_lineage(self):
            '''
            Update frequency and genotype counts.
            '''
            self.selection()
            self.mutation()
            self.recombination()
            self.prvs_gen = self.num_genotype
            # new sampling
            self.num_genotype = self.sampling(self.nlist[self.t], self.freq_genotype)
            self.freq_genotype = self.num_genotype / self.nlist[self.t]
            try:
                np.sum(self.freq_genotype_lineage)
            except:
                pass
            
    def generation_lineage(self):
        '''
        Update frequency and genotype counts. 
        Update lineage array and VOC records.
        '''

        selection = self.selection()
        [mutate_in, mutate_out] = self.mutation()
        [rec_in, rec_out] = self.recombination()
        # for non mutants
        self.freq_genotype_lineage[:self.dms-1] = self.freq_genotype[:self.dms-1]
        # if no mutant in current generation
        if np.sum(self.freq_genotype_lineage[self.dms-1:]) == 0:
            self.freq_genotype_lineage[-1] = self.freq_genotype[-1]
            mutant = False
        else:
            mutant = True
            # previous exsistent mutants
            self.freq_genotype_lineage[self.dms-1:-1] *= selection
            lineage_portion = self.freq_genotype_lineage[self.dms-1:-1]/np.sum(
                self.freq_genotype_lineage[self.dms-1:-1])
            self.freq_genotype_lineage[self.dms-1:-
                                       1] -= (mutate_out + rec_out)*lineage_portion
            # probability of new mutant
            self.freq_genotype_lineage[-1] = mutate_in + rec_in

        scalar = 1
        if self.args.sampling_method == "dirichlet":
            # if mutant is present
            r0 = 2.8
            k = 0.1
            n = self.nlist[self.t]
            if mutant:
                r0 *= 1.38
            scalar = (-r0+np.sqrt(4*k*k*n+4*k*n*r0+r0*r0))/(2*(k+r0))
        self.prvs_gen = self.num_genotype
        # sampling process
        # print(self.freq_genotype_lineage)
        self.num_genotype = self.sampling(self.nlist[self.t], self.freq_genotype_lineage, scalar)
        new_lineage = self.num_genotype[-1]
        # detect new lineage
        if new_lineage > 0:
            self.lineage_count += new_lineage
            self.num_genotype = np.append(self.num_genotype[:-1], [1]*new_lineage+[0]).astype(np.uint64)
        # new frequency after the sampling
        self.freq_genotype_lineage = self.num_genotype / self.nlist[self.t]
        # update genotype frequency
        self.freq_genotype[:-1] = self.freq_genotype_lineage[:self.dms-1]
        self.freq_genotype[-1] = np.sum(self.freq_genotype_lineage[self.dms-1:])
        # record voc
        self.num_voc = np.count_nonzero(self.num_genotype[self.dms-1:] > 1/self.args.s)

    def output(self, data, filepath):
        '''
        Output data. 
        Format: run, time, data
        '''
        print(self.run, file=filepath, end=",")
        print(self.t, file=filepath, end=",")
        print(','.join(str(n) for n in data), file=filepath)

    def file_add_header(self, file_name, header, count):
        '''
        Add header and lineage/voc names to file.
        count: lineage/voc count
        '''
        vocs = np.arange(count).astype(np.uint16) + 1
        header += ",".join("L"+str(i) for i in vocs)
        header += "\n"
        prepend_line(file_name, header)
    
    def output_voc(self):
        '''
        Calcultate the time of VOC occurance (freq > 1/s).
        Read from traj file.
        '''
        lineagefile = open(self.args.outpath + ".lineage", 'w')
        self.trajs = pd.read_csv(self.args.outpath + ".traj", sep=",")
        # lineages['Mutants'] = lineages.sum(axis=1)
        voc_th = 1/self.args.s
        voc_t = []
        max_voc = 0
        for run_index in self.trajs.run.unique():
            traj_run = self.trajs[self.trajs.run == run_index]
            # get lineage cols
            lineages = traj_run.iloc[:, self.dms+1:]
            vocs = lineages.loc[:, lineages.iloc[-1] > voc_th]
            voc_t_run = []
            t0 = vocs.index[0]
            for index, col in vocs.iteritems():
                voc_t_run.append(col.index[col>voc_th].tolist()[0] - t0)
            voc_t_run = np.sort(voc_t_run)
            print(run_index, file=lineagefile, end=",")
            print(','.join(str(n) for n in voc_t_run), file=lineagefile)
            max_voc = np.max([max_voc, len(voc_t_run)])
            voc_t.append(np.sort(voc_t_run))
        lineagefile.close()
        # add header
        header = "run,"  
        self.file_add_header(self.args.outpath + ".lineage", header, max_voc)
    
    def traj_plot(self, run_index=0):
        '''
        Plot out trajectories of all genotypes.
        For the full mutants, it would plot all lineages (with plot = 'lineage')
        or only vocs (with plot = 'voc')
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if self.args.plot == 'voc':
            self.voc_plot(ax, run_index)
        if self.args.plot == 'lineage':
            self.lineage_plot(ax, run_index)
        # legend for genotypes (without full mutant)
        legend = []
        for genotype in self.genotypes[:-1]:
            genotype = np.where(genotype == -1, 0, genotype)
            legend.append("".join(str(i) for i in genotype))
        ax.legend(legend, loc=0)

        plt.yscale("log", base=10)
        ymax = np.max(self.nlist)
        plt.ylim([1, 2.5*ymax])
        plt.yscale("log", base=10)
        plt.xlabel("T")
        plt.ylabel("Cases")
        plt.show()

    def lineage_plot(self, ax, run_index=0):
        '''
        Plot trajectories of all genotypes and lineages.
        '''
        traj_run = self.trajs[self.trajs.run == run_index]
        t = traj_run.t
        for index, row in traj_run.iloc[:,2:].iteritems():
            ax.plot(t, row)
    
    def voc_plot(self, ax, run_index=0):
        '''
        Plot trajectories of all genotypes and VOCs.
        '''
        traj_run = self.trajs[self.trajs.run == run_index]
        voc_th = 1/self.args.s
        # select out col of vocs
        cond = traj_run.iloc[-1] > voc_th
        # do not plot time and run col
        cond[0] = False
        cond[1] = False
        #  plot all mutant genotypes
        for i in np.arange(2, self.dms+1):
            cond[i]=True
        traj_run.plot(x='t', y=list(traj_run.loc[:,cond]), ax=ax, legend=False)
        # read lineage emerging times
        lineage_t0 = pd.read_csv(self.args.outpath + ".lineage", sep=",")
        lineage_t0_run = lineage_t0[lineage_t0.run == run_index]
        lineage_t0_run = lineage_t0_run.loc[:,lineage_t0_run.iloc[0] >= 0].iloc[:,1:]
        voc_text  = 'Voc T: ' + ','.join(str(int(lin)) for lin in lineage_t0_run.to_numpy()[0])
        ymax = np.max(self.nlist)
        plt.text(0, 1.1*ymax, voc_text, fontsize='large')

    def check(self, eps=1e-15):
        if self.t == 0:
            return True
        status = 0
        if self.args.lineage:
            frequency = self.freq_genotype_lineage
        else:
            frequency = self.freq_genotype
        if self.num_genotype[-1] > 0 and np.sum(self.prvs_gen[-self.args.k-1:]) == 0:
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

        if status > 0:
            print("Previous generation: ", self.prvs_gen,
                  "\nPre-sampling number of genotypes: ", ' '.join(
                      str(x) for x in self.prvs_gen),
                  "\nSampling frequency: ", ' '.join(str(x)
                                                         for x in frequency),
                  "\nNew generation: ", self.num_genotype, "\n", file=self.logfile)
            return False
        return True


if __name__ == "__main__":
    tik = time.time()
    pop = Population()
    pop.evolve()
    tok = time.time()
    print("Finished {}! Used time: {}s".format(pop.args.outpath[2:], tok-tik))
