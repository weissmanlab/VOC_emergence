# /usr/bin/env python3
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
    if np.all(p > eps):  # No small probabilities, ordinary multinomial is safe.
        return np.random.multinomial(N, p)
    else:  # Handle the small probabilities with Poisson draws, then do multinomial for the rest.

        n = np.zeros(len(p), dtype=np.uint64)
        ismall = np.nonzero(p <= eps)[0]
        ilarge = np.nonzero(p > eps)[0]
        for i in ismall:
            n[i] = np.random.poisson(N * p[i])
        n_large = np.random.multinomial(
            N - np.sum(n), p[ilarge] / np.sum(p[ilarge]))
        for i, nl_i in enumerate(n_large):
            n[ilarge[i]] = nl_i
        return n


class population:
    def __init__(self):
        # Get the parameters from the command line.
        parser = argparse.ArgumentParser()
        parser.add_argument("--N", type=int, default=10000,
                            help="Population size")
        parser.add_argument("--mut", type=float,
                            default=1e-3, help="mutation rate")
        parser.add_argument("--rec", type=float,
                            default=0, help="frequency of sex")
        parser.add_argument("--s", type=float, default=0.25,
                            help="advantage of triple mutant")
        parser.add_argument("--k", type=int, default=2,
                            help="mutations to valley crossing")
        parser.add_argument("--poptype", choices=['C', 'E', 'F'], default='C',
                            help="population type: C (Constant), E (Exponential), or F (Read from file n.csv)")
        parser.add_argument("--runs", type=int, default=1,
                            help="repeat time for simulation")
        parser.add_argument("--out", default='./',
                            help="prefix for output files")
        parser.add_argument("--g", type=float, default=1,
                            help="growth rate for exponential population")
        parser.add_argument("--tmax", type=int, default=100,
                            help="max number of generations to run for")
        parser.add_argument("--tstep", type=int, default=1,
                            help="record the population state every tstep generations")
        parser.add_argument("--seed", type=int, default=None,
                            help="seed for random number generator")
        parser.add_argument("--lineage", action='store_true', default=True,
                            help="keep track of lineage. Recomend run = 1 and plot out.")
        parser.add_argument("--accumulative", action='store_true', default=False,
                            help="Each mutation has the same selection effect: s/k.")
        parser.add_argument('--plot', action='store_true', default=True,
                            help="plot trajectories (and lineages)")

        # Pack all params into self.args.
        self.args = parser.parse_args()
        # Frequently use params
        k = self.args.k
        s = self.args.s
        # Record important params
        with open(self.args.out + 'params.txt', 'w') as outfile:
            print("\n".join(["N = {:.3g}", "poptype = {}", "mu = {}", "r = {}", "s = {}",
                             "k = {}", "g = {}", "tstep = {}", "seed = {}"])
                  .format(self.args.N, self.args.poptype, self.args.mut, self.args.rec, self.args.s,
                          self.args.k, self.args.g, self.args.tstep, self.args.seed), file=outfile)

        # Form the population list nlist by population type. (constant, exponential or from file).
        if self.args.poptype == 'F':
            self.nlist = np.loadtxt('casesD.csv', delimiter=',').astype(int)
            self.args.tmax = len(self.nlist)
        if self.args.poptype == 'C':
            self.nlist = np.ones(self.args.tmax).astype(int) * self.args.N
        if self.args.poptype == 'E':
            # There is a growth rate as g.
            self.nlist = np.ones(self.args.tmax).astype(int) * self.args.N
            for i in range(self.args.tmax):
                self.nlist[i] *= pow(self.args.g, i)
            self.nlist = self.nlist.astype(int)  # force it to be int.

        # Generate genotypes: -1 as wildtype, 1 as mutant in a site.
        self.genotypes = np.asarray(
            list(it.product((-1, 1), repeat=k)))
        # Sort the genotypes, so that the last one is full mutant
        self.genotypes = self.genotypes[np.argsort(self.genotypes.sum(axis=1))]
        # Genotype dimension
        self.dms = pow(2, k)

        # form mutation matrix: m_i,j as mutation from i to j.
        self.mutmat = np.zeros((self.dms, self.dms))
        for i in range(self.dms):
            for j in range(self.dms):
                # If the hamming disntance of two genotype is 1, mutation probability is mut.
                if np.count_nonzero(self.genotypes[i] - self.genotypes[j]) == 1:
                    self.mutmat[i, j] = self.args.mut

        # Form recombination matrix: r_i,j as rec from i to j.
        self.recmat = np.array([[[mendel_prob(x, y, z) for z in self.genotypes]
                               for y in self.genotypes] for x in self.genotypes])

        # Form fitness array: s is the breeding advantage.
        self.fit_genotype = np.ones(self.dms)
        self.fit_genotype[-1] += s
        if self.args.accumulative:
            # Each mutations has s/k increase in fitness.
            for i in np.arange(1, k):
                self.fit_genotype[k*(i-1)+1:k*i+1] += s/k
        # print(self.fit_genotype)

        # Initial random generator.
        np.random.seed(self.args.seed)
        # Open files.
        self.outfile = open(self.args.out + "out.txt", 'w')
        self.trafile = open(self.args.out + "trajectory.txt", 'w')
        self.logfile = open(self.args.out + "log.txt", 'w')
        if self.args.lineage:
            self.lineagefile = open(self.args.out + "lineage.txt", 'w')

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
        if self.args.lineage:
            self.lincount = 0
            self.lindel = 0
            self.voct = np.array([])
            # indicator is 1 when a new lineage is occur.
            # Delete the last lineage record after selection, to avoid record twice in the same gen.
            self.lineageindicator = False
            self.lineagearray = []

    def mutation(self):
        '''
        Calculate changes in frequency based on mutation rate.

        Lineage may be recorded.
        '''
        if self.args.mut > 0:
            # The first part is mutating to certain genotype, the latter one is mutating out.
            self.freq_genotype += self.freq_genotype @ self.mutmat - \
                self.args.mut * self.args.k * self.freq_genotype
            # If need to record lineage.
            if self.args.lineage:
                self.lineagecount()

    def lineagecount(self):
        '''
        Record lineage. Consider a new lineage occurs when mutates a full mutant.
        '''
        self.prvs_gen = self.num_genotype
        totpop = np.sum(self.num_genotype)
        self.num_genotype = sampling(
            totpop, self.freq_genotype, eps=1/self.args.N)
        self.freq_genotype = self.num_genotype / totpop
        # Compare prev and new mutant.
        diff = int(self.num_genotype[-1] > self.prvs_gen[-1])
        if diff > 0:
            self.lincount += 1
            pos = len(self.lineagearray)
            self.lineagearray.append(np.array([], dtype=int))
            self.lineagearray[pos] = np.append(self.lineagearray[pos], self.t)
            self.lineageindicator = True
            # It should be deleted after selection.
            # Use indicator to delete it.
            self.lineagearray[pos] = np.append(self.lineagearray[pos], diff)

    def recombination(self):
        '''
        Calculate changes in frequency based on recombination rate.
        '''
        if self.args.rec > 0:
            self.freq_genotype += -self.args.rec/2 * self.freq_genotype + \
                self.args.rec/2 * \
                self.freq_genotype @ (self.freq_genotype @ self.recmat)

    def selection(self):
        '''
        Calculate changes in frequency based on selection.
        '''
        self.freq_genotype *= self.fit_genotype / \
            (self.freq_genotype @ self.fit_genotype)

    def evolve(self):
        '''
        Evolution of designed population.
        '''
        for self.run in np.arange(self.args.runs):
            # Initial population
            self.initpop()
            for self.t in np.arange(self.args.tmax):
                # update mutant counts
                self.generation()
                self.check()

                if self.t % self.args.tstep == 0:
                    self.output(self.num_genotype, self.trafile)
                # Consider full mutant is occupying the population
                if len(self.voct) > 10 or self.freq_genotype[-1] > 1/5:
                    self.output(self.num_genotype, self.trafile)
                    if self.args.lineage:
                        self.VOCoutput()
                    break

        self.trafile.flush()
        self.trafile.close()
        self.outfile.flush()
        self.outfile.close()
        self.logfile.flush()
        self.logfile.close()

        if self.args.plot == True:
            # Plot wt and mutants
            self.trajplot("trajectory.txt", 0)
            if self.args.lineage:
                self.lineagefile.flush()
                self.lineagefile.close()
                # print("Total occurance = ", self.lincount)
                # print("Total del = ", self.lindel)
                # print("Size of lineage = ", len(self.lineagearray))
                self.VOCplot()
            plt.show()

    def generation(self):
        '''
        Update frequency and genotype counts. 
        Update lineage array and VOC records.
        '''
        self.mutation()
        self.recombination()
        self.selection()
        self.prvs_gen = self.num_genotype
        self.num_genotype = sampling(self.nlist[self.t], self.freq_genotype)
        self.freq_genotype = self.num_genotype / self.nlist[self.t]
        # Sample lineage population based on their freq. Add up to full mutant.

        if self.args.lineage:
            prevmut = np.array([sublist[-1]
                                for sublist in self.lineagearray])
            totmut = self.num_genotype[-1]
            totpremut = np.sum(prevmut)
            # If there are mutants
            if totpremut > 0:
                newmut = sampling(
                    totmut, prevmut/totpremut, eps=1/self.args.N)
                # Delete the new added mutant number, to avoid double record.
                if self.lineageindicator:
                    self.lineagearray[-1] = np.delete(
                        self.lineagearray[-1], -1)
                    self.lineageindicator = False
                # Update lineagearray
                deleted = 0
                for ind, numut in enumerate(newmut):
                    if numut == 0:
                        del(self.lineagearray[ind-deleted])
                        self.lindel += 1
                        deleted += 1
                    else:
                        self.lineagearray[ind-deleted] = np.append(
                            self.lineagearray[ind-deleted], int(numut))
                # Update the recording of vocs
                self.VOCt()

    def VOCt(self):
        '''
        Calcultate the time of VOC occurance (freq > 1/s).
        '''
        self.voct = np.array([])
        # check every lineage
        for lineage in self.lineagearray:
            # if it is a voc
            pos = np.argwhere(lineage[1:] > 1/self.args.s)
            if len(pos) > 0:
                t0 = lineage[0]
                t0 += pos[0][0]
                # record the time when it becomes a voc
                self.voct = np.append(self.voct, int(t0))


    def VOCoutput(self):
        '''
        Output VOC data.
        Each line is a lineage. First number is the time of occurance.
        '''
        # output every lineage
        print('run = {}'.format(self.run), file=self.lineagefile)
        for lineage in self.lineagearray:
            print(','.join(str(lin) for lin in lineage), file=self.lineagefile)
        # calculate the time difference of emerging vocs.
        self.voctext = ""
        if len(self.voct) >= 2:
            self.voct = np.sort(self.voct).astype(int)
            t0 = self.voct[:-1]
            t1 = self.voct[1:]
            # only output 10 vocs at max
            voccount = 10
            for s in (np.array(t1) - np.array(t0)):
                self.voctext += str(s) + ","
                voccount -= 1
                if voccount <= 0:
                    break
            print('run = {}'.format(self.run), file=self.outfile)
            print(str(self.voct[0])+","+self.voctext, file=self.outfile)

    def VOCplot(self):
        '''
        Plot VOCs. Only plot the last run.
        '''

        for lineage in self.lineagearray:
            # time appear
            tAppear = lineage[0]
            # lineage count in every generation
            y = lineage[1:]
            # if it is a voc in the future
            if len(np.argwhere(y > 1/self.args.s)) > 0:
                # full time series
                tott = len(y)
                x = [tAppear+i for i in range(tott)]
                plt.plot(x, y, 'o-')
        # output delta t
        if len(self.voct) >= 2:
            ymax = np.max(self.nlist)
            plt.text(0, ymax/1.5, "Delta t: " + self.voctext, fontsize='large')

    def output(self, data, filepath):
        '''
        Output data.
        '''
        if self.t == 0:
            print('run = {}'.format(self.run), file=filepath)
        # lieagefile output differently: it does not need to output current time
        print(self.t, file=filepath, end=",")
        print(','.join(str(n) for n in data), file=filepath)

    def trajplot(self, filename, pos=-1):
        '''
        Plot mutants.
        pos -- Index of plot data. Output full line when equals 0.
        '''
        plotfile = open(filename, 'r')
        x = np.array([])
        y = np.array([])
        plotindex = self.run + 1
        # get to the data of the final run
        while True:
            line = plotfile.readline()
            if line[0] == "r":
                plotindex -= 1
            if plotindex == 0:
                break
        # read data
        while True:
            line = plotfile.readline()
            # if it reaches the end
            if len(line) == 0:
                # y is 2d array. Each row is a traj
                y = np.stack(y, axis=-1)
                # plot a specific traj
                if pos != 0:
                    plt.plot(x, y[pos], 'o-')
                # plot all trajs
                else:
                    for traj in y:
                        plt.plot(x, traj, linestyle="-")
                break
            # Omit the newline and commas.
            lineitems = line.split('\n')[0].split(',')
            x = np.append(x, int(lineitems[0]))
            lineint = np.array([int(i) for i in lineitems])
            if len(y) == 0:
                # 0.0001 is to ensure log function is valid
                y = np.append(y, lineint[1:] + 0.0001)
            else:
                y = np.vstack((y, lineint[1:] + 0.0001))

        plt.yscale("log", base=10)
        plt.xlabel("T")
        plt.ylabel("Mutant")
        ymax = np.max(self.nlist)
        plt.ylim([0.8, 2.5*ymax])
        # Theoretical exp line
        expmut = np.exp(self.args.s*x)
        plt.plot(x, expmut, color='black', markersize=0, linestyle='-')

    def check(self):
        if self.t == 0:
            return True
        status = 0
        if self.num_genotype[-1] > 0 and np.sum(self.prvs_gen[-self.args.k-1:]) == 0:
            print("Warning: full mutant appear out of nothing at run = {}, t={}.".format(
                self.run, self.t), file=self.logfile)
            status = 1
        if np.abs(np.sum(self.freq_genotype)-1) > 1e-6 or np.any(self.freq_genotype < 0) or np.any(self.freq_genotype > 1):
            print("Warning: improper genotype frequencies run = {}, t={}.".format(
                self.run, self.t), file=self.logfile)
            status = 2
        if np.any(np.logical_and(0 < self.freq_genotype, self.freq_genotype < 1/self.args.N)):
            print("Warning: small probability occurs run = {}, t={}.".format(
                self.run, self.t), file=self.logfile)
            status = 3
        if status > 0:
            print("Previous generation: ", self.prvs_gen,
                  "\nPre-sampling number of genotypes: ", ' '.join(
                      str(x) for x in self.prvs_gen),
                  "\nPre-sampling frequency: ", ' '.join(str(x)
                                                         for x in self.freq_genotype),
                  "\nNew generation: ", self.num_genotype, "\n", file=self.logfile)
            return False
        return True


if __name__ == "__main__":
    pop = population()
    pop.evolve()
