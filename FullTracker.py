import numpy as np
import matplotlib.pyplot as plt
import bisect
from collections import defaultdict
import pickle
from useful_functions import incomplete_moment, add_incom_to_plot
from scipy.sparse import lil_matrix
import gzip


class GeneralSimClass(object):
    """
    Common functions for all simulation algorithms.
    Functions for setting up simulations and for plotting results
    """
    def __init__(self, parameters):
        # Get attributes from the parameters
        self.total_pop = parameters.initial_cells
        self.initial_size_array = parameters.initial_size_array
        self.initial_clones = len(self.initial_size_array)
        self.mutation_rate = parameters.mutation_rate
        self.mutation_generator = parameters.mutation_generator
        self.division_rate = parameters.division_rate
        self.max_time = parameters.max_time
        self.samples = parameters.samples
        self.times = parameters.times
        self.sample_points = parameters.sample_points
        self.non_zero_calc = parameters.non_zero_calc

        self.parameters = parameters

        self.sim_length = len(self.times)

        self.clones_array = None  # Will store the information about the clones. One row per clone.
        # A clone here will contain exactly the same combination of mutations.
        self.population_array = None  # Will store the clone sizes. One row per clone. One column per sample.

        # Include indices here for later use. These are the columns of self.clones_array
        self.id_idx = 0  # Unique integer id for each clone. Int.
        self.label_idx = 1  # The type of the clone. Inherited label does not change. Int. Represents GFP or similar.
        self.fitness_idx = 2  # The fitness of the clone. Float.
        self.generation_born_idx = 3  # The sample the clone first appeared in.  Int.
        self.parent_idx = 4  # The id of the clone that this clone emerged from.  Int.

        self.s_muts = set()  # Synonymous mutations. Indices of the first clone they appear in
        self.ns_muts = set()  # Non-synonymous mutations. Indices of the first clone they appear in

        # We can calculate the number of mutations added in each generation beforehand and make the arrays the correct
        # size. This should speed things up for long, mutation heavy simulations.
        self.precalculate_mutations()
        self.total_clone_count = self.initial_clones + self.new_mutation_count
        if parameters.progress:
            print(self.new_mutation_count, 'mutations to add', flush=True)

        # Make the arrays the correct size.
        self.init_arrays(parameters.label_array, parameters.fitness_array)

        self.next_mutation_index = self.initial_clones  # Keeping track of how many mutations added

        self.clone_descendants = defaultdict(set)  # Keys = clone id. Values = set of all clones descended from the key
        self.clone_ancestors = defaultdict(set)  # Keys = clone id. Values = set of all clones the key descended from.
        # For convenience of how these are later used, a clone is counted as its own ancestor and descendant
        for i in range(self.total_clone_count):
            self.clone_descendants[i].add(i)
            self.clone_ancestors[i].add(i)

        # Details for plotting
        self.figsize = parameters.figsize
        self.plot_order = []  # Used to generate the plot of clones
        self.descendant_counts = {}
        self.progress = parameters.progress  # Prints update every n samples
        self.plot_idx = 1  # Keeping track of x-coordinate of the plot
        self.i = 0
        self.colours = None

        # Stores the sizes of clones containing particular mutants.
        self.mutant_clone_array = None

        # For storing temporary results during the simulation.
        self.tmp_store = parameters.tmp_store
        self.store_rotation = 0  # Alternates between two tmp stores (0, 1) in case error occurs during pickle dump.
        self.is_sparse = True  # Is the population array stored in scipy.sparse.lil_matrix (True) or numpy array (False)
        self.finished = False
        self.random_state = None  # For storing the state of the random sequence for continuing

    # Functions for running the simulations
    def precalculate_mutations(self):
        # To be overwritten. Will calculate the number and timing of all mutations in the simulation
        self.new_mutation_count = 0

    def init_arrays(self, labels_array, fitness_array):
        """Defines self.clones_array and self.population_array.
        Fills the self.clones_array with any information given about the intial cells.
        """
        self.clones_array = np.zeros((self.total_clone_count, 5))
        self.clones_array[:, self.id_idx] = np.arange(len(self.clones_array))  # Give clone an identifier

        if labels_array is None:
            labels_array = 0
        self.clones_array[:self.initial_clones, self.label_idx] = labels_array  # Give each intial cell a type

        if fitness_array is None:
            fitness_array = 1
        self.clones_array[:self.initial_clones, self.fitness_idx] = fitness_array  # Give each initial cell a fitness

        self.clones_array[:self.initial_clones, self.generation_born_idx] = 0
        self.clones_array[:self.initial_clones, self.parent_idx] = -1

        self.population_array = lil_matrix((self.total_clone_count,
                                            self.sim_length))  # Will store the population counts

        # Start with the initial_quantities
        self.population_array[:self.initial_clones, 0] = self.initial_size_array.reshape(len(self.initial_size_array),
                                                                                         1)

    def continue_sim(self):
        if self.random_state is not None:
            np.random.set_state(self.random_state)
        self.run_sim(continue_sim=True)

    def run_sim(self, continue_sim=False):
        # Functions which runs any of the simulation types.
        # self.sim_step will include the differences between the methods.
        # Each step can be a generation (Wright-Fisher) or a single birth-death-mutation event (Moran).

        # In it is possible for a clone to not survive until the next sample after being created by a mutation.
        # This will leave an all zeros row in the population array.
        if self.population_array[:, 1].sum() > 0:
            # Not the first time it has been run
            if self.finished:
                print('Simulation already run')
            elif continue_sim:
                print('Continuing from step', self.i)
            else:
                print('Simulation already started but incomplete')
                return

        current_population = self.population_array[:, self.plot_idx - 1].copy()
        current_population = current_population.T.toarray()[0]
        current_population = current_population.astype(int)
        if self.non_zero_calc:
            non_zero_clones = np.where(current_population > 0)[0]
            current_population = current_population[non_zero_clones]
        else:
            non_zero_clones = None

        if self.progress:
            print('Steps completed:')

        while self.plot_idx < self.sim_length:
            current_population, non_zero_clones = self.sim_step(self.i, current_population,
                                               non_zero_clones)  # Run step of the simulation
            self.i += 1
            self.take_sample(self.i, current_population, non_zero_clones)  # Record the current state

        if self.progress:
            print('Finished', self.i, 'steps')

        self.finish_up()
        self.finished = True

        # self.proportional_populations = self.population_array / self.population_array.sum(axis=0)

    def take_sample(self, i, current_population, non_zero_clones):
        """
        Record the results at the point the simulation is up to.
        Report progress if required
        :param i:
        :param current_population:
        :return:
        """
        if i == self.sample_points[self.plot_idx]:  # Regularly take a sample for the plot
            if self.non_zero_calc:
                self.population_array[non_zero_clones, self.plot_idx] = current_population.reshape(
                    len(current_population), 1)
            else:
                self.population_array[:, self.plot_idx] = current_population.reshape(len(current_population), 1)
            self.plot_idx += 1
            if self.tmp_store is not None:  # Store current state of the simulation.
                if self.store_rotation == 0:
                    self.pickle_dump(self.tmp_store)
                    self.store_rotation = 1
                else:
                    self.pickle_dump(self.tmp_store + '1')
                    self.store_rotation = 0

        if self.progress:
            if i % self.progress == 0:
                print(i, end=', ', flush=True)

    def pickle_dump(self, filename):
        self.random_state = np.random.get_state()
        with gzip.open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=4)

    def finish_up(self):
        """
        Some of the simulations may required some tidying up at the end,
        for example, removing unused rows in the arrays.
        :return:
        """
        pass

    def unsparsify(self):
        if self.is_sparse:
            self.population_array = self.population_array.toarray()  # Convert back to numpy array
        self.is_sparse = False

    def sim_step(self, i, current_population, non_zero_clones):  # Overwrite
        return current_population, non_zero_clones

    def draw_mutation_and_add_to_array(self, parent_idx):
        """Select a fitness for the new mutation and the cell in which the mutation occurs
        parent_idx = the id of the clone in which the mutation occurs
        """
        selected_clone = self.clones_array[parent_idx]
        new_type = selected_clone[self.label_idx]  # Is the new clone labelled or not
        old_fitness = selected_clone[self.fitness_idx]

        # Get a fitness value for the new clone.
        new_fitness, synonymous = self.mutation_generator.get_new_fitness(old_fitness)
        if synonymous:
            self.s_muts.add(self.next_mutation_index)
        else:
            self.ns_muts.add(self.next_mutation_index)

        # Add the new clone to the clone_array
        self.clones_array[self.next_mutation_index] = self.next_mutation_index, new_type, new_fitness, \
                                                      self.plot_idx, parent_idx

        # Update ancestors and descendants. Note, all clones already have themselves as ancestor and descendant.
        self.clone_ancestors[self.next_mutation_index].update(self.clone_ancestors[parent_idx])
        for a in self.clone_ancestors[parent_idx]:
            self.clone_descendants[a].add(self.next_mutation_index)

        self.next_mutation_index += 1

    def track_mutations(self, selection='all'):
        """
        Get a dictionary of the clones which contain each mutation.
        :param selection: 'all', 'ns', 's'. All/non-synonymous only/synonymous only.
        :return: Dict. Key: mutatation id (id of first clone which contains the mutation),
        value: set of clone ids which contain that mutation
        """
        if selection == 's':
            mutant_clones = {k: self.clone_descendants[k] for k in self.s_muts}
        elif selection == 'ns':
            mutant_clones = {k: self.clone_descendants[k] for k in self.ns_muts}
        elif selection == 'all':
            mutant_clones = {k: self.clone_descendants[k] for k in range(len(self.clones_array))}
        elif selection == 'mutations':
            mutant_clones = {k: self.clone_descendants[k] for k in self.ns_muts.union(self.s_muts)}
        else:
            print("Please select from 'all', 's', 'ns', or 'label'")
            raise ValueError("Please select from 'all', 's', 'ns' or 'label'")

        return mutant_clones

    def create_mutant_clone_array(self):
        """
        Create an array with the clone sizes for each mutant across the entire simulation.
        The populations will usually add up to more than the total since many clones will have multiple mutations
        """
        mutant_clones = self.track_mutations(selection='all')
        self.mutant_clone_array = np.array([np.atleast_1d(self.population_array[list(mutant_clones[mutant])].sum(axis=0))
                                                       for mutant in range(len(self.clones_array))])


    def get_mutant_clone_sizes(self, t=None, selection='mutations', index_given=False, non_zero_only=False):
        """
        Get an array of mutant clone sizes at a particular time
        WARNING: This may not work exactly as expected if there were multiple initial clones!
        :param t: time/sample index
        :param selection: 'all', 'ns', 's'. All/non-synonymous only/synonymous only.
        :param index_given: True if t is an index of the sample, False if t is a time.
        :param non_zero_only: Only return mutants with a positive cell count.
        :return: np.array of ints
        """
        if t is None:
            t = self.max_time
            index_given = False
        if not index_given:
            i = self.convert_time_to_index(t)
        else:
            i = t
        if self.mutant_clone_array is None:
            # If the mutant clone array has not been created yet, create it.
            self.create_mutant_clone_array()
        # We now find all rows in the mutant clone array that we want to keep
        if selection == 'all':
            muts = set(range(self.initial_clones, len(self.clones_array)))  # Get all rows except the initial clones
        elif selection == 'mutations':
            muts = self.ns_muts.union(self.s_muts)
        elif selection == 'ns':
            muts = self.ns_muts
        elif selection == 's':
            muts = self.s_muts

        muts = list(muts)

        mutant_clones = self.mutant_clone_array[muts][:, i].astype(int)

        if non_zero_only:
            return mutant_clones[mutant_clones > 0]
        else:
            return self.mutant_clone_array[muts][:, i].astype(int)

    def get_mutant_clone_size_distribution(self, t=None, selection='mutations', index_given=False):
        """
        Get the frequencies of mutant clone sizes. Not normalised.
        :param t: time/sample index
        :param selection: 'mutations', 'ns', 's'. All/non-synonymous only/synonymous only.
        :param index_given: True if t is an index of the sample, False if t is a time.
        :return: np.array of ints.
        """
        if t is None:
            t = self.max_time
            index_given = False
        if not index_given:
            i = self.convert_time_to_index(t)
        else:
            i = t
        if selection == 'mutations':
            if self.ns_muts and not self.s_muts:
                selection = 'ns'
            elif self.s_muts and not self.ns_muts:
                selection = 's'
            elif not self.s_muts and not self.ns_muts:
                print('No mutations at mutations')
                return None
        elif selection == 'ns' and not self.ns_muts:
            print('No non-synonymous mutations')
            return None
        elif selection == 's' and not self.s_muts:
            print('No synonymous mutations')
            return None

        clones = self.get_mutant_clone_sizes(i, selection=selection, index_given=True)

        counts = np.bincount(clones)
        counts[0] = 0   # Don't observe clones with zero cells.
        return counts

    def convert_time_to_index(self, t):
        """Find the index at or just before the time of interest"""
        i = bisect.bisect_right(self.times, t)
        if i:
            return i - 1
        raise ValueError

    def plot_incomplete_moment(self, t=None, selection='mutations', xlim=None, ylim=None, ax=None):
        """
        Plots the incomplete moment
        :param t: The time to plot the incomplete moment for. If None, will use the end of the simulation
        :param selection: 'mutations' for all mutations, 'ns' for non-synonymous only, or 's' for synonymous only
        :param xlim: Tuple/list for the x-limits of the plot
        :param ylim: Tuple/list for the y-limits of the plot
        :return:
        """
        if t is None:
            t = self.max_time
        clone_size_dist = self.get_mutant_clone_size_distribution(t, selection)

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if clone_size_dist is not None:
            incom = incomplete_moment(clone_size_dist)
            if incom is not None:
                x = np.arange(len(incom))

                ax.plot(x, incom)
                ax.set_yscale("log")
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)

    def get_dnds(self, t=None, min_size=0):
        """
        Returns the dN/dS at a particular time.
        :param t: Time. If None, will be the end of the simulation.
        :param min_size: Int. The minimum size of clones to include.
        :return:
        """
        if t is None:
            t = self.max_time
        ns_mut = self.get_mutant_clone_sizes(t, selection='ns')
        s_mut = self.get_mutant_clone_sizes(t, selection='s')
        ns_mut_measured = ns_mut[ns_mut > min_size]
        total_ns = len(ns_mut_measured)
        s_mut_measured = s_mut[s_mut > min_size]
        total_s = len(s_mut_measured)

        expected_ns = total_s * (1 / self.mutation_generator.synonymous_proportion - 1)
        try:
            dnds = total_ns / expected_ns
            return dnds
        except ZeroDivisionError as e:
            return np.nan

    def plot_dnds(self, plt_file=None, min_size=0, clear_previous=True, legend_label=None):
        if clear_previous:
            plt.close('all')
        dndss = [self.get_dnds(t, min_size) for t in self.times]
        plt.plot(self.times, dndss, label=legend_label)
        if plt_file is not None:
            plt.savefig('{0}'.format(plt_file))

    def plot_overall_population(self, label=None, legend_label=None):
        """
        With no label, plots for simulations without a fixed total population
        (will also run for the fixed population, but will not be interesting)

        With a label, will track the
        """
        pop = self.get_labeled_population(label=label)
        plt.plot(self.times, pop, label=legend_label)

    def get_labeled_population(self, label=None):
        """
        If label is None, will return the total population (not interesting for the fixed population models)
        :param label:
        :return: Array of population at all time points
        """
        if label is not None:
            clones_to_select = np.where(self.clones_array[:, self.label_idx] == label)
            pop = self.population_array[clones_to_select, :][0].astype(int)
        else:
            pop = self.population_array
        return pop.sum(axis=0)


def pickle_load(filename, unsparsify=True):
    """
    Load a simulation from a gzipped pickle
    :param filename:
    :return:
    """
    with gzip.open(filename, 'rb') as f:
        sim = pickle.load(f)

    if unsparsify:
        sim.unsparsify()

    return sim
