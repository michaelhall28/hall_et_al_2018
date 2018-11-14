from FitnessClasses import MutationGenerator, NormalDist
from MoranTracker import MoranStyleSim
from Moran2D import Moran2D
from branching_process import SimpleBranchingProcess
import sys
import numpy as np


class ParameterException(Exception):
    pass


class Parameters(object):
    """
    Defines the parameters for a simulation
    Performs some checks on the parameters before running.
    """
    def __init__(self,
                 algorithm=None,
                 max_time=None,
                 division_rate=None,
                 mutation_generator=None,
                 initial_cells=None,
                 initial_size_array=None,
                 fitness_array=None,
                 label_array=None,
                 mutation_array=None,
                 mutation_rate=None,
                 samples=None,
                 figsize=None,
                 progress=False,
                 grid_shape=None,  # 2D simulations only
                 initial_grid=None,  # 2D simulations only
                 simulation_steps=None,  # Alternative to supplying max_time and division_rate
                 print_warnings=True,  # Will warn when using a default parameter or calculating one
                 tmp_store=None,  # File to store the partial results of simulation.
                 ):
        self.algorithm = algorithm
        self.max_time = max_time
        self.division_rate = division_rate
        self.mutation_generator = mutation_generator
        self.initial_cells = initial_cells
        self.initial_size_array = initial_size_array
        self.fitness_array = fitness_array
        self.label_array = label_array
        self.mutation_array = mutation_array
        self.mutation_rate = mutation_rate
        self.samples = samples
        self.figsize = figsize
        self.progress = progress
        self.grid_shape = grid_shape # 2D simulations only
        self.initial_grid = initial_grid  # 2D simulations only
        self.simulation_steps = simulation_steps  # Alternative to supplying max_time and division_rate (for some algorithms)
        self.tmp_store = tmp_store

        self.non_zero_calc = True  # Needs to be true for some algorithms, false for others. Will be set automatically
        # Essentially, if the algorithm uses the entire current population to calculate the next step,
        # this should speed it up.

        # Values to calculate from those provided
        self.times = None  # Numpy array of times at which the samples are taken
        self.sample_points = None  # Array of simulation_steps at which the samples are taken

        # DEFAULT VALUES. If any of the required parameters are not provided, the defaults will be used
        self.default_division_rate = 1
        self.default_max_time = 10
        self.default_mutation_generator = MutationGenerator(combine_mutations='multiply',
                                                            mutation_distribution=NormalDist(0.1),
                                                            synonymous_proportion=0.5,
                                                            diminishing_returns=None)
        self.default_mutation_rate = 0.01
        self.default_samples = 100
        self.default_figsize = (10, 10)
        self.default_label = 0
        self.default_fitness = 1

        # OPTIONS. Where one of a few limited options are required, put here
        self.algorithm_options = {'Moran', 'Moran2D', 'Branching'}

        # Other. Attributes used internally to help printing etc.
        self.initial_grid_provided = False
        self.print_warnings = print_warnings
        self.warnings = []

        if not self.check_parameters():
            sys.exit(1)
        self.sim_class = None
        self.select_simulator_class()

    def __str__(self):
        s = "Parameters:"
        s += "\n\tAlgorithm: {0}".format(self.algorithm)
        s += "\n\tMax time: {0}".format(self.max_time)
        s += "\n\tDivision rate: {0}".format(self.division_rate)
        s += "\n\tMutation generator: {0}".format(self.mutation_generator)
        s += "\n\tInitial cells: {0}".format(self.initial_cells)
        s += "\n\tInitial size array: {0}".format(self.initial_size_array)
        s += "\n\tInitial fitness array: {0}".format(self.fitness_array)
        s += "\n\tInitial label array: {0}".format(self.label_array)
        s += "\n\tMutation rate: {0}".format(self.mutation_rate)
        s += "\n\tNumber of samples: {0}".format(self.samples)
        s += "\n\tFigsize: {0}".format(self.figsize)
        s += "\n\tProgress reporting: {0}".format(self.progress)
        s += "\n\tSimulation steps: {0}".format(self.simulation_steps)
        if self.algorithm == 'Moran2D':
            s += "\n\tGrid shape: {0}".format(self.grid_shape)
            s += "\n\tInitial grid provided: {0}".format(self.initial_grid_provided)

        return s

    def check_parameters(self):
        """Check parameters for consistency. Defines any missing parameters which can be calculated from others."""
        try:
            if self.mutation_generator is None:
                self.warnings.append('Using the default mutation generator: {0}'.format(self.default_mutation_generator.__str__()))
                self.mutation_generator = self.default_mutation_generator
            self.check_algorithm()
            self.check_populations()
            self.check_timing()
            self.check_samples()
            self.get_sample_times()  # Must be called after the timing and samples have been checked

            if self.mutation_rate is None:
                self.warnings.append('Using the default mutation rate: {0}'.format(self.default_mutation_rate))
                self.mutation_rate = self.default_mutation_rate

            if self.figsize is None:
                self.figsize = self.default_figsize

            if self.progress is None:
                self.progress = False

        except ParameterException as e:
            print('Error with parameters, unable to run simulation.\n')
            print(e)
            return False

        if self.print_warnings:
            print('============== Setting up ==============')
            for w in self.warnings:
                print(w)
            print('========================================')
        return True

    def check_algorithm(self):
        """Checks that the algorithm asked for is one of the options"""
        if self.algorithm not in self.algorithm_options:
            raise ParameterException('Algorithm {0} is not valid. Pick from {1}'.format(self.algorithm, self.algorithm_options))
        if self.algorithm in ['Branching', 'Moran']:
            self.non_zero_calc = True
        elif self.algorithm in ['Moran2D']:
            self.non_zero_calc = False

    def check_populations(self):
        """Checks that only one population parameter has been given"""
        num_defined = sum([self.initial_cells is not None, self.initial_size_array is not None,
                           self.grid_shape is not None, self.initial_grid is not None])
        if num_defined != 1:
            raise ParameterException('Must provide exactly one of:\n\tinitial_cells\n\tinitial_size_array\n\t'
                                     'grid_shape (Moran2D only)\n\tinitial_grid (Moran2D only)')
        if self.algorithm != 'Moran2D':
            if self.initial_cells is None and self.initial_size_array is None:
                raise ParameterException('Must provide initial_cells or initial_size_array')
            self.setup_initial_population_non_spatial()
        else:
            self.setup_2D_initial_population()

        self.define_remaining_initial_arrays()
        self.convert_lists_to_arrays()

    def convert_lists_to_arrays(self):
        # Some cases need to be numpy arrays not lists
        self.initial_size_array = np.array(self.initial_size_array)

    def setup_initial_population_non_spatial(self):
        if self.initial_cells is not None:
            self.initial_size_array = [self.initial_cells]
        elif self.initial_size_array is not None:
            self.initial_cells = sum(self.initial_size_array)

    def setup_2D_initial_population(self):
        if self.initial_cells is not None:
            self._try_making_square_grid()
            self.initial_size_array = [self.initial_cells]
            self.initial_grid = np.zeros(self.grid_shape, dtype=int)
        elif self.initial_size_array is not None:
            if len(self.initial_size_array) == 1:
                self.initial_cells = sum(self.initial_size_array)
                self._try_making_square_grid()
                self.initial_grid = np.zeros(self.grid_shape, dtype=int)
            else:
                raise ParameterException('Cannot use initial_size_array with 2D simulation. Provide initial_grid instead.')
        elif self.grid_shape is not None:
            self.initial_cells = self.grid_shape[0] * self.grid_shape[1]
            self.initial_size_array = [self.initial_cells]
            self.initial_grid = np.zeros(self.grid_shape, dtype=int)
        elif self.initial_grid is not None:
            self.grid_shape = self.initial_grid.shape
            self.initial_cells = self.grid_shape[0] * self.grid_shape[1]
            self.create_initial_size_array_from_grid()
            self.initial_grid_provided = True
        else:
            raise ParameterException('Please provide one of the population size inputs')

        self.check_other_2D_parameters()

    def _try_making_square_grid(self):
        poss_grid_size = int(np.sqrt(self.initial_cells))
        if poss_grid_size ** 2 == self.initial_cells:
            self.grid_shape = (poss_grid_size, poss_grid_size)
            self.warnings.append('Using a grid of {0}x{0}'.format(self.grid_shape[0], self.grid_shape[1]))
        else:
            raise ParameterException('Square grid not compatible with {0} cells. To run a rectangular grid provide a grid shape'.format(
                self.initial_cells))

    def define_remaining_initial_arrays(self):
        # Define the initial arrays if they have not been defined yet.
        if self.label_array is None:
            self.label_array = [self.default_label for i in range(len(self.initial_size_array))]
        elif len(self.label_array) != len(self.initial_size_array):
            raise ParameterException('Inconsistent initial_size_array and label_array. Ensure same length.')
        if self.fitness_array is None:
            self.fitness_array = [self.default_fitness for i in range(len(self.initial_size_array))]
        elif len(self.fitness_array) != len(self.initial_size_array):
            raise ParameterException('Inconsistent initial_size_array and fitness_array. Ensure same length.')

    def check_other_2D_parameters(self):
        # Check that the hexagonal grid has only even dimensions.
        if self.grid_shape[0] % 2 != 0 or self.grid_shape[1] % 2 != 0:
            raise ParameterException('Must have even number of rows/columns in the hexagonal grid.')

    def check_timing(self):
        """Checks that max_time, division_rate and simulation_steps are consistent and defines any missing values"""
        if self.simulation_steps is not None:
            if self.algorithm in ['Branching']:
                raise ParameterException('Cannot specify number of simulations steps for the branching process algorithm.\n' 
                                         'Please provide a max_time and division_rate instead')
            if self.max_time is not None:
                if self.division_rate is not None:
                    # All defined, check they are consistent
                    sim_steps = self.get_simulation_steps()
                    if sim_steps != self.simulation_steps:   # Raise error if not consistent
                        st = 'Simulation_steps does not match max_time and division_rate.\n' \
                             'Provide only 2 of the three or ensure all are consistent.\n' \
                             'simulation_steps={0}, steps calculated from time and division rate={1}'.format(
                            self.simulation_steps, sim_steps
                        )
                        raise ParameterException(st)

                else:
                    # simulation_steps and max_time given. Calculate division rate
                    self.division_rate = self.get_division_rate()
                    self.warnings.append('Division rate for the simulation is {0}'.format(self.division_rate))
            else:
                if self.division_rate is None:
                    # Simulation steps defined but not max_time and division_rate
                    # Use the default division rate
                    self.use_default_division_rate()
                # simulation_steps and division_rate given, calculate max_time
                self.max_time = self.get_max_time()
                self.warnings.append('Max time for the simulation is {0}'.format(self.max_time))
        else:  # No simulation steps defined. Calculate from max_time and division rate if given
            if self.division_rate is None:
                self.use_default_division_rate()
            if self.max_time is None:
                self.use_default_max_time()
            self.simulation_steps = self.get_simulation_steps()
            self.warnings.append('{0} simulation_steps'.format(self.simulation_steps))

    def check_samples(self):
        if self.samples is None:
            self.samples = self.default_samples
        if self.simulation_steps is not None:
            if self.samples > self.simulation_steps:
                self.samples = self.simulation_steps

    def get_sample_times(self):
        self.times = np.linspace(0, self.max_time, self.samples+1)  # The time points at each sample. Used for plotting.
        if self.algorithm not in ['Branching']:
            steps_between_samples = self.simulation_steps / self.samples
            self.sample_points = (np.arange(len(self.times)) * steps_between_samples).astype(int)  # Which points to take a sample
        else:
            self.sample_points = None

    def create_initial_size_array_from_grid(self):
        """For the 2D simulations, if an initial grid of clone positions is provided, fill in the initial_size_array"""

        # If the initial size array is not given, define it here.
        idx_counts = {k:v for k,v in zip(*np.unique(self.initial_grid, return_counts=True))}
        self.initial_size_array = []
        for i in range(max(idx_counts)+1):
            if i in idx_counts:
                self.initial_size_array.append(idx_counts[i])
            else:
                self.initial_size_array.append(0)

    def get_simulation_steps(self):
        if self.algorithm in ['Moran', 'Moran2D']:
            sim_steps = int(self.max_time * self.division_rate * self.initial_cells)
        elif self.algorithm in ['Branching']:
            sim_steps = None
        else:
            raise ParameterException('Calculation of sim_steps for {0} not implemented.'.format(self.algorithm))
        return sim_steps

    def get_division_rate(self):
        if self.algorithm in ['Moran', 'Moran2D']:
            div_rate = self.simulation_steps/self.max_time/self.initial_cells
        else:
            raise ParameterException('Calculation of division rate for {0} not implemented. Please provide.'.format(self.algorithm))
        return div_rate

    def get_max_time(self):
        if self.algorithm in ['Moran', 'Moran2D']:
            max_time = self.simulation_steps/self.division_rate/self.initial_cells
        else:
            raise ParameterException('Calculation of max_time for {0} not implemented. Please provide.'.format(self.algorithm))
        return max_time

    def use_default_division_rate(self):
        self.warnings.append('Using the default division rate: {0}'.format(self.default_division_rate))
        self.division_rate = self.default_division_rate

    def use_default_max_time(self):
        self.warnings.append('Using the default max time: {0}'.format(self.default_max_time))
        self.max_time = self.default_max_time

    def select_simulator_class(self):
        if self.algorithm == 'Moran':
            self.sim_class = MoranStyleSim
        elif self.algorithm == 'Moran2D':
            self.sim_class = Moran2D
        elif self.algorithm == 'Branching':
            self.sim_class = SimpleBranchingProcess

    def get_simulator(self):
        return self.sim_class(self)