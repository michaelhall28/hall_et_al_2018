from parameters import Parameters
from FitnessClasses import *
from simulation_scraping_disjoint import get_rsquared_all_sims
from plot_functions import plot_incomplete_moment_with_random_selection
import os
import glob

import numpy as np
import matplotlib.pyplot as plt


def neutral_example_2D():
    np.random.seed(0)
    mutation_generator = MutationGenerator(mutation_distribution=FixedValue(1), synonymous_proportion=0.5)
    p = Parameters(algorithm='Moran2D', max_time=100, grid_shape=(100, 100), mutation_rate=0.1,
                   mutation_generator=mutation_generator, division_rate=0.5, progress=50000)
    sim = p.get_simulator()
    sim.run_sim()
    sim.unsparsify()


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    mutant_clone_size_distribution = sim.get_mutant_clone_size_distribution()
    ax1.bar(range(1, len(mutant_clone_size_distribution)), mutant_clone_size_distribution[1:])
    ax1.set_title('Clone size distribution')
    ax1.set_xlabel('Clone size')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(left=0)


    dnds = sim.get_dnds()
    print('dN/dS', dnds)

    sim.plot_incomplete_moment(ax=ax2)
    ax2.set_title('First incomplete moment')
    ax2.set_xlim(left=0)
    plt.show()


def non_neutral_example_2D():
    np.random.seed(0)
    mutation_generator = MutationGenerator(mutation_distribution=NormalDist(mean=1.1, std=0.1),
                                           synonymous_proportion=0.99)
    p = Parameters(algorithm='Moran2D', max_time=100, grid_shape=(100, 100), mutation_rate=0.1,
                   mutation_generator=mutation_generator, division_rate=0.5, progress=50000)
    sim = p.get_simulator()
    sim.run_sim()
    sim.unsparsify()


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    mutant_clone_size_distribution = sim.get_mutant_clone_size_distribution()
    ax1.bar(range(1, len(mutant_clone_size_distribution)), mutant_clone_size_distribution[1:])
    ax1.set_title('Clone size distribution')
    ax1.set_xlabel('Clone size')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(left=0)


    dnds = sim.get_dnds()
    print('dN/dS', dnds)

    sim.plot_incomplete_moment(ax=ax2)
    ax2.set_title('First incomplete moment')
    ax2.set_xlim(left=0)
    plt.show()


def multiplot_example():
    """
    This will take a longer amount of time ~10 minutes.

    To recreate the figures:
     - the grid size should be increased to 500
     - the max time set at 3000
     - num_simulations = 1000
     - Use the longer list of biopsy locations (locs = [0, 100, 200, 300, 400])
    Smaller simulations used here so that the example can run in a reasonable amount of time.

    An example is given for the 1% non-neutral simulations.
    The neutral and 25% non-neutral simulations can be run using settings in the commented-out code

    """
    output_dir = 'non_neutral_1perc_simulations'
    try:
        os.mkdir(output_dir)
    except FileExistsError as e:
        for f in glob.glob("{}/*.pickle".format(output_dir)):
            print('.pickle files already exist in the non_neutral_1perc_simulations directory. Remove before running.')
            exit(1)

    grid_size = 200  # Increase to 500 to recreate the figures.
    num_simulations = 5  # Increase to 1000 to recreate the figures.

    non_neutral_1perc_mutation_generator = MutationGenerator(combine_mutations='add',
                                                             mutation_distribution=NormalDist(std=0.1, mean=1.1),
                                                             synonymous_proportion=0.99)
    non_neutral_division_rate = 0.033
    non_neutral_mutation_rate = 0.015

    max_time = 3000
    num_cells = grid_size ** 2

    p = Parameters(algorithm='Moran2D', mutation_generator=non_neutral_1perc_mutation_generator,
                      initial_cells=num_cells,
                      division_rate=non_neutral_division_rate, max_time=max_time,
                      mutation_rate=non_neutral_mutation_rate, samples=10)


    # To run the 25% non-neutral simulations use:
    # non_neutral_25perc_mutation_generator = MutationGenerator(combine_mutations='add',
    #                                                           mutation_distribution=NormalDist(std=0.1, mean=1.1),
    #                                                           synonymous_proportion=0.75)
    # p = Parameters(algorithm='Moran2D', mutation_generator=non_neutral_25perc_mutation_generator,
    #                          initial_cells=num_cells,
    #                          division_rate=non_neutral_division_rate, max_time=max_time,
    #                          mutation_rate=non_neutral_mutation_rate, samples=10)

    ## To run the neutral simulations use
    # neutral_mutation_generator = MutationGenerator(combine_mutations='add',
    #                                                mutation_distribution=FixedValue(1),
    #                                                synonymous_proportion=0.5)
    # neutral_mutation_rate = 0.001
    # neutral_division_rate = 0.5
    # p = Parameters(algorithm='Moran2D', mutation_generator=neutral_mutation_generator,
    #                        initial_cells=num_cells,
    #                        division_rate=neutral_division_rate, max_time=max_time,
    #                        mutation_rate=neutral_mutation_rate, samples=10)

    for i in range(1, num_simulations+1):
        # Run a simulation
        np.random.seed(i)
        sim_2D = p.get_simulator()
        sim_2D.run_sim()
        output_file = '{}/Moran2D_{}-{}-{}-{}.pickle'.format(output_dir, '1perc', max_time, num_cells, i)
        sim_2D.pickle_dump(output_file)
        print('Completed {} of {} simulations'.format(i, num_simulations))

    # Define some biopsies
    biopsy_edge = 70
    locs = [0, 100]  # Use [0, 100, 200, 300, 400] to recreate the figures.

    biopsies = []
    for i in locs:
        for j in locs:
            biopsies.append({'biopsy_origin': (i, j), 'biopsy_edge': biopsy_edge}, )

    coverage = 1000
    detection_limit = 10
    fixed_interval_clone_size = 25 # Defining the size of the bins to group the clones into.

    print('Processing simulation results')
    # This will include all simulation results in the directory.
    res = get_rsquared_all_sims(output_dir, biopsies=biopsies, coverage=coverage, detection_limit=detection_limit,
                                fixed_interval=fixed_interval_clone_size)


    plot_incomplete_moment_with_random_selection(res,
                                                 x_vals=np.arange(0, 3000, fixed_interval_clone_size),
                                                 with_biopsy=False,
                                                 convert_to_clone_size=True,
                                                 biopsy_size=grid_size**2,
                                                 linecolour='k',
                                                 rangecolour='b',
                                                 num_shown=min(20, num_simulations),
                                                 output_file='{}/Full_data.pdf'.format(output_dir))
    plt.close()

    plot_incomplete_moment_with_random_selection(res,
                                                 x_vals=np.arange(0, 3000, fixed_interval_clone_size),
                                                 with_biopsy=True,
                                                 convert_to_clone_size=True,
                                                 biopsy_size=biopsy_edge**2,
                                                 linecolour='k',
                                                 rangecolour='b',
                                                 num_shown=min(20, num_simulations),
                                                 output_file='{}/Biopsy_sampling_data.pdf'.format(output_dir))
    plt.close()

if __name__ == "__main__":
    # neutral_example_2D()
    # non_neutral_example_2D()
    multiplot_example()