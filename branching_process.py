from FullTracker import GeneralSimClass
import numpy as np
from useful_functions import find_ge
from scipy.sparse import lil_matrix


class SimpleBranchingProcess(GeneralSimClass):
    """
    A simplified version of the single progenitor model
    Progenitor cells either divide to form two new progenitor cells or they die
    The chance of division/death is determined by the fitness of the clone
    Unlike the Wright-Fisher or Moran models, the total population size is not fixed.
    Each cell/clone acts independently of all others.
    """
    def __init__(self, parameters):

        self.time = 0
        super().__init__(parameters)

    def take_sample(self, i, current_population, non_zero_clones):
        """
        Record the results at the point the simulation is up to.
        Report progress if required
        :param i:
        :param current_population:
        :return:
        """
        while self.plot_idx < len(self.times) and self.time >= self.times[self.plot_idx]:
            # If moved on more than one time point, fill up to that point
            self.population_array[non_zero_clones, self.plot_idx] = current_population.reshape(
                len(current_population),
                1)
            self.plot_idx += 1
            if self.tmp_store is not None:  # Store current state of the simulation.
                if self.store_rotation == 0:
                    self.pickle_dump(self.tmp_store)
                    self.store_rotation = 1
                else:
                    self.pickle_dump(self.tmp_store + '1')
                    self.store_rotation = 0

            if self.progress:
                print(self.times[self.plot_idx - 1], end=', ', flush=True)

    def sim_step(self, i, current_population, non_zero_clones):
        """
        One of the current living cells is selected at random.
        A random number determines if it will divide or die.
        Another random draw determines if there will be a mutation in one of the new cells.
        """

        # Select random number to select which population will divide or die
        population_selector = np.random.random()
        cumsum = np.cumsum(current_population, axis=0)
        selected_idx = find_ge(cumsum, population_selector * cumsum[-1])  # This relates to the non_zero_clones

        # Division rate is taken as r*lambda.
        # The rate of either a symmetric AA or BB division is then 2*r*lambda = 2*division_rate
        # This then matches with the Moran model.
        # This branching model requires twice as many simulations steps as the Moran as the divisions and deaths
        # happen in different steps
        self.time += np.random.exponential(1 / (cumsum[-1] * 2 * self.division_rate))

        # Fitness=1 is balanced.
        # A random draw from [0,2) is taken.
        # If the draw is above the fitness, the cell will die.
        # If the draw is below the fitness, the cell will divide (and possibly mutate).
        # This means the higher the fitness, the more the clone will proliferate.
        # Fitnesses above 2 are essentially infinite, the clone will not die.
        if np.random.uniform(0, 2) <= self.clones_array[non_zero_clones[selected_idx], self.fitness_idx]:  # Division
            new_muts = np.random.poisson(self.mutation_rate)
            if new_muts > 0:
                if self.next_mutation_index + new_muts >= self.population_array.shape[0]:  # Run out of room. Must extend the arrays.
                    self.extend_arrays(current_population, min_extension=self.population_array.shape[
                                                                             0] - self.next_mutation_index + new_muts)

                current_population = np.concatenate([current_population, [1]])  # Add the new mutant population

                # Add the first mutation
                self.draw_mutation_and_add_to_array(non_zero_clones[selected_idx])
                for j in range(new_muts - 1):  # Add any subsequent mutations
                    self.draw_mutation_and_add_to_array(self.next_mutation_index - 1)

                # Only add the last mutation
                non_zero_clones = np.concatenate([non_zero_clones, [self.next_mutation_index - 1]])
            else:
                current_population[selected_idx] += 1
        else:  # Death
            current_population[selected_idx] -= 1
            if current_population[selected_idx] == 0:  # A clone has gone extinct. Remove from the current arrays
                current_population = np.concatenate([current_population[:selected_idx],
                                                     current_population[selected_idx + 1:]])
                non_zero_clones = np.concatenate([non_zero_clones[:selected_idx],
                                                  non_zero_clones[selected_idx + 1:]])
                if current_population.sum() == 0:
                    # The population can go extinct in this simulation. Must then stop the sim.
                    self.time = self.max_time

        return current_population, non_zero_clones

    def extend_arrays(self, current_population, min_extension=1):
        """
        We cannot pre-calculate the number of mutations (and therefore clones) as the population is not fixed
        so we must extend the arrays once they get full
        """
        # Take a rough guess at the number of mutations remaining. Can add more or remove rows from the arrays later.
        chunk_increase = max(int(self.mutation_rate * self.division_rate *
                                 current_population.sum() * (self.max_time - self.time)) + 1, min_extension)

        s = self.population_array.shape[0]
        new_pop_array = lil_matrix((s + chunk_increase, self.sim_length))
        new_pop_array[:s] = self.population_array
        self.population_array = new_pop_array

        self.clones_array = np.concatenate([self.clones_array, np.zeros((chunk_increase, 5))], axis=0)

        for i in range(self.next_mutation_index, self.next_mutation_index + chunk_increase):
            self.clone_descendants[i].add(i)
            self.clone_ancestors[i].add(i)

    def finish_up(self):
        """
        Some of the plotting/post processing steps assume that all rows in the arrays are used in the simulation
        Remove rows that have not been used
        """
        self.clones_array = self.clones_array[:self.next_mutation_index]
        self.population_array = self.population_array[:self.next_mutation_index]

        for i in range(self.next_mutation_index, max(self.clone_ancestors.keys()) + 1):
            del self.clone_ancestors[i]
            del self.clone_descendants[i]
