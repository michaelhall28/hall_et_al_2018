from FullTracker import *
from useful_functions import find_ge


class MoranStyleSim(GeneralSimClass):
    """Runs a simulation of the clonal growth, mutation and competition"""

    def __init__(self, parameters):

        super().__init__(parameters)

    def precalculate_mutations(self):
        total_divisions = self.sample_points[-1]  # Length of the simulation
        self.mutations_to_add = np.random.poisson(self.mutation_rate, total_divisions)
        self.new_mutation_count = self.mutations_to_add.sum()

    def sim_step(self, i, current_population, non_zero_clones):
        """One cell is selected to die at random. Another cell is selected to replicate and replace the dead cell
        with its offspring. The replicating cell is selected in proportion with its relative fitness"""

        ### Select population to replicate cell ###
        # Select random number to select which population
        birth_selector = np.random.random()
        # make cumulative list of the fitnesses
        fitness_cumsum = np.cumsum(current_population * self.clones_array[non_zero_clones, self.fitness_idx], axis=0)
        # Pick out the selected population
        # birth_idx is the index for the current population. The clone number is non_zero_clones[birth_idx]
        birth_idx = find_ge(fitness_cumsum, birth_selector * fitness_cumsum[-1])

        ### Select replaced population ###
        # death_idx is the index for the current population. The clone number is non_zero_clones[death_idx]
        death_selector = np.random.random()
        cumsum = np.cumsum(current_population, axis=0)
        death_idx = find_ge(cumsum, death_selector * cumsum[-1])

        if self.mutations_to_add[i] > 0:  # If True, this division has been assigned a mutation
            new_muts = self.mutations_to_add[i]
            # New mutation means extending the current_population.
            # Only have to add one clone to the current population. The rest with not be non-zero clones.
            current_population = np.concatenate([current_population, [1]])

            # Add the first mutation
            self.draw_mutation_and_add_to_array(non_zero_clones[birth_idx])
            for j in range(new_muts - 1):  # Add any subsequent mutations
                self.draw_mutation_and_add_to_array(self.next_mutation_index - 1)

            # Only add the last mutation
            non_zero_clones = np.concatenate([non_zero_clones, [self.next_mutation_index - 1]])
        else:
            current_population[birth_idx] += 1
        current_population[death_idx] -= 1
        if current_population[death_idx] == 0:  # A clone has gone extinct. Remove from the current arrays
            current_population = np.concatenate([current_population[:death_idx], current_population[death_idx + 1:]])
            non_zero_clones = np.concatenate([non_zero_clones[:death_idx], non_zero_clones[death_idx + 1:]])

        return current_population, non_zero_clones
