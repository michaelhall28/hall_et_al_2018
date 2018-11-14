import numpy as np
from MoranTracker import MoranStyleSim


class Moran2D(MoranStyleSim):
    def __init__(self, parameters):

        super().__init__(parameters)

        self.grid = parameters.initial_grid.copy()  # The 2D grid for the simulation.
                                                    # Copy in case same grid used for other simulations.
        self.grid_shape = parameters.grid_shape

        self.grid_results = [self.grid.copy()]

    def sim_step(self, i, current_population, non_zero_clones):

        death_idx, coord = self.random_death()
        birth_idx = self.get_divider(coord)
        if self.mutations_to_add[i] > 0:  # If True, this division has been assigned at least one mutation
            new_muts = self.mutations_to_add[i]
            # Add the first mutation
            self.draw_mutation_and_add_to_array(birth_idx)
            for j in range(new_muts - 1):  # Add any subsequent mutations
                self.draw_mutation_and_add_to_array(self.next_mutation_index - 1)

            # Only add the last mutation to the population arrays
            new_cell = self.next_mutation_index - 1
        else:
            new_cell = birth_idx

        current_population[new_cell] += 1
        current_population[death_idx] -= 1
        self.grid[coord[0], coord[1]] = new_cell

        if i == self.sample_points[self.plot_idx] - 1:  # Must compare to -1 since increment is after this function
            self.grid_results.append(self.grid.copy())

        return current_population, non_zero_clones

    def random_death(self):
        coord = np.random.randint(0, self.grid_shape[0]), np.random.randint(0, self.grid_shape[1])
        cell = self.grid[coord[0], coord[1]]
        return cell, coord

    def get_neighbours(self, coord):
        grid = self.grid
        neighbours = [grid[(x, y)] for x, y in self.get_neighbour_coords(coord)]
        return neighbours

    def get_neighbour_coords(self, coord):

        x, y = coord
        grid_x, grid_y = self.grid_shape
        if y % 2 == 0:
            three_col = (x - 1) % grid_x
            one_col = (x + 1) % grid_x
        else:
            three_col = (x + 1) % grid_x
            one_col = (x - 1) % grid_x

        yield (x, (y - 1) % grid_y)
        yield (x, (y + 1) % grid_y)

        for i in range(y - 1, y + 2):
            yield (three_col, i % grid_y)

        yield (one_col % grid_x, y)

    def get_divider(self, coord):
        """Pick a cell from the surrounding cells that will replace the dead cell.
        The cell is selected at random in proportion to its relative fitness"""
        neighbours = self.get_neighbours(coord)
        weights = self.clones_array[neighbours, self.fitness_idx]
        relative_weights = weights / weights.sum()
        return neighbours[np.random.multinomial(1, relative_weights).argmax()]  # faster than np.random.choice

