# Functions/classes to calculate the fitness of clones from the random mutation fitnesses
import math
import numpy as np


# Probability distributions for drawing the fitness of new mutations.
# Set up so can be called like functions without argument, but can print the attributes
class NormalDist(object):
    def __init__(self, std, mean=1):
        # 1 here means a neutral mutation.
        self.std = std
        self.mean = mean

    def __str__(self):
        return 'Normal distribution(mean {0}, std {1})'.format(self.mean, self.std)

    def __call__(self):
        return np.random.normal(self.mean, self.std)


class FixedValue(object):
    def __init__(self, value):
        self.mean = value

    def __str__(self):
        return 'Fixed value {0}'.format(self.mean)

    def __call__(self):
        return self.mean


class ExponentialDist(object):
    def __init__(self, mean, offset=1):
        self.mean = mean
        self.offset = offset  # Offset of 1 means the mutations will start from neutral.

    def __str__(self):
        return 'Exponential distribution(mean {0}, offset {1})'.format(self.mean, self.offset)

    def __call__(self):
        g = self.offset + np.random.exponential(self.mean)
        return g


class UniformDist(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __str__(self):
        return 'Uniform distribution(low {0}, high {1})'.format(self.low, self.high)

    def __call__(self):
        g = np.random.uniform(self.low, self.high)
        return g


class MixedDist(object):
    """A weighted mix of any of the other distributions"""

    def __init__(self, dists, proportions):
        self.dists = dists
        self.proportions = proportions

    def __str__(self):
        txt = 'Mixed distribution:\n'
        for d, p in zip(self.dists, self.proportions):
            txt += str(d) + '  prop:{}\n'.format(p)
        return txt

    def __call__(self):
        d = np.random.choice(self.dists, p=self.proportions)
        return d()


##################
# Classes for converting a combination of mutation fitnesses into a single fitness.
# Used for diminishing returns of fitness increases.

class BoundedLogisticFitness:
    """
    The effect of new (beneficial) mutations tails off as the clone gets stronger.
    There is a maximum fitness.
    """

    def __init__(self, a, b=math.exp(1)):
        """
        fitness = a/(1+c*b**(-x)) where x is the product of all mutation effects
        c is picked so that fitness(1) = 1
        a = max value.
        b determines the slope
        """
        if a <= 1:
            raise ValueError('a must be greater than 1')
        if b <= 1:
            raise ValueError('b must be greater than 1')
        self.a = a
        self.b = b
        self.c = (a - 1) * self.b

    def __str__(self):
        return 'Bounded Logistic: a {0}, b {1}, c {2}'.format(self.a, self.b, self.c)

    def fitness(self, x):
        return self.a / (1 + self.c * (self.b ** (-x)))

    def inverse(self, y):
        return math.log(self.c / (self.a / y - 1), self.b)


##################
# Class to put it all together

class MutationGenerator(object):
    """
    New mutations are drawn at random from the given fitness distribution
    """
    combine_options = ('multiply', 'add', 'replace', 'replace_lower')

    def __init__(self, combine_mutations='multiply', mutation_distribution=NormalDist(0.1),
                 synonymous_proportion=0.5, diminishing_returns=None):
        if combine_mutations not in self.combine_options:
            raise ValueError(
                "'{0}' is not a valid option for 'combine_mutaions'. Pick from {1}".format(combine_mutations,
                                                                                           self.combine_options))

        self.combine_mutations = combine_mutations
        self.mutation_distribution = mutation_distribution
        self.synonymous_proportion = synonymous_proportion

        self.diminishing_returns = diminishing_returns  # E.g. BoundedLogisticFitness above
        self.params = {
            'combine_mutations': combine_mutations,
            'mutation_distribution': self.mutation_distribution.__str__(),
            'diminishing_returns': diminishing_returns.__str__(),
            'synonymous_proportion': self.synonymous_proportion
        }

    def is_synonymous(self):
        return np.random.binomial(1, self.synonymous_proportion)

    def get_new_fitness(self, old_fitness):
        # np.random.random()
        syn = self.is_synonymous()
        if syn:
            new_fitness = old_fitness
        else:
            new_fitness = self.update_fitness(old_fitness)
        return new_fitness, syn

    def update_fitness(self, old_fitness):
        new_mutation_fitness = self.mutation_distribution()  # The fitness of the new mutation alone

        new_fitness = self.combine_fitness(old_fitness, new_mutation_fitness)

        return new_fitness

    def combine_fitness(self, old_fitness, new_mutation_fitness):
        if self.diminishing_returns is not None:
            # Convert to the "raw" fitness
            # Then move back to the diminished fitness after combining.
            old_fitness = self.diminishing_returns.inverse(old_fitness)

        if self.combine_mutations == 'multiply':
            combined_fitness = old_fitness * new_mutation_fitness
        elif self.combine_mutations == 'add':
            combined_fitness = max(old_fitness + new_mutation_fitness - 1, 0)
        elif self.combine_mutations == 'replace':
            combined_fitness = new_mutation_fitness
        elif self.combine_mutations == 'replace_lower':
            combined_fitness = max(new_mutation_fitness, old_fitness)
        else:
            raise NotImplementedError

        if combined_fitness < 0:
            combined_fitness = 0

        if self.diminishing_returns is not None:
            combined_fitness = self.diminishing_returns.fitness(combined_fitness)

        return combined_fitness


