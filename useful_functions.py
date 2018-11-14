import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import bisect



def mean_clone_size(clone_size_dist):
    # Mean of surviving clones from a clone size frequency array
    """Gets the mean of clones > 1 cell. For dists that start at 0 cell clones"""
    return sum([(i) * clone_size_dist[i] for i in range(1, len(clone_size_dist))]) / clone_size_dist[1:].sum()


# Incomplete moment functions
def incomplete_moment(clone_size_dist):
    # Assuming clone_size_dist starts from zero
    if clone_size_dist[1:].sum() == 0:
        return None
    mcs = mean_clone_size(clone_size_dist)
    total_living_clones = clone_size_dist[1:].sum()
    proportions = clone_size_dist / total_living_clones
    sum_terms = proportions * np.arange(len(proportions))
    moments = np.cumsum(sum_terms[::-1])[::-1]
    return moments / mcs


def add_incom_to_plot(incom, clone_size_dist):
    """Add an incomplete moment trace to a plot. Can also add a fit to the trace."""
    min_size = np.nonzero(clone_size_dist)[0][0]
    plot_incom = incom[min_size:]  # We don't plot for clone size 0
    plot_x = np.arange(min_size, len(incom))
    plt.plot(plot_x, plot_incom, label=label, color=colour)


def find_ge(a, x):
    """Find leftmost item greater than or equal to x"""
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i
    raise ValueError

