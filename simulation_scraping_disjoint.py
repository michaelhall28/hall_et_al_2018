import numpy as np
import pickle
import glob
import os
from FullTracker import pickle_load
from collections import defaultdict


def get_rsquared_all_sims(sim_dir, biopsies, coverage, detection_limit, fixed_interval):
    """

    :param sim_dir: Directory containing the pickle file outputs of simulations. Must be named ending ".pickle"
    Must not contain other files ending ".pickle"
    :param biopsies:
    :param coverage:
    :param detection_limit:
    :param fixed_num:
    :param fixed_interval:
    :return:
    """
    all_results = {}
    count = 1
    for sim_file in glob.glob(os.path.join(sim_dir, '*.pickle')):
        print('sim', count, os.path.basename(sim_file), flush=True)
        sim_results = get_results_for_sim(sim_file, biopsies, coverage, detection_limit, fixed_interval)

        all_results[os.path.basename(sim_file)] = sim_results

        count += 1

    return all_results


def rsq_calc(x, y):
    if len(y) > 1:
        x_min = x.min()
        log_incom = np.log(y)
        B = np.vstack([x]).T - x_min
        slope, resids = np.linalg.lstsq(B, log_incom)[0:2]
        rsq = 1 - resids[0] / (len(log_incom) * log_incom.var())
    else:
        print('Only 1 clone size')
        rsq = np.nan

    return rsq


def incomplete_moment_vaf_fixed_intervals(vafs, interval):
    vafs = np.flip(np.sort(vafs), axis=0)
    mean_clone_size = vafs.mean()

    x = np.arange(int(vafs.min() / interval) * interval, round(vafs.max() / interval) * interval,
                  interval)

    if len(x) == 0:  # No variation in clone sizes.
        return np.array([]), np.array([])

    x_idx = -1
    last_x = x[x_idx]
    y = []
    incom = 0
    for v in vafs:
        while v < last_x:
            y.append(incom)
            x_idx -= 1
            last_x = x[x_idx]
        incom += v
    y.append(incom)

    return x, np.flip(np.array(y), axis=0) / mean_clone_size / len(vafs)


def get_results_for_sim(sim_file, biopsies, coverage, detection_limit, fixed_interval):
    """
    Get the vafs for full data, biopsies and matched sample size full data.
    Return the sample size and
    for each of (full data points, fixed num points, fixed interval points):
        Full x and y for incomplete moment
        X,y and r2 for incomplete moment with biopsies and coverage
        X,y and r2 with matched sample size (no biopsies)

    also return the raw vafs for re-calculation of rsquares

    The fixed interval has to be adjusted for the biopsies and not, since the calculations are done of VAF,
    but plotted in clone cell count.

    :param sim_file:
    :param biopsies:
    :param coverage:
    :param detection_limit:
    :param sample_size:
    :return:
    """
    sim = pickle_load(sim_file)
    grid = sim.grid_results[-1]

    # Convert the fixed interval to VAF
    # Assumes all biopsies are equal size and square
    fixed_interval_full = fixed_interval / (grid.shape[0] * grid.shape[1]) / 2
    fixed_interval_biop = fixed_interval / biopsies[0]['biopsy_edge']**2 / 2


    vafs_full = get_vafs(grid, sim.clone_ancestors, biopsies=None, coverage=None, detection_limit=None, subsample=None)
    vafs_biopsy = get_vafs(grid, sim.clone_ancestors, biopsies, coverage, detection_limit, subsample=None)
    vafs_match = get_vafs(grid, sim.clone_ancestors, biopsies=None, coverage=None, detection_limit=None,
                          subsample=len(vafs_biopsy))

    dnds_full = get_dnds(vafs_full, sim)
    dnds_biopsy = get_dnds(vafs_biopsy, sim)
    dnds_match = get_dnds(vafs_match, sim)

    # Now only need the vafs themselves, not the clone numbers.
    vafs_full = vafs_full[:, 1]
    vafs_biopsy = vafs_biopsy[:, 1]
    vafs_match = vafs_match[:, 1]

    res_full = get_r2_results(vafs_full, fixed_interval_full, 'full')
    res_biop = get_r2_results(vafs_biopsy, fixed_interval_biop, 'biop')
    res_matched = get_r2_results(vafs_match, fixed_interval_full, 'matched')

    results = {
        **res_full,
        **res_biop,
        **res_matched,
        'sample_size': len(vafs_biopsy),
        'vafs_full': vafs_full,
        'vafs_biopsy': vafs_biopsy,
        'vafs_match': vafs_match,
        'dnds_full': dnds_full,
        'dnds_biopsy': dnds_biopsy,
        'dnds_match': dnds_match
    }
    return results


def get_r2_results(vafs, fixed_interval, prefix):
    if len(vafs) > 1:
        x_interval, y_interval = incomplete_moment_vaf_fixed_intervals(vafs, fixed_interval)
        rsq_interval = rsq_calc(x_interval, y_interval)

    else:
        rsq_interval = np.nan
        x_interval, y_interval = vafs, [1]

    results = {
        '{0}_r2_interval'.format(prefix): rsq_interval,
        '{0}_x_interval'.format(prefix): x_interval,
        '{0}_y_interval'.format(prefix): y_interval
    }
    return results


def get_vafs(grid, ancestors, biopsies, coverage, detection_limit, subsample=None):
    vaf_arr = biopsies_samples(grid, ancestors, biopsies)

    if coverage is not None:
        vaf_arr = small_detection_limit(vaf_arr, coverage, detection_limit)

    if subsample is not None:
        vaf_arr = random_sample_vafs(vaf_arr, subsample)

    return vaf_arr


def random_sample_vafs(vaf_arr, num):
    # Take random subsample without replacement if enough clones.
    # Otherwise take with replacement (not required for paper figures)
    if len(vaf_arr) >= num:
        return vaf_arr[np.random.choice(vaf_arr.shape[0], num, replace=False), :]
    else:
        return vaf_arr[np.random.choice(vaf_arr.shape[0], num, replace=True), :]


def biopsies_samples(grid, ancestors, biopsies):
    if biopsies is None or type(biopsies) == dict:
        vaf_arr = biopsy_sample(grid, ancestors, biopsies)
    else:
        all_biopsy_samples = [biopsy_sample(grid, ancestors, biopsy) for biopsy in biopsies]
        try:
            vaf_arr = np.concatenate([s for s in all_biopsy_samples if len(s) > 0])
        except ValueError as e:
            print(len(all_biopsy_samples))
            print([len(a) for a in all_biopsy_samples])
            print(all_biopsy_samples)
            raise e
    return vaf_arr


def biopsy_sample(grid, ancestors, biopsy):
    # Assume the biopsies are square
    if biopsy is not None:
        x, y = biopsy['biopsy_origin']
        biopsy_edge = biopsy['biopsy_edge']
        x2, y2 = x + biopsy_edge, y + biopsy_edge
        assert x2 <= grid.shape[0]
        assert y2 <= grid.shape[1]
        biopsy = grid[x:x2, y:y2]
    else:
        biopsy = grid
        biopsy_edge = grid.shape[0]

    sample_counts = defaultdict(int)
    for i in biopsy.reshape(-1):
        mutants = get_mutants_from_clone_number(ancestors, i)
        for m in mutants:
            sample_counts[m] += 1

    vaf_arr = []  # Convert to VAF assuming heterozygous mutations
    for s, v in sample_counts.items():
        vaf_arr.append([s, v / (biopsy_edge ** 2) / 2])

    return np.array(vaf_arr)


def small_detection_limit(vafs_arr, coverage, limit):
    observed = []
    for clone, v in vafs_arr:
        v_obs = np.random.binomial(coverage, v)
        if v_obs >= limit:
            observed.append([clone, v_obs / coverage])

    return np.array(observed)


def get_mutants_from_clone_number(ancestors, clone_number):
    mutants = ancestors[clone_number]
    if 0 in mutants:
        mutants.remove(0)
    return mutants


def get_dnds(observed_vafs, sim):
    ns = 0
    s = 0

    for clone in observed_vafs[:, 0]:
        if clone in sim.ns_muts:
            ns += 1
        else:
            s += 1

    expected_ns = s * (1 / sim.mutation_generator.synonymous_proportion - 1)
    try:
        dnds = ns / expected_ns
        return dnds
    except ZeroDivisionError as e:
        return np.nan
