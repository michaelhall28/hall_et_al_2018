import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = [8, 6]
matplotlib.rcParams.update({'font.size': 22})


def standardise_x_values(res_dict, x_vals, with_biopsy, merged=False):
    """
    For calculation of the mean, make the x values (vafs/clone sizes) consistent
    
    For each of the x_vals, find the y_values from the nearest x_val to go with it.
    """
    if merged:
        if with_biopsy:
            x = res_dict['biop_merged_x_interval']
            y = res_dict['biop_merged_y_interval']
        else:
            x = res_dict['full_merged_x_interval']
            y = res_dict['full_merged_y_interval']
    elif with_biopsy:
        x = res_dict['biop_x_interval']
        y = res_dict['biop_y_interval']
    else:
        x = res_dict['full_x_interval']
        y = res_dict['full_y_interval']
    
    y_adj = []
    for val in x_vals:
        idx = (np.abs(x - val)).argmin()
        
        y_adj.append(y[idx])
        if x[idx] == x[-1]:  # Stop when reaching the end. Written this way in case last values are equal.
            break
    
    
    return y_adj


def plot_incomplete_moment_with_random_selection(all_res, x_vals, with_biopsy, merged=False, convert_to_clone_size=True,
                                                 biopsy_size=70**2, linecolour='k', rangecolour='y', num_shown=10,
                                                 output_file=None):
    max_clone_size = 0
    longest_incom = 0
    full_incom_array = np.zeros((len(all_res), len(x_vals)))

    # convert the x_values to vaf to match the form of the clone size data.
    # Will be converted back to clone size later.
    x_vals = x_vals / biopsy_size / 2

    if with_biopsy:
        k = 'biop_x_interval'
    else:
        k = 'full_x_interval'
    
    for i, (sim_name, res) in enumerate(all_res.items()):
        if len(res[k]) > 0:
            max_clone_size = max(max_clone_size, res[k].max())
            inc = standardise_x_values(res, x_vals, with_biopsy, merged)
            full_incom_array[i, :len(inc)] = inc
            longest_incom = max(len(inc), longest_incom)


    smallest_clone = 0
    for i in range(longest_incom):
        if full_incom_array[:, i].sum() < len(all_res) - 0.01:
            break
        smallest_clone = i


    x_vals = x_vals[smallest_clone:longest_incom]
    if convert_to_clone_size:
        x_vals *= biopsy_size * 2

    full_incom_array = full_incom_array[:, smallest_clone:longest_incom]
        
    means = full_incom_array.mean(axis=0)

    for i in range(num_shown):
        yy = full_incom_array[i][full_incom_array[i] > 0]
        xx = x_vals[:len(yy)]
        plt.plot(xx, yy, alpha=0.2, c=rangecolour)
        
    plt.plot(x_vals, means, c=linecolour, linewidth=2)

    plt.xlim(left=0)
    plt.yscale('log')
    
    plt.xlabel('Clone size (cells)')
    plt.ylabel('First Incomplete Moment')
    
    if output_file is not None:
        plt.tight_layout()
        plt.savefig(output_file)