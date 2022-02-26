#%%
import numpy as np

def rearrange_axial_grid(initial_list, reference_list):
    return_list = []
    if type(initial_list) != list and type(initial_list) != tuple and type(initial_list) != np.ndarray:
        raise ValueError('Initial list must be a list or a tuple')
    elif type(reference_list) != list and type(reference_list) != tuple and type(initial_list) != np.ndarray:
        raise ValueError('Reference list must be a list or a tuple')
    else:
        previous_slice = 0
        for sub_list in reference_list:
            subl_len = len(sub_list)
            return_list.append(np.asanyarray(initial_list[previous_slice:previous_slice + subl_len]))

    return return_list

def find_mach_disk(array, grid):
    if type(array) != (list and np.ndarray):
        print(type(array))
        print(array)
        raise ValueError('A list must be the input to this function')
    if type(array[0]) == list:
        raise TypeError('List given must be a 1D array!')
    axial_disk_indicies = []
    for index, value in enumerate(array):
            if value > 0 and array[index-1] < 0:
                    axial_disk_indicies.append(index)

    return_x_values = [grid[x] for x in axial_disk_indicies]
    
    return return_x_values