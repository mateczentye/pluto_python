#%%
import numpy as np

def magnetic_field(sigma_z, sigma_phi):
    a = 0.8
    Pa = 0.6

    b2_z = 2 * sigma_z * Pa
    b2_m = (2 * sigma_phi * Pa) / (a**2 * (0.5 - 2*np.log(a)))
    print(f'B = {np.sqrt(b2_z + b2_m)}')
    return np.sqrt(b2_z + b2_m)

def magnetosonic_speed(eta, kappa, b):

    return np.sqrt((kappa + b**2)/eta)

def alfven_velocity(B, density):
    return B / np.sqrt(density)

def get_magnitude(list):
    sum = 0
    for element in list:
        sum += element**2
    return np.sqrt(sum)