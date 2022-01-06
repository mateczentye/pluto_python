#%%
import numpy as np

v_jet = 12.328828005937952 


def magnetic_field(sigma_z, sigma_phi):
    a = 0.8
    Pa = 0.6

    b2_z = 2 * sigma_z * Pa
    b2_m = (2 * sigma_phi * Pa) / (a**2 * (0.5 - 2*np.log(a)))
    print(f'B = {np.sqrt(b2_z + b2_m)}')
    return np.sqrt(b2_z + b2_m)

def magnetosonic_speed(eta, kappa, b):

    return np.sqrt((kappa + b**2)/eta)


vms = magnetosonic_speed(
    eta = 0.1,
    kappa = 2,
    b = magnetic_field(
        sigma_z = 0.01,
        sigma_phi = 0.01

        )
)

print(v_jet/vms)
print(f'set v_jet = {2*vms} to have M_ms = 2')