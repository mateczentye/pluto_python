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

def magneto_acoustic_velocity(b, prs, rho, gamma):
    term1 = 1/(2*rho)
    term2 = gamma * prs
    root_term = term2**2 + b**4 - 2*term2*(b**2)
   
    slow = term1 * (term2 + b**2 - np.sqrt(root_term))
    fast = term1 * (term2 + b**2 + np.sqrt(root_term))
   
    return [np.sqrt(slow), np.sqrt(fast)]

def mach_number(fluid_velocity, wave_velocity):
    return fluid_velocity/wave_velocity

def energy_density(pressure, density, velocity, magnetic_field, gamma):
    pot = pressure / (gamma - 1)
    kin = 0.5 * density * (velocity**2)
    mag = 0.5 * (magnetic_field**2)
    return [pot, kin, mag]

def magnetic_pressure(mag_field):
    return 0.5*(mag_field**2)


