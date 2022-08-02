#%%
"""
Contains physics relevant functions that are called in one or more of the classes within the package.

    - Module specific functions get defined here like magnetic pressure, or plasma beta
    - Current version contains all used physics equations, may break it into classes to categorise them later.
    
"""
import numpy as np

def magnetic_field(sigma_z, sigma_phi, kappa=2):
    """
    Calculates the magnetic field strength from the magnetisation of the jet, as defined in the init.c in the 
    PLUTO directory and used to compile the simulator with the right physics relations.
    """
    a = 0.8
    Pa = 0.6

    b2_z = 2 * sigma_z * Pa
    b2_m = (2 * sigma_phi * Pa)*kappa / (a**2 * (0.5 - 2*np.log(a)))
    
    return np.sqrt(b2_z + b2_m)

def magnetosonic_speed(eta, kappa, b):
    """
    Returns the magneto-sonic speed from the environmental variables like jet overpressure, and density ratio
    """
    return np.sqrt((kappa + b**2)/eta)

def alfven_velocity(B, density):
    """
    Returns the alvén wave velocity from the magnetic field and the density in each stencil.
    """
    return B / np.sqrt(density)

def magneto_acoustic_velocity(b, prs, rho, gamma):
    """
    Magneto aucustoc velocity from computational stencil values of pressure, density, adiabatic constant and
    magnetic fields.
    """
    term1 = 1/(2*rho)
    term2 = gamma * prs
    root_term = term2**2 + b**4 - 2*term2*(b**2)
   
    slow = term1 * (term2 + b**2 - np.sqrt(root_term))
    fast = term1 * (term2 + b**2 + np.sqrt(root_term))
   
    return [np.sqrt(slow), np.sqrt(fast)]

def mach_number(fluid_velocity, wave_velocity):
    """
    Returns the wave specific Mach number, in each stencil
    """ 
    return fluid_velocity/wave_velocity

def energy_density(pressure, density, velocity, gamma, magnetic_field=None):
    """
    Calculates the Energy density of each stencil and returns a list of thermal/potential, kinetic and magnetic energy
    density respectively [pot, kin, mag].
    """
    pot = (pressure * gamma) / (gamma - 1)
    kin = 0.5 * density * (velocity**2)

    if type(magnetic_field) is type(None):
        mag = np.zeros_like(pot)
    else:
        mag = 0.5 * (magnetic_field**2)

    return [pot, kin, mag]

def magnetic_pressure(mag_field):
    """
    Calculates magnetic pressure from ideal conditions given the magnetic field strength.
    """
    return 0.5*(mag_field**2)

def sound_speed(gamma, prs, rho):
    """
    Returns the sound speed of each stencil from pressure, density and adiabatic constant
    """
    return np.sqrt((gamma * prs) / rho)

def effective_kappa(therma_prs, magnetic_prs, ambient_prs):
    """
    Calculates an effective over pressure, due to the thermal, magnetic and ambient contributions.
    """
    return (therma_prs + magnetic_prs)/ambient_prs

def RH_MHD(gamma, beta, mach, prs1, prs2):
    """
    This calculates the Ranking-Huginoit relation (both values of the quadratic) as a function of Mach number squared
    """
    gm2 = gamma * mach**2
    y = prs1 / prs2

    x1 = -0.5*beta * (gm2 + (gm2**2 + ((4 * (gm2 + 1 + y)) / beta))**0.5)
    x2 = (0.5*beta) * (-gm2 + (gm2**2 + ((4 * (gm2 + 1 + y)) / beta))**0.5)
    return [x1, x2]