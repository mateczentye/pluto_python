Examples of Visualisation and Analysis tools
============================================

Plot method
-----------

Plot method is used to plot PLUTO native outputs that are contained within the 
h5 files, or any calculated data from the superclass such as sonic wave speeds 
or mach numbers. 
All controlled via a string passed as data2plot argument. The key values are 
stored in the object's attribute "varname_dict", if none given, the keys are 
printed in the console and a value error is raised.

.. automethod:: pluto_python.mhd_jet.plot

Hist method
-----------

The histogram takes the whole spatial domain, and counts across all cells to 
compose the histogram data. The data is chosen by the data2plot argument that 
must be passed when calling the method. All valid strings are contained in the
"varname_dict" object attribute which is a dictionary. If none given the key 
values are printed in the console and a value error is raised.

.. automethod:: pluto_python.mhd_jet.hist

Shocks method
-------------

The shocks method is used to plot magnetohydrodynamic strong shocks, as it is 
categorised in literature quite often (doi.org/10.1063/1.873124). The plot_shock 
argument takes a string containing number pairs based on the transition of 
supersonic, to subsonic across all three wave modes in MHD case. In hydrodynamic
case with no magnetic field the '12' also known as fast shock replaces all MHD 
shocks due to its singular wave mode.
Possible values are '12, 13, 14, 23, 24, 34' but any combination of these 2 digit
numbers within the given string will plot the defined shocks (fast, slow and a 
range of intermediate).
All shocked cells location are stored in the object's shocks_list attribute as
(X, Y) coordinates, where X and Y are lists.

.. automethod:: pluto_python.mhd_jet.shocks

plot_spacetime method
---------------------

Space-time diagram is used to give a 2D representation of the full or partial 
simulation timelapse. It will produce a full read of all data files within the 
chosen time steps inside the working directory. 

.. automethod:: pluto_python.mhd_jet.plot_spacetime

plot_power method
-----------------

The power plot method plots the integral of the power at each radial slice 
across the computational domain. It also initializes the "list_power" object 
attribute that contains the list of integrated values, in the following sequence:
Total system power, total jet power, kinteic jet power, thermal jet power and 
magnetic jet power

.. automethod:: pluto_python.mhd_jet.plot_power

plot_energy method
------------------

The energy plot method plots the integral of the energy at each radial slice 
across the computational domain. It also initializes the "list_energy" object
attribute that contains the list of integrated values, in the following sequence:
Total system energy, total jet energy, kinteic jet energy, thermal jet energy and 
magnetic jet energy

.. automethod:: pluto_python.mhd_jet.plot_energy

plot_energy_density method
--------------------------

The energy plot method plots the integral of the energy density at each radial 
slice across the computational domain. It also initializes the "list_E_dens" 
object attribute that contains the list of integrated values, in the following 
sequence: Total jet energy density, kinteic jet energy density, thermal jet 
energy density and magnetic jet energy density

.. automethod:: pluto_python.mhd_jet.plot_energy_density

plot_fieldlines method
----------------------

This method was created to find entangled magnetic field lines at low 
magnetisation. This is unique to the MHD data, and can only be ploted at a 
limited uniform sub-grid as defined in PLUTO.
It plots the field lines as a stream plot, with the magnetic field magnitude as
a countour plot in the background.

.. automethod:: pluto_python.mhd_jet.plot_fieldlines

plot_azimuthal_energy method
----------------------------

This method was created for checking jet stability based on Magnetic-Kinetic
energy comparison for a self balancing structure as described in 
doi.org/10.1007/BF00642266

.. automethod:: pluto_python.mhd_jet.plot_azimuthal_energy

oblique_shocks method
---------------------

Here the oblique shocks' strength are plotted as a contour plot. The shock 
calculations are based on the Ranking-Hugoniot relation for an ideal MHD system.
It takes in two arguments, one is the minimum of the shock values, the other is
the maximum of the shock values in unit less quantity, based on the mach number.

.. automethod:: pluto_python.mhd_jet.oblique_shocks