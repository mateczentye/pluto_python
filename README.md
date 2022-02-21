[![Documentation Status](https://readthedocs.org/projects/pluto-python/badge/?version=latest)](https://pluto-python.readthedocs.io/en/latest/?badge=latest)

Ducumentation available here: https://pluto-python.readthedocs.io/en/latest/

Package was made to analyse the h5 output files from PLUTO

Package works with 2D data set. PLUTO can output 2.5D data sets which is also handled in here. Or 3D data set if one dimension is only 1 element long.
    So Grid shapes of (X, Z), (X, Y=1, Z)

pluto.py: contains the data reading and sorting methods to define all the variables that are used in other sub-classes to plot or calculate with.

mhd_jet.py: contains the subclass to py3Pluto to plot MHD simulation (no B-field splitting currently) data. This includes MHD shocks, space-time diagram and power/energy/energy density plots.

tools.py: contains the functions which are non-physics related and are present for data manipulation

calculator.py: Contains all the physics related functions and relations in PLUTO units