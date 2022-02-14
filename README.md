Package was made to analyse the h5 output files from PLUTO
Currently capable of analysing MHD simulations with no magnetic field splitting.
The main module, "analyse.py" has outgrown itself, hence process to split it into sub classes and modules has begun.
Package works with 2D data set the best. PLUTO can output 2.5D data sets which is also handled in here.

analyse.py: contains all the methods that I needed/wanted for my thesis work. (this will be legacy module once PIP installable, as all other modules/sub-classes will contain all its methods in a more user friendly way.)

pluto.py: contains the data reading and sorting methods to define all the variables that are used in other sub-classes to plot or calculate with.

mhd_jet.py: contains the subclass to py3Pluto to plot MHD simulation (no B-field splitting) data.

spacetime.py: contains the subclass that loops through all the data in the given directory to plot the space-time diagram - incomplete; currently use the one in analyse.py

tools.py: contains the functions which are non-physics related and are present for data manipulation