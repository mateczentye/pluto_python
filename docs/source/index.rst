.. Python3 PLUTO documentation master file, created by
   sphinx-quickstart on Sat Apr  2 02:13:20 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Python3 PLUTO's documentation!
=========================================
This package was written as part of the authors Master's thesis project, focusing on data analysis of the HDF5
files that are exported during simulation from PLUTO (http://plutocode.ph.unito.it/code-overview.html)



.. toctree::
   :maxdepth: 2
   :caption: Contents:


The code was written as part of the author's (M.I.Czentye) MPhys researh project to extend the methods used to analyse .h5 file output from PLUTO.

Current capability of the code is to analyse 2.5D datasets produced by solving the ideal MHD equations. The code consist of a Superclass
and a subclass, that is focusing on the MHD attributes that the jets can have.

Classes and methods
==========

pluto.py
^^^^^^^^
.. automodule:: pluto_python.pluto
   :members:
   :undoc-members:
   :private-members:
   
mhd_jet.py
^^^^^^^^^^
.. automodule:: pluto_python.mhd_jet
   :members:
   :undoc-members:
   :private-members:

caculator.py
^^^^^^^^^^^^
.. automodule:: pluto_python.calculator
   :members:
   :undoc-members:
   :private-members:

tools.py
^^^^^^^^
.. automodule:: pluto_python.tools
   :members:
   :undoc-members:
   :private-members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
