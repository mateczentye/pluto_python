Quick Tour of the package
=========================

Usage - py3Pluto
----------------

The superclass contains all the variables that are calculated and ready to be
visualised via the subclass. This can be utilised for methods that require 
unique visualisation tools custom made by the user, by calling out all relevent
data in the initialised object.

.. code-block:: python

    import pluto_python as pp
    pp.py3Pluto('<data_path>')

Once the object is initialised, the object attributes can be used to plot any 
desired graphical representation of the chosen attribute.

This could include multi-plot figures which were part of the initial code, but
was dropped at version 0.0.2 to enable greater flexibility of plotting methods.


Usage - mhd_jet
---------------

.. code-block:: python

    import pluto_python as pp
    pp.mhd_jet('<data_path>')

Now the built in visualisation methods can be utilised with great flexibility,
by invoking the method of choice with the correct/desired arguments to plot a 
wide range of graphs and data that are calculated at the superclass level.