#%%
from h5py._hl import selections
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size

class PlutoPython:
    """
    This class creates an opject for pluto output HDF5 files
    """
    def __init__(
        self,
        data_path,
        dpi,
        image_size,
    ):

        self.data_path = data_path
        self.data_list = None
        self.dpi = dpi
        self.image_size = image_size
        ### Classifier variables
        self.variables = None
        self.selector = None
        self.b_radial = None
        self.b_azimuthal = None
        self.b_axial = None
        self.pressure = None
        self.psi = None
        self.density = None
        self.tracer1 = None
        self.radial_velocity = None
        self.azimuthal_velocity = None
        self.axial_velocity = None

    def reader(self):
        path = os.getcwd()
        files  = os.listdir(self.data_path)
        extension = '.h5'
        self.data_list = [file for file in files if extension in file]
        
    def classifier(self):
        self.reader()
        data_file = self.data_list[70]
        h5_read = h5py.File(self.data_path + data_file, 'r')

        self.timestep, cell_coord, node_coord = h5_read.keys()
        
        data = h5_read[self.timestep]['vars']
        self.variables = list(data.keys())
        b_rad, b_azi, b_axi, pressure, psi, density, tracer1, v_rad, v_azi, v_axi = self.variables
        
        self.b_radial = b_rad
        self.b_azimuthal = b_azi
        self.b_axial = b_axi
        self.pressure = pressure
        self.psi = psi
        self.density = density
        self.tracer1 = tracer1
        self.radial_velocity = v_rad
        self.azimuthal_velocity = v_azi
        self.axial_velocity = v_axi
        
        self.selector = 3

        self.data = data[self.variables[self.selector]] # data[z][phi][rad]
        self.data_reshape = np.reshape(self.data, (600,300)).T

        grid = h5_read[cell_coord]
        
        self.radial_grid = [r[0] for r in list(np.reshape(grid['X'], (600, 300)).T)]
        self.axial_grid = np.reshape(grid['Z'], (600, 300)).T[0]


    def plot(self):
        self.classifier()
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)

        if self.variables[self.selector] == self.pressure: # or variables[selector] == density:
            im = axes.imshow(np.log(self.data_reshape), cmap='viridis')
            axes.contourf(self.axial_grid, self.radial_grid, np.log(self.data_reshape))
            axes.contour(self.axial_grid, self.radial_grid, np.log(self.data_reshape))
            plt.title(f'log({self.variables[self.selector]}) at {self.timestep.replace("_", " ")}')
        else:
            im = axes.imshow(self.data_reshape, cmap='viridis')
            axes.contourf(self.axial_grid, self.radial_grid, self.data_reshape)
            axes.contour(self.axial_grid, self.radial_grid, self.data_reshape)
            plt.title(f'{self.variables[self.selector]} at {self.timestep.replace("_", " ")}')

        figure.colorbar(im, location='right', shrink=0.83, aspect=20,pad=0.02)
        plt.xlim(0, max(self.axial_grid))
        plt.ylim(0, max(self.radial_grid))
        plt.ylabel('Radial distance in Jet Radii')
        plt.xlabel('Axial distance in Jet Radii')

        plt.show()
        

if __name__== "__main__":

    obj = PlutoPython('/mnt/f/OneDrive/ResearchProject/data/low_b_low_eta_outflow/', 300, (10,5))
    obj.plot()
