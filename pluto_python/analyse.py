#%%
from h5py._hl import selections
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap
from numpy.lib.arraypad import pad
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
import pandas
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.animation as pla
import matplotlib

class PlutoPython:
    """
    This class creates an opject for pluto output HDF5 files
    """
    def __init__(
        self,
        data_path,
        time_step,
        select_variable = None,
        dpi = 300,
        image_size = (10,5),
        ylim = (0,10),
        xlim = (0,20),
        cmap = 'Spectral'
        
    ):

        self.data_path = data_path
        self.dpi = dpi
        self.image_size = image_size
        self.selector = select_variable
        self.time_step = time_step
        self.xlim = xlim
        self.ylim = ylim
        self.cmap = cmap

        self.closure = False

        ### Classifier variables
        self.variables = None
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
        
    def classifier(self, output_selector=None):
        self.reader()
        data_file = self.data_list[self.time_step]
        h5_read = h5py.File(self.data_path + data_file, 'r')

        self.timestep, cell_coord, node_coord = h5_read.keys()
        
        data = h5_read[self.timestep]['vars']
        self.variables = list(data.keys())
        b_rad, b_azi, b_axi, pressure, psi_glm, density, tracer1, v_rad, v_azi, v_axi = self.variables
        
        self.b_radial = b_rad
        self.b_azimuthal = b_azi
        self.b_axial = b_axi
        self.pressure = pressure
        self.psi_glm = psi_glm
        self.density = density
        self.tracer1 = tracer1
        self.radial_velocity = v_rad
        self.azimuthal_velocity = v_azi
        self.axial_velocity = v_axi

        grid = h5_read[cell_coord]
            
        self.radial_grid = [r[0] for r in list(np.reshape(grid['X'], (600, 300)).T)]
        self.axial_grid = np.reshape(grid['Z'], (600, 300)).T[0]

        if output_selector == None:
            self.data = data[self.variables[self.selector]] # data[z][phi][rad]
            self.data_reshape = np.reshape(self.data, (600,300)).T
            return [self.data_reshape, self.axial_grid, self.axial_grid]

        elif output_selector == 'all':
            return data


    def plot(self,close=False):
        """
        General plotting with more arguments to select what data set to plot.
        
        Returns a matplotlib plot object
        """
        if self.selector == None:
            msg_str = 'Please select a variable when initialising the class to use the plot() method\n \
               or use one of the specific plotting methods'
            raise AttributeError(msg_str)
            return None

        self.classifier()
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)

        if self.variables[self.selector] == self.pressure: # or variables[selector] == density:
            pl = axes.contourf(self.axial_grid, self.radial_grid, np.log(self.data_reshape), cmap=self.cmap, levels=128)
            axes.set_title(f'log({self.variables[self.selector]}) at {self.timestep.replace("_", " ")}')

        else:
            pl = axes.contourf(self.axial_grid, self.radial_grid, self.data_reshape, cmap=self.cmap, levels=128)
            axes.set_title(f'{self.variables[self.selector]} at {self.timestep.replace("_", " ")}')

        figure.colorbar(pl, location='right', shrink=0.83, aspect=20,pad=0.02)
        axes.set_xlim(0, self.xlim[1])
        axes.set_ylim(0, self.ylim[1])
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        return pl

    def plot_bfield_magnitude(self,close=False):
        """
        plots the magnitude of the magnetic fields, with no direction,
        calculated from the 3 components produced by PLUTO

        Returns the data set that is being plotted
        """
        data = self.classifier(output_selector='all')
        bx1 = np.reshape(data[self.b_radial], (600, 300)).T
        bx2 = np.reshape(data[self.b_azimuthal], (600, 300)).T
        bx3 = np.reshape(data[self.b_axial], (600, 300)).T
        data2plot = np.sqrt(np.asarray(bx1)**2 + np.asarray(bx2)**2 + np.asarray(bx3)**2)
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=128)
        
        figure.colorbar(pl, 
            location='right',  
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Magnetic Field magnitude', 
            format='%.2f'
        )

        axes.set_title(f'Magnetic field magnitude at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, self.xlim[1])
        axes.set_ylim(0, self.ylim[1])
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return data2plot

    def plot_glm(self,close=False):
        """
        Plots the General Lagrangian Multiplier which is an output due to the divB control
        
        Returns the data set that is being plotted
        """     
        data = self.classifier(output_selector='all')
        data2plot = np.reshape(data[self.variables[4]], (600, 300)).T
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        er = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=128)
        
        figure.colorbar(er, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='divB error megnitude', 
            format='%.2f'
            )
        
        axes.set_title(f'General Lagrangian Multiplier at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, self.xlim[1])
        axes.set_ylim(0, self.ylim[1])
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return data2plot

    def plot_bx1(self,close=False):
        """
        Plots the magnetic field in the first defined axis direction, x1 from the init file.
        
        Returns the data set that is being plotted
        """    
        data = self.classifier(output_selector='all')
        data2plot = np.reshape(data[self.b_radial], (600, 300)).T
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Magnetic Field magnitude', 
            format='%.2f'
            )
        
        axes.set_title(f'Magnetic field in x1 direction at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, self.xlim[1])
        axes.set_ylim(0, self.ylim[1])
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return data2plot

    def plot_bx2(self,close=False):
        """
        Plots the magnetic field in the second defined axis direction, x2 from the init file.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all')
        data2plot = np.reshape(data[self.b_azimuthal], (600, 300)).T
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Magnetic Field magnitude', 
            format='%.2f'
            )
        
        axes.set_title(f'Magnetic field in x2 direction at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, self.xlim[1])
        axes.set_ylim(0, self.ylim[1])
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return data2plot

    def plot_bx3(self,close=False):
        """
        Plots the magnetic field in the third defined axis direction, x3 from the init file.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all')
        data2plot = np.reshape(data[self.b_axial], (600, 300)).T
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Magnetic Field magnitude', 
            format='%.2f'
            )
        
        axes.set_title(f'Magnetic field in x3 direction at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, self.xlim[1])
        axes.set_ylim(0, self.ylim[1])
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return data2plot

    def plot_pressure(self,close=False):
        """
        Plots the pressure field.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all')
        data2plot = np.reshape(data[self.pressure], (600, 300)).T
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Pressure', 
            format='%.2f'
            )
        
        axes.set_title(f'Pressure at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, self.xlim[1])
        axes.set_ylim(0, self.ylim[1])
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return data2plot

    def plot_log_pressure(self,close=False):
        """
        Plots the log pressure field.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all')
        data2plot = np.reshape(data[self.pressure], (600, 300)).T
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, np.log(data2plot), cmap=self.cmap, levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='log(Pressure)', 
            format='%.2f'
            )
        
        axes.set_title(f'Log Pressure at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, self.xlim[1])
        axes.set_ylim(0, self.ylim[1])
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return np.log(data2plot)

    def plot_density(self,close=False):
        """
        Plots the mass density field.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all')
        data2plot = np.reshape(data[self.density], (600, 300)).T
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Mass Density', 
            format='%.2f'
            )
        
        axes.set_title(f'Mass Density at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, self.xlim[1])
        axes.set_ylim(0, self.ylim[1])
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return data2plot

    def plot_tracer(self,close=False):
        """
        Plots the tracer.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all')
        data2plot = np.reshape(data[self.tracer1], (600, 300)).T
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Tracer', 
            format='%.2f'
            )
        
        axes.set_title(f'Tracer at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, self.xlim[1])
        axes.set_ylim(0, self.ylim[1])
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return data2plot

    def plot_vx1(self,close=False):
        """
        Plots the velocity field in x1 direction.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all')
        data2plot = np.reshape(data[self.radial_velocity], (600, 300)).T
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Velocity, vx1', 
            format='%.2f'
            )
        
        axes.set_title(f'Velocity field in x1 direction at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, self.xlim[1])
        axes.set_ylim(0, self.ylim[1])
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return data2plot

    def plot_vx2(self,close=False):
        """
        Plots the velocity field in x2 direction.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all')
        data2plot = np.reshape(data[self.azimuthal_velocity], (600, 300)).T
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Velocity, vx2', 
            format='%.2f'
            )
        
        axes.set_title(f'Velocity field in x2 direction at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, self.xlim[1])
        axes.set_ylim(0, self.ylim[1])
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return data2plot

    def plot_vx3(self,close=False):
        """
        Plots the velocity field in x3 direction.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all')
        data2plot = np.reshape(data[self.axial_velocity], (600, 300)).T
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Velocity, vx3', 
            format='%.2f'
            )
        
        axes.set_title(f'Velocity field in x3 direction at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, self.xlim[1])
        axes.set_ylim(0, self.ylim[1])
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return data2plot

    def plot_velocity_field_magnitude(self,close=False):
        """
        Plots the magnitude of the velocity fields, with no direction,
        calculated from the 3 components produced by PLUTO

        Returns the data set that is being plotted
        """
        data = self.classifier(output_selector='all')
        vx1 = np.reshape(data[self.radial_velocity], (600, 300)).T
        vx2 = np.reshape(data[self.azimuthal_velocity], (600, 300)).T
        vx3 = np.reshape(data[self.axial_velocity], (600, 300)).T
        data2plot = np.sqrt(np.asarray(vx1)**2 + np.asarray(vx2)**2 + np.asarray(vx3**2))
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=128)
        
        figure.colorbar(pl, 
            location='right',  
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Velocity Field magnitude', 
            format='%.2f'
        )

        axes.set_title(f'Velocity field magnitude at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, self.xlim[1])
        axes.set_ylim(0, self.ylim[1])
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return data2plot

    def _velocity_quad(self,close=False):
        """
        Plots a graph with 4 plots of the velocity components and overall magnetude.

        Returns a matlpotlib figure object
        """
        data1 = self.plot_vx1(close=True)
        data2 = self.plot_vx2(close=True)
        data3 = self.plot_vx3(close=True)
        data4 = self.plot_velocity_field_magnitude(close=True)
        min_val = 0
        for x in [data1,data2,data3,data4]:
            if np.min(x) < min_val:
                min_val = np.min(x)
        
        negative_levels = np.linspace(-np.max(data4),0,64)
        positive_levels = np.linspace(0,np.max(data4),65)
        
        levels = np.concatenate([negative_levels, positive_levels[1:]])
        scalar = 1.1
        figure, axes = plt.subplots(figsize=(self.image_size[0]*scalar, self.image_size[1]*scalar), dpi=self.dpi)
        cax = plt.axes([1.01, 0.05, 0.05, 0.9], label='velocity')

        plt.tight_layout()        
        plt.subplot(221, title='vx1', ylabel=r'radial distance [$R_{jet}$]', ylim=self.ylim, xlim=self.xlim)
        plt.contourf(self.axial_grid, self.radial_grid, data1, cmap=self.cmap, levels=levels)
        plt.subplot(222, title='vx2', ylim=self.ylim, xlim=self.xlim)
        plt.contourf(self.axial_grid, self.radial_grid, data2, cmap=self.cmap, levels=levels)
        plt.subplot(223, title='vx3', xlabel=r'axial distance [$R_{jet}$]', ylabel=r'radial distance [$R_{jet}$]', ylim=self.ylim, xlim=self.xlim)
        plt.contourf(self.axial_grid, self.radial_grid, data3, cmap=self.cmap, levels=levels)
        plt.subplot(224, title='|v|', xlabel=r'axial distance [$R_{jet}$]', ylim=self.ylim, xlim=self.xlim)
        plt.contourf(self.axial_grid, self.radial_grid, data4, cmap=self.cmap, levels=levels)
        plt.colorbar(cax=cax, format='%.2f', label='Velocity')
        
        if close==True:
            plt.close()

        return figure

    def _magneticfield_quad(self,close=False):
        """
        Plots a graph with 4 plots of the velocity components and overall magnetude.

        Returns a matlpotlib figure object
        """
        data1 = self.plot_bx1(close=True)
        data2 = self.plot_bx2(close=True)
        data3 = self.plot_bx3(close=True)
        data4 = self.plot_bfield_magnitude(close=True)

        min_val = 0
        for x in [data1,data2,data3,data4]:
            if np.min(x) < min_val:
                min_val = np.min(x)
        
        negative_levels = np.linspace(-np.max(data4),0,64)
        positive_levels = np.linspace(0,np.max(data4),65)
        
        levels = np.concatenate([negative_levels, positive_levels[1:]])
        scalar = 1.1
        figure, axes = plt.subplots(figsize=(self.image_size[0]*scalar, self.image_size[1]*scalar), dpi=self.dpi)
        cax = plt.axes([1.01, 0.05, 0.05, 0.9], label='magnetic field')
        
        plt.tight_layout()
        plt.subplot(221, title='bx1', ylabel=r'radial distance [$R_{jet}$]', ylim=self.ylim, xlim=self.xlim)
        plt.contourf(self.axial_grid, self.radial_grid, data1, cmap=self.cmap, levels=levels)
        plt.subplot(222, title='bx2', ylim=self.ylim, xlim=self.xlim)
        plt.contourf(self.axial_grid, self.radial_grid, data2, cmap=self.cmap, levels=levels)
        plt.subplot(223, title='bx3', xlabel=r'axial distance [$R_{jet}$]', ylabel=r'radial distance [$R_{jet}$]', ylim=self.ylim, xlim=self.xlim)
        plt.contourf(self.axial_grid, self.radial_grid, data3, cmap=self.cmap, levels=levels)
        plt.subplot(224, title='|B|', xlabel=r'axial distance [$R_{jet}$]', ylim=self.ylim, xlim=self.xlim)
        plt.contourf(self.axial_grid, self.radial_grid, data4, cmap=self.cmap, levels=levels)
        plt.colorbar(cax=cax, format='%.2f', label='Magnetif Field')
        
        if close==True:
            plt.close()

        return figure

    def plot_pressure_density(self,close=False):
        """
        Plots a pressure and density plot on one figure

        Returns None
        """
        data = self.classifier(output_selector='all')
        density_data = np.reshape(data[self.density], (600, 300)).T
        pressure_data = np.reshape(data[self.pressure], (600, 300)).T
        
        figure, axes = plt.subplots(2,1,figsize=self.image_size, dpi=self.dpi)
        self.cmap = 'hot'
        density_levels = np.linspace(0, np.max(density_data), 128)
        
        axes[0].contourf(self.axial_grid, self.radial_grid, density_data, cmap=self.cmap, levels=density_levels)
        im1 = axes[0].imshow(density_data, cmap=self.cmap)
        divider = mal(axes[0])
        cax = divider.append_axes('right',size='5%',pad=0.05)
        plt.colorbar(im1,cax,ticks=np.linspace(0,np.max(density_data), 6))
        axes[0].set_title(f'Mass Density at {self.timestep.replace("_", " ")}')
        axes[0].set_xlim(0, self.xlim[1])
        axes[0].set_ylim(0, self.ylim[1])
        axes[0].set_ylabel('Radial distance in Jet Radii')
        axes[0].set_xlabel('Axial distance in Jet Radii')
        
        axes[1].contourf(self.axial_grid, self.radial_grid, pressure_data, cmap=self.cmap, levels=128)
        im1 = axes[1].imshow(pressure_data, cmap=self.cmap)
        divider = mal(axes[1])
        cax = divider.append_axes('right',size='5%',pad=0.05)
        plt.colorbar(im1,cax,ticks=np.linspace(0,np.max(pressure_data), 6))
        axes[1].set_title(f'Pressure at {self.timestep.replace("_", " ")}')
        axes[1].set_xlim(0, self.xlim[1])
        axes[1].set_ylim(0, self.ylim[1])
        axes[1].set_ylabel('Radial distance in Jet Radii')
        axes[1].set_xlabel('Axial distance in Jet Radii')

        plt.subplots_adjust(left=0.125,
                            right=0.9,
                            bottom=-0.05,
                            top=0.9,
                            wspace=0.2,
                            hspace=0.35)

        if close==True:
            plt.close()

        return None
    
    def magneticfield_quad(self,close=False):
        """
        Plots a graph with 4 plots of the magnetic field components and overall magnitude.

        Returns a matlpotlib figure object
        """
        data1 = self.plot_bx1(close=True)
        data2 = self.plot_bx2(close=True)
        data3 = self.plot_bx3(close=True)
        data4 = self.plot_bfield_magnitude(close=True)

        min_val = 0
        for x in [data1,data2,data3,data4]:
            if np.min(x) < min_val:
                min_val = np.min(x)
                
        negative_levels = np.linspace(min_val,0,64)
        positive_levels = np.linspace(0,np.max(data4),65)
        
        levels = np.concatenate([negative_levels, positive_levels[1:]])
                
        figure, axes = plt.subplots(2,2,figsize=(self.image_size[0]*1.2, self.image_size[1]*1.8), dpi=self.dpi)
        cax = plt.axes([1, 0.05, 0.05, 0.9], label='magnetic field')
        
        plt.tight_layout()
        #plt.subplot(221, title='bx1', ylabel=r'radial distance [$R_{jet}$]', ylim=self.ylim, xlim=self.xlim)
        axes[0][0].contourf(self.axial_grid, self.radial_grid, data1, cmap=self.cmap, levels=levels)
        axes[0][0].set_title(f'Magnetic field in x1 direction at {self.timestep.replace("_", " ")}')
        axes[0][0].set_xlim(0, self.xlim[1])
        axes[0][0].set_ylim(0, self.ylim[1])
        axes[0][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][0].set_xlabel(r'Axial distance [$R_{jet}$]')
       
        axes[0][1].contourf(self.axial_grid, self.radial_grid, data2, cmap=self.cmap, levels=levels)
        axes[0][1].set_title(f'Magnetic field in x2 direction at {self.timestep.replace("_", " ")}')
        axes[0][1].set_xlim(0, self.xlim[1])
        axes[0][1].set_ylim(0, self.ylim[1])
        axes[0][1].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][1].set_xlabel(r'Axial distance [$R_{jet}$]')

        axes[1][0].contourf(self.axial_grid, self.radial_grid, data3, cmap=self.cmap, levels=levels)
        axes[1][0].set_title(f'Magnetic field in x3 direction at {self.timestep.replace("_", " ")}')
        axes[1][0].set_xlim(0, self.xlim[1])
        axes[1][0].set_ylim(0, self.ylim[1])
        axes[1][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[1][0].set_xlabel(r'Axial distance [$R_{jet}$]')
        
        axes[1][1].contourf(self.axial_grid, self.radial_grid, data4, cmap=self.cmap, levels=levels)
        axes[1][1].set_title(f'Magnetic field magnitude at {self.timestep.replace("_", " ")}')
        axes[1][1].set_xlim(0, self.xlim[1])
        axes[1][1].set_ylim(0, self.ylim[1])
        axes[1][1].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[1][1].set_xlabel(r'Axial distance [$R_{jet}$]')
        
        plt.subplots_adjust(left=0.0,
                            right=0.95,
                            bottom=0.1,
                            top=0.9,
                            wspace=0.15,
                            hspace=0.25)

        norm = matplotlib.colors.Normalize(vmin=min_val, vmax=np.max(data4))
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=self.cmap), cax=cax, format='%.2f', label='Magnetic Field')
        
        if close==True:
            plt.close()

        return figure

    def velocity_quad(self,close=False):
        """
        Plots a graph with 4 plots of the velocity components and overall magnitude.

        Returns a matlpotlib figure object
        """
        data1 = self.plot_vx1(close=True)
        data2 = self.plot_vx2(close=True)
        data3 = self.plot_vx3(close=True)
        data4 = self.plot_velocity_field_magnitude(close=True)

        min_val = 0
        for x in [data1,data2,data3,data4]:
            if np.min(x) < min_val:
                min_val = np.min(x)
                
        negative_levels = np.linspace(-np.max(data4),0,64)
        positive_levels = np.linspace(0,np.max(data4),65)
        
        levels = np.concatenate([negative_levels, positive_levels[1:]])
                
        figure, axes = plt.subplots(2,2,figsize=(self.image_size[0]*1.2, self.image_size[1]*1.8), dpi=self.dpi)
        cax = plt.axes([1, 0.05, 0.05, 0.9], label='Velocity field')
        
        plt.tight_layout()
        #plt.subplot(221, title='bx1', ylabel=r'radial distance [$R_{jet}$]', ylim=self.ylim, xlim=self.xlim)
        axes[0][0].contourf(self.axial_grid, self.radial_grid, data1, cmap=self.cmap, levels=levels)
        axes[0][0].set_title(f'Velocity field in x1 direction at {self.timestep.replace("_", " ")}')
        axes[0][0].set_xlim(0, self.xlim[1])
        axes[0][0].set_ylim(0, self.ylim[1])
        axes[0][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][0].set_xlabel(r'Axial distance [$R_{jet}$]')
       
        axes[0][1].contourf(self.axial_grid, self.radial_grid, data2, cmap=self.cmap, levels=levels)
        axes[0][1].set_title(f'Velocity field in x2 direction at {self.timestep.replace("_", " ")}')
        axes[0][1].set_xlim(0, self.xlim[1])
        axes[0][1].set_ylim(0, self.ylim[1])
        axes[0][1].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][1].set_xlabel(r'Axial distance [$R_{jet}$]')

        axes[1][0].contourf(self.axial_grid, self.radial_grid, data3, cmap=self.cmap, levels=levels)
        axes[1][0].set_title(f'Velocity field in x3 direction at {self.timestep.replace("_", " ")}')
        axes[1][0].set_xlim(0, self.xlim[1])
        axes[1][0].set_ylim(0, self.ylim[1])
        axes[1][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[1][0].set_xlabel(r'Axial distance [$R_{jet}$]')
        
        axes[1][1].contourf(self.axial_grid, self.radial_grid, data4, cmap=self.cmap, levels=levels)
        axes[1][1].set_title(f'Velocity field magnitude at {self.timestep.replace("_", " ")}')
        axes[1][1].set_xlim(0, self.xlim[1])
        axes[1][1].set_ylim(0, self.ylim[1])
        axes[1][1].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[1][1].set_xlabel(r'Axial distance [$R_{jet}$]')
        
        plt.subplots_adjust(left=0.0,
                            right=0.95,
                            bottom=0.1,
                            top=0.9,
                            wspace=0.15,
                            hspace=0.25)

        norm = matplotlib.colors.Normalize(vmin=min_val, vmax=np.max(data4))
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=self.cmap), cax=cax, format='%.2f', label='Magnetic Field')
        
        if close==True:
            plt.close()

        return figure

if __name__== "__main__":

    #obj = PlutoPython('/mnt/f/OneDrive/ResearchProject/data/low_b_low_eta_outflow/', 300, (10,5), 5)
    obj = PlutoPython('/mnt/f/OneDrive/ResearchProject/data/Mms2_low_b/', time_step=27, cmap='seismic')
    
    #obj.plot()
    #obj.plot_bfield_magnitude()
    #obj.plot_glm()
    #obj.plot_log_pressure()
    #obj.plot_density()
    #obj.plot_tracer()
    #obj.plot_vx1()
    #obj.plot_vx2()
    #obj.plot_vx3()
    #obj.plot_velocity_field_magnitude()
    #obj.velocity_quad()
    obj.magneticfield_quad()
    obj.velocity_quad()
    #obj.plot_pressure_density()
    #obj.plot_bx1()