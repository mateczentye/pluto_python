#%%
from h5py._hl import selections
from matplotlib.colors import Colormap
from numpy.lib.arraypad import pad
import pandas
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.animation as pla

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
        
    ):

        self.data_path = data_path
        self.dpi = dpi
        self.image_size = image_size
        self.selector = select_variable
        self.time_step = time_step

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
            pl = axes.contourf(self.axial_grid, self.radial_grid, np.log(self.data_reshape), cmap='Spectral', levels=128)
            axes.set_title(f'log({self.variables[self.selector]}) at {self.timestep.replace("_", " ")}')

        else:
            pl = axes.contourf(self.axial_grid, self.radial_grid, self.data_reshape, cmap='Spectral', levels=128)
            axes.set_title(f'{self.variables[self.selector]} at {self.timestep.replace("_", " ")}')

        figure.colorbar(pl, location='right', shrink=0.83, aspect=20,pad=0.02)
        axes.set_xlim(0, max(self.axial_grid))
        axes.set_ylim(0, max(self.radial_grid))
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
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='Spectral', levels=128)
        
        figure.colorbar(pl, 
            location='right',  
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Magnetic Field magnitude', 
            format='%.2f'
        )

        axes.set_title(f'Magnetic field magnitude at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, max(self.axial_grid))
        axes.set_ylim(0, max(self.radial_grid))
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
        er = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='Spectral', levels=128)
        
        figure.colorbar(er, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='divB error megnitude', 
            format='%.2f'
            )
        
        axes.set_title(f'General Lagrangian Multiplier at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, max(self.axial_grid))
        axes.set_ylim(0, max(self.radial_grid))
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
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='Spectral', levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Magnetic Field magnitude', 
            format='%.2f'
            )
        
        axes.set_title(f'Magnetic field in x1 direction at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, max(self.axial_grid))
        axes.set_ylim(0, max(self.radial_grid))
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
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='Spectral', levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Magnetic Field magnitude', 
            format='%.2f'
            )
        
        axes.set_title(f'Magnetic field in x2 direction at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, max(self.axial_grid))
        axes.set_ylim(0, max(self.radial_grid))
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
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='Spectral', levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Magnetic Field magnitude', 
            format='%.2f'
            )
        
        axes.set_title(f'Magnetic field in x3 direction at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, max(self.axial_grid))
        axes.set_ylim(0, max(self.radial_grid))
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
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='Spectral', levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Pressure', 
            format='%.2f'
            )
        
        axes.set_title(f'Pressure at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, max(self.axial_grid))
        axes.set_ylim(0, max(self.radial_grid))
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
        pl = axes.contourf(self.axial_grid, self.radial_grid, np.log(data2plot), cmap='Spectral', levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='log(Pressure)', 
            format='%.2f'
            )
        
        axes.set_title(f'Log Pressure at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, max(self.axial_grid))
        axes.set_ylim(0, max(self.radial_grid))
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return data2plot

    def plot_density(self,close=False):
        """
        Plots the mass density field.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all')
        data2plot = np.reshape(data[self.density], (600, 300)).T
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='Spectral', levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Mass Density', 
            format='%.2f'
            )
        
        axes.set_title(f'Mass Density at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, max(self.axial_grid))
        axes.set_ylim(0, max(self.radial_grid))
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
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='Spectral', levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Tracer', 
            format='%.2f'
            )
        
        axes.set_title(f'Tracer at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, max(self.axial_grid))
        axes.set_ylim(0, max(self.radial_grid))
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
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='Spectral', levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Velocity, vx1', 
            format='%.2f'
            )
        
        axes.set_title(f'Velocity field in x1 direction at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, max(self.axial_grid))
        axes.set_ylim(0, max(self.radial_grid))
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
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='Spectral', levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Velocity, vx2', 
            format='%.2f'
            )
        
        axes.set_title(f'Velocity field in x2 direction at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, max(self.axial_grid))
        axes.set_ylim(0, max(self.radial_grid))
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
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='Spectral', levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Velocity, vx3', 
            format='%.2f'
            )
        
        axes.set_title(f'Velocity field in x3 direction at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, max(self.axial_grid))
        axes.set_ylim(0, max(self.radial_grid))
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
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='Spectral', levels=128)
        
        figure.colorbar(pl, 
            location='right',  
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Velocity Field magnitude', 
            format='%.2f'
        )

        axes.set_title(f'Velocity field magnitude at {self.timestep.replace("_", " ")}')
        axes.set_xlim(0, max(self.axial_grid))
        axes.set_ylim(0, max(self.radial_grid))
        axes.set_ylabel('Radial distance in Jet Radii')
        axes.set_xlabel('Axial distance in Jet Radii')

        if close==True:
            plt.close()
        return data2plot

    def velocity_quad(self,close=False):
        """
        Plots a graph with 4 plots of the velocity components and overall magnetude.
        """
        data = self.classifier(output_selector='all')

        data1 = self.plot_vx1(close=True)
        data2 = self.plot_vx2(close=True)
        data3 = self.plot_vx3(close=True)
        data4 = self.plot_velocity_field_magnitude(close=True)
        figure = plt.figure(figsize=(self.image_size[0]*2,self.image_size[1]*2), dpi=self.dpi)
        subfigs = figure.add_gridspec(2,2,hspace=0.15, wspace=0)
        axes = subfigs.subplots(sharex='col', sharey='row')
        figure.suptitle('Velocity Quad plot')
        axes[0,0].contourf(self.axial_grid, self.radial_grid, data1, cmap='Spectral', levels=128,label='vx1')
        axes[0,0].set_title('vx1', pad=0.5)
        axes[0,1].contourf(self.axial_grid, self.radial_grid, data2, cmap='Spectral', levels=128)
        axes[0,1].set_title('vx2', pad=0.5)
        axes[1,0].contourf(self.axial_grid, self.radial_grid, data3, cmap='Spectral', levels=128)
        axes[1,0].set_title('vx3', pad=0.5)
        axes[1,1].contourf(self.axial_grid, self.radial_grid, data4, cmap='Spectral', levels=128)
        axes[1,1].set_title('|v|', pad=0.5)
                
        if close==True:
            plt.close()

        return figure

if __name__== "__main__":

    #obj = PlutoPython('/mnt/f/OneDrive/ResearchProject/data/low_b_low_eta_outflow/', 300, (10,5), 5)
    obj = PlutoPython('/mnt/f/OneDrive/ResearchProject/data/Mms2_low_b/', time_step=18)
    
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
    obj.velocity_quad()