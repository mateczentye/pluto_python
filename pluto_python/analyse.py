#%%
from h5py._hl import selections
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap
from numpy.lib.arraypad import pad
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
from pluto_python.calculator import get_magnitude, alfven_velocity
import pandas
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.animation as pla
import matplotlib
import os
#import calculator as calc

class PlutoPython:
    """
    This class creates an opject for pluto output HDF5 files
    Use methods for each variable to be plotted at certain time step.
    Group variables have multi-plot graph options as separate methods.
    """
    def __init__(
        self,
        data_path,
        time_step,
        ini_path = None,
        select_variable = None,
        dpi = 300,
        image_size = (10,5),
        ylim = None,
        xlim = None,
        cmap = 'bwr',
        global_limits = False,
        
    ):

        self.data_path = data_path
        self.dpi = dpi
        self.image_size = image_size
        self.selector = select_variable
        self.time_step = time_step
        self.global_limits_bool = global_limits
        self.cmap = cmap
        self.ini_path = ini_path
        
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

        self.reader()
        self.shape_limits = self.read_init()
        self.XZ_shape = (self.shape_limits['X3-grid'], self.shape_limits['X1-grid'])

        ### Define axes limits from defaults from the ini file if not given. To see max grid,
        if xlim == None:
            self.xlim = (
                    float(self.ini_content['[Grid]']['X3-grid']['Subgrids Data'][0][0]),
                    float(self.ini_content['[Grid]']['X3-grid']['Subgrids Data'][1][0]))
        else:
            self.xlim = xlim

        if ylim == None:
            self.ylim = (
                    float(self.ini_content['[Grid]']['X1-grid']['Subgrids Data'][0][0]),
                    float(self.ini_content['[Grid]']['X1-grid']['Subgrids Data'][1][0]))
        else:
            self.ylim = ylim
        ### If global limit bool is set to False, the limits are not run
        if self.global_limits_bool == True:
            self.global_limits = self.get_limits()

    def read_init(self):
        """
        Method reads in data from the .ini file currently works for cylindrical polar coords
        """
        files = os.listdir(self.data_path)
        ini_file = [file for file in files if '.ini' in file]
        self.ini_content = {
            '[Grid]' : {},
            '[Chombo Refinement]' : {},
            '[Time]' : {},
            '[Solver]' : {},
            '[Boudary]' : {},
            '[Static Grid Output]' : {},
            '[Particles]' : {},
            '[Parameters]' : {},
        }

        if len(ini_file) == 0:
            print('*.ini file is not present in given directory, please specify location directory')
        elif len(ini_file) > 1:
            print('There are multiple ".ini" files contained in the working directory, please select the apropriate file!')
        else:
            ### Chose the right ini file
            if self.ini_path != None:
                print(f'.ini file is given as:\n{self.ini_path}')
                ini_data = list(open(self.ini_path))
            else:
                ini_data = list(open(self.data_path + ini_file[0], 'r'))
            ### tidy up the file read in
            filtr_data = [line for line in ini_data if line != '\n'] # removed empty lines
            filtr2_data = [line.replace('\n', '') for line in filtr_data]# remove unwanted new line operators
            split_data = [element.split(' ') for element in filtr2_data if '[' not in element] # this also removes the headers
            trim_data = [[element for element in line if element != ''] for line in split_data]
            ### Define grid parameters
            for line in trim_data:
                if '-grid' in line[0]:
                    sub_grid_num = {'Subgrids' : int(line[1])}
                    low_lim = {'Lower limit' : float(line[2])}
                    high_lim = {'Upper limit' : float(line[-1])}
                    # split subgrid into 3 element blocks while adding to dict
                    sub_grid_data = {'Subgrids Data' : np.reshape(line[2:-1], (sub_grid_num['Subgrids'], 3))}
                    new_dict = {line[0] : {}}
                    new_dict[line[0]].update(sub_grid_num)
                    new_dict[line[0]].update(low_lim)
                    new_dict[line[0]].update(high_lim)
                    new_dict[line[0]].update(sub_grid_data)
                    self.ini_content['[Grid]'].update(new_dict)

            ### Define axis grid size via subgrids 2nd element values
            grid_size = {}
            keys = self.ini_content['[Grid]'].keys()
            for grid in keys:
                x_grid_size = 0
                for subgrid in self.ini_content['[Grid]'][grid]['Subgrids Data']:
                    x_grid_size += int(subgrid[1])
                grid_size.update({grid : x_grid_size})
        
        self.grid_size = grid_size
        
        return grid_size

    def reader(self):
        path = os.getcwd()
        files  = os.listdir(self.data_path)
        extension = '.h5'
        self.data_list = [file for file in files if extension in file]
        return self.data_list
        
    def classifier(self, delta_time, output_selector=None):
        self.reader()
        data_file = self.data_list[delta_time]
        h5_read = h5py.File(self.data_path + data_file, 'r')

        self.timestep, self.cell_coord, self.node_coord = h5_read.keys()
        
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

        self.grid = h5_read[self.cell_coord]
            
        self.radial_grid = [r[0] for r in list(np.reshape(self.grid['X'], self.XZ_shape).T)]
        self.axial_grid = np.reshape(self.grid['Z'], self.XZ_shape).T[0]

        if output_selector == None:
            self.data = data[self.variables[self.selector]] # data[z][phi][rad]
            self.data_reshape = np.reshape(self.data, self.XZ_shape).T
            return [self.data_reshape, self.axial_grid, self.axial_grid]

        elif output_selector == 'all':
            return data

    def get_limits(self):
        """
        This method runs through all available data to set the colour limits up
        for each variable globally, across all time steps.
        """
        limits = {}
        ### Gets the last data file to find the data limits through the entire data range
        max_file = self.data_list[-1].replace('.dbl.h5', '').replace('data.', '')
        self.max_file = int(max_file)
        ### loops through everything to find limits for the data
        for step in range(-1, int(max_file)):
            # read files in here
            data = self.classifier(output_selector='all', delta_time=step)
            keys = list(data.keys())
            # set step to a valid value when its -1 for the classifier
            if step == -1:
                for key in keys:
                    limits[key] = {'min' : 0, 'max' : 0}
                    step = 0
            else:
                for index, variable in enumerate(keys):
                    var_min = np.min(list(data[variable]))
                    var_max = np.max(list(data[variable]))
                    
                    current_min = limits[variable]['min']
                    current_max = limits[variable]['max']

                    if var_min < current_min:
                        limits[variable].update({'min' : var_min})
                    if var_max > current_max:
                        limits[variable].update({'max' : var_max})
        
        self.limits = limits
        return limits

    def set_levels(self,variable):
        """
        This sets the global limits for the colourbars
        """
        if self.global_limits_bool == False:
            
            return 128

        levels = np.linspace(self.global_limits[variable]['min'],
                             self.global_limits[variable]['max'],
                             128)
        
        if len(levels[levels != 0]) == 0:
                levels = 128
        elif levels[0] < 0:
            min_abs = abs(levels[0])
            if min_abs < levels[-1]:
                levels = np.linspace(-abs(levels[-1]), levels[-1], 128)
            elif min_abs > levels[-1]:
                levels = np.linspace(-min_abs, min_abs, 128)
            else:
                print('Something went wrong, levels set to default 128')
                levels = 128

        return levels

    def plot_bfield_magnitude(self,close=False,save=False):
        """
        plots the magnitude of the magnetic fields, with no direction,
        calculated from the 3 components produced by PLUTO

        Returns the data set that is being plotted
        """
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        bx1 = np.reshape(data[self.b_radial], self.XZ_shape).T
        bx2 = np.reshape(data[self.b_azimuthal], self.XZ_shape).T
        bx3 = np.reshape(data[self.b_axial], self.XZ_shape).T
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
        axes.set_ylabel(r'Radial distance [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}v_quad'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}v_quad/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return data2plot

    def plot_glm(self,close=False,save=False):
        """
        Plots the General Lagrangian Multiplier which is an output due to the divB control
        
        Returns the data set that is being plotted
        """     
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        data2plot = np.reshape(data[self.psi_glm], self.XZ_shape).T

        levels = self.set_levels(self.psi_glm)
                
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        er = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=levels)
        
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
        axes.set_ylabel(r'Radial distance [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}glm'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}glm/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return data2plot

    def plot_bx1(self,close=False,save=False):
        """
        Plots the magnetic field in the first defined axis direction, x1 from the init file.
        
        Returns the data set that is being plotted
        """    
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        data2plot = np.reshape(data[self.b_radial], self.XZ_shape).T

        levels = self.set_levels(self.b_radial)
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=levels)
        
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
        axes.set_ylabel(r'Radial distance [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}bx1'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}bx1/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return data2plot

    def plot_bx2(self,close=False,save=False):
        """
        Plots the magnetic field in the second defined axis direction, x2 from the init file.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        data2plot = np.reshape(data[self.b_azimuthal], self.XZ_shape).T

        levels = levels = self.set_levels(self.b_azimuthal)
        
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
        axes.set_ylabel(r'Radial distance [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}bx2'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}bx2/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return data2plot

    def plot_bx3(self,close=False,save=False):
        """
        Plots the magnetic field in the third defined axis direction, x3 from the init file.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        data2plot = np.reshape(data[self.b_axial], self.XZ_shape).T

        levels = levels = self.set_levels(self.b_axial)
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=levels)
        
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
        axes.set_ylabel(r'Radial distance [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}bx3'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}bx3/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return data2plot

    def plot_pressure(self,close=False,save=False):
        """
        Plots the pressure field.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        data2plot = np.reshape(data[self.pressure], self.XZ_shape).T

        levels = levels = self.set_levels(self.pressure)
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=levels)
        
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
        axes.set_ylabel(r'Radial distance [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}pressure'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}pressure/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return data2plot

    def plot_log_pressure(self,close=False,save=False):
        """
        Plots the log pressure field.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        data2plot = np.reshape(data[self.pressure], self.XZ_shape).T

        p_levels = self.set_levels(self.pressure)
        levels = np.linspace(-np.max(p_levels), np.max(p_levels), 128)
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, np.log(data2plot), cmap='hot', levels=128)
        
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
        axes.set_ylabel(r'Radial distance [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}log_pressure'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}log_presure/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return np.log(data2plot)

    def plot_density(self,close=False,save=False):
        """
        Plots the mass density field.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        data2plot = np.reshape(data[self.density], self.XZ_shape).T

        levels = self.set_levels(self.density)
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='hot', levels=levels)
        
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
        axes.set_ylabel(r'Radial distance [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}density'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}density/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return data2plot

    def plot_tracer(self,close=False,save=False):
        """
        Plots the tracer.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        data2plot = np.reshape(data[self.tracer1], self.XZ_shape).T

        levels = self.set_levels(self.tracer1)
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=levels)
        
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
        axes.set_ylabel(r'Radial distance [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}tracer'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}tracer/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return data2plot

    def plot_vx1(self,close=False,save=False):
        """
        Plots the velocity field in x1 direction.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        data2plot = np.reshape(data[self.radial_velocity], self.XZ_shape).T

        levels = self.set_levels(self.radial_velocity)
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=levels)
        
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
        axes.set_ylabel(r'Radial distance [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}vx1'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}vx1/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return data2plot

    def plot_vx2(self,close=False,save=False):
        """
        Plots the velocity field in x2 direction.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        data2plot = np.reshape(data[self.azimuthal_velocity], self.XZ_shape).T

        levels = self.set_levels(self.azimuthal_velocity)
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=levels)
        
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
        axes.set_ylabel(r'Radial distance [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}vx2'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}vx2/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return data2plot

    def plot_vx3(self,close=False,save=False):
        """
        Plots the velocity field in x3 direction.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        data2plot = np.reshape(data[self.axial_velocity], self.XZ_shape).T

        levels = self.set_levels(self.axial_velocity)

        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap=self.cmap, levels=levels)
        
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
        axes.set_ylabel(r'Radial distance [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}vx3'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}vx3/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return data2plot

    def plot_velocity_field_magnitude(self,close=False,save=False):
        """
        Plots the magnitude of the velocity fields, with no direction,
        calculated from the 3 components produced by PLUTO

        Returns the data set that is being plotted
        """
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        vx1 = np.reshape(data[self.radial_velocity], self.XZ_shape).T
        vx2 = np.reshape(data[self.azimuthal_velocity], self.XZ_shape).T
        vx3 = np.reshape(data[self.axial_velocity], self.XZ_shape).T
        data2plot = np.sqrt(np.asarray(vx1)**2 + np.asarray(vx2)**2 + np.asarray(vx3**2))

        levels1 = self.set_levels(self.radial_velocity)
        levels2 = self.set_levels(self.azimuthal_velocity)
        levels3 = self.set_levels(self.axial_velocity)
        
        collection_array = get_magnitude([
                                                np.asarray([np.min(levels1),np.max(levels1)]),
                                                np.asarray([np.min(levels2),np.max(levels2)]),
                                                np.asarray([np.min(levels3),np.max(levels3)])
                                              ]
                                              )
    
        levels = np.linspace(-np.max(collection_array), np.max(collection_array), 128)
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='hot', levels=levels)
        
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
        axes.set_ylabel(r'Radial distance [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}v_mag'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}v_mag/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return data2plot

    def plot_pressure_density(self,close=False,save=False):
        """
        Plots a pressure and density plot on one figure

        Returns None
        """
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        density_data = np.reshape(data[self.density], self.XZ_shape).T
        pressure_data = np.reshape(data[self.pressure], self.XZ_shape).T
        
        figure, axes = plt.subplots(2,1,figsize=self.image_size, dpi=self.dpi)
        self.cmap = 'hot'
        density_levels = levels = self.set_levels(self.density)
        
        axes[0].contourf(self.axial_grid, self.radial_grid, density_data, cmap=self.cmap, levels=density_levels)
        im1 = axes[0].imshow(density_data, cmap=self.cmap)
        divider = mal(axes[0])
        cax = divider.append_axes('right',size='5%',pad=0.05)
        plt.colorbar(im1,cax,ticks=np.linspace(0,np.max(density_data), 6))
        axes[0].set_title(f'Mass Density at {self.timestep.replace("_", " ")}')
        axes[0].set_xlim(0, self.xlim[1])
        axes[0].set_ylim(0, self.ylim[1])
        axes[0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0].set_xlabel(r'Axial distance [$R_{jet}$]')
        
        pressure_levels = self.set_levels(self.pressure)
        axes[1].contourf(self.axial_grid, self.radial_grid, np.log(pressure_data), cmap=self.cmap, levels=128)
        im1 = axes[1].imshow(pressure_data, cmap=self.cmap)
        divider = mal(axes[1])
        cax = divider.append_axes('right',size='5%',pad=0.05)
        plt.colorbar(im1,cax,ticks=np.linspace(0,np.max(pressure_data), 6))
        axes[1].set_title(f'Log of Pressure at {self.timestep.replace("_", " ")}')
        axes[1].set_xlim(0, self.xlim[1])
        axes[1].set_ylim(0, self.ylim[1])
        axes[1].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[1].set_xlabel(r'Axial distance [$R_{jet}$]')

        plt.subplots_adjust(left=0.125,
                            right=0.9,
                            bottom=-0.05,
                            top=0.9,
                            wspace=0.2,
                            hspace=0.35)

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}pressure_density'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}pressure_density/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return None
    
    def magneticfield_quad(self,close=False,save=False):
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
        if min_val == 0:
            levels = np.linspace(0,np.max(data4),128)
        else:
            levels = np.concatenate([negative_levels, positive_levels[1:]])
                
        figure, axes = plt.subplots(2,2,figsize=(self.image_size[0]*1.2, self.image_size[1]*1.8), dpi=self.dpi)
        cax = plt.axes([1, 0.05, 0.05, 0.9], label='magnetic field')
        # x1 direction 
        axes[0][0].contourf(self.axial_grid, self.radial_grid, data1, cmap=self.cmap, levels=levels)
        axes[0][0].set_title(f'Magnetic field in x1 direction at {self.timestep.replace("_", " ")}')
        axes[0][0].set_xlim(0, self.xlim[1])
        axes[0][0].set_ylim(0, self.ylim[1])
        axes[0][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][0].set_xlabel(r'Axial distance [$R_{jet}$]')
        # x2 direction 
        axes[0][1].contourf(self.axial_grid, self.radial_grid, data2, cmap=self.cmap, levels=levels)
        axes[0][1].set_title(f'Magnetic field in x2 direction at {self.timestep.replace("_", " ")}')
        axes[0][1].set_xlim(0, self.xlim[1])
        axes[0][1].set_ylim(0, self.ylim[1])
        axes[0][1].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][1].set_xlabel(r'Axial distance [$R_{jet}$]')
        # x3 direction 
        axes[1][0].contourf(self.axial_grid, self.radial_grid, data3, cmap=self.cmap, levels=levels)
        axes[1][0].set_title(f'Magnetic field in x3 direction at {self.timestep.replace("_", " ")}')
        axes[1][0].set_xlim(0, self.xlim[1])
        axes[1][0].set_ylim(0, self.ylim[1])
        axes[1][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[1][0].set_xlabel(r'Axial distance [$R_{jet}$]')
        # magnitude
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

        if save==True:
            check_dir = f'{self.data_path}b_quad'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}b_quad/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
        return figure

    def velocity_quad(self,close=False,save=False):
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
        
        if np.max(data4) == 0:
            levels = None
        elif min_val == 0:
            levels = np.linspace(0,np.max(data4),128)
        else:
            levels = np.concatenate([negative_levels, positive_levels[1:]])
                
        figure, axes = plt.subplots(2,2,figsize=(self.image_size[0]*1.2, self.image_size[1]*1.8), dpi=self.dpi)
        cax = plt.axes([1, 0.05, 0.05, 0.9], label='Velocity field')
        
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
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=self.cmap), cax=cax, format='%.2f', label='Velocity Field')
        
        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}v_quad'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}v_quad/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()
        return figure

    def alfven_velocity(self,close=False,save=False):
        """
        Method calculates the alfven velocity components and plots them side by side.
        """

        bx1 = self.plot_bx1(close=True)
        bx2 = self.plot_bx2(close=True)
        bx3 = self.plot_bx3(close=True)
        bxm = self.plot_bfield_magnitude(close=True)

        density = self.plot_density(close=True)

        data1 = alfven_velocity(bx1, density)
        data2 = alfven_velocity(bx2, density)
        data3 = alfven_velocity(bx3, density)
        data4 = alfven_velocity(bxm, density)

        min_val = 0
        for x in [data1,data2,data3,data4]:
            if np.min(x) < min_val:
                min_val = np.min(x)
        
        negative_levels = np.linspace(-np.max(data4),0,64)
        positive_levels = np.linspace(0,np.max(data4),65)
        
        if np.max(data4) == 0:
            levels = None
        elif min_val == 0:
            levels = np.linspace(0,np.max(data4),128)
        else:
            levels = np.concatenate([negative_levels, positive_levels[1:]])
        
        figure, axes = plt.subplots(2,2,figsize=(self.image_size[0]*1.2, self.image_size[1]*1.8), dpi=self.dpi)
        cax = plt.axes([1, 0.05, 0.05, 0.9], label='Alfvén Velocity field')
        
        axes[0][0].contourf(self.axial_grid, self.radial_grid, data1, cmap=self.cmap, levels=levels)
        axes[0][0].set_title(f'Alfvén Velocity field in x1 direction at {self.timestep.replace("_", " ")}')
        axes[0][0].set_xlim(0, self.xlim[1])
        axes[0][0].set_ylim(0, self.ylim[1])
        axes[0][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][0].set_xlabel(r'Axial distance [$R_{jet}$]')
       
        axes[0][1].contourf(self.axial_grid, self.radial_grid, data2, cmap=self.cmap, levels=levels)
        axes[0][1].set_title(f'Alfvén Velocity field in x2 direction at {self.timestep.replace("_", " ")}')
        axes[0][1].set_xlim(0, self.xlim[1])
        axes[0][1].set_ylim(0, self.ylim[1])
        axes[0][1].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][1].set_xlabel(r'Axial distance [$R_{jet}$]')

        axes[1][0].contourf(self.axial_grid, self.radial_grid, data3, cmap=self.cmap, levels=levels)
        axes[1][0].set_title(f'Alfvén Velocity field in x3 direction at {self.timestep.replace("_", " ")}')
        axes[1][0].set_xlim(0, self.xlim[1])
        axes[1][0].set_ylim(0, self.ylim[1])
        axes[1][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[1][0].set_xlabel(r'Axial distance [$R_{jet}$]')
        
        axes[1][1].contourf(self.axial_grid, self.radial_grid, data4, cmap=self.cmap, levels=levels)
        axes[1][1].set_title(f'Alfvén Velocity field magnitude at {self.timestep.replace("_", " ")}')
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

        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=self.cmap), cax=cax, format='%.2f', label='Alfvén Velocity Field')
        


        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}av_quad'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}av_quad/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return figure

    def magnetic_streamlines(self,close=False,save=False):
        """
        Plots a vector plot of the magnetic field lines.
        """
        self.classifier(output_selector='all', delta_time=self.time_step)
        data1 = self.plot_bx1(close=True)
        data2 = self.plot_bx3(close=True)
        cmp = 'Wistia'
        cmp='viridis'

        subgrid_x_low = float(self.ini_content['[Grid]']['X3-grid']['Subgrids Data'][0][0])
        subgrid_y_low = float(self.ini_content['[Grid]']['X1-grid']['Subgrids Data'][0][0])
        subgrid_x = float(self.ini_content['[Grid]']['X3-grid']['Subgrids Data'][1][0])
        subgrid_y = float(self.ini_content['[Grid]']['X1-grid']['Subgrids Data'][1][0])
        subgrid_x_res = int(self.ini_content['[Grid]']['X3-grid']['Subgrids Data'][0][1])
        subgrid_y_res = int(self.ini_content['[Grid]']['X1-grid']['Subgrids Data'][0][1])
        
        X, Y = np.meshgrid(np.linspace(subgrid_x_low,subgrid_x,subgrid_x_res),
                            np.linspace(subgrid_y_low,subgrid_y,subgrid_y_res))
        test = np.ones(np.shape(X))

        figure, axes = plt.subplots(1,1,figsize=(10,5),dpi=self.dpi)

        y_data = np.array([row[:subgrid_x_res] for row in data1][:subgrid_y_res])
        x_data = np.array([row[:subgrid_x_res] for row in data2][:subgrid_y_res])

        mag_dat = np.sqrt(x_data**2 + y_data**2)

        

        axes.contourf(
                    np.linspace(subgrid_x_low, subgrid_x, subgrid_x_res), 
                    np.linspace(subgrid_y_low, subgrid_y, subgrid_y_res),
                    mag_dat,
                    cmap=cmp)

        axes.streamplot(X,
                        Y,
                        x_data,
                        y_data,
                        density=2.5,
                        color='black',
                        #cmap='Greys',
                        integration_direction='both',
                        maxlength=20,
                        arrowsize=0.95,
                        arrowstyle='->',
                        linewidth=0.5

                        )

        norm = matplotlib.colors.Normalize(vmin=np.min(mag_dat), vmax=np.max(mag_dat))
        #cax = plt.axes([0.95, 0.1, 0.25, 1], label='Magnetic field')
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmp), ax=axes, format='%.2f', label='Magnetic Field')
        plt.title('Magnetic field with field line direction')
        plt.ylim(self.ylim)
        plt.xlim(self.xlim)


        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}field_line'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}field_line/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()


        return figure

    def histogram(self, data, title, bins=None, log=False):
        """
        Method to plot histogram of the data which is passed in as the argument
        """
        shape = data.shape
        x_shape = shape[0]*shape[1]
        new_shape = (x_shape, 1)
        data2plot = np.reshape(data, new_shape)
        plt.rc('font', size=8)
        fig, axes = plt.subplots(1,1,figsize=self.image_size, dpi=self.dpi)
        axes.set_title(f'Histogram data for {title} value (normalised) at {self.time_step}')
        hist = axes.hist(data2plot, bins=bins, align='mid', edgecolor='white')
        axes.set_xlabel('Value')
        axes.set_ylabel('Frequency')
        cols = axes.patches
        labels = [f'{int(x)/x_shape*100:.2f}%' for x in hist[0]]
        for col, label in zip(cols, labels):
            height = col.get_height()
            axes.text(col.get_x() + col.get_width() / 2, height+0.01, label, ha='center', va='bottom')
    
        if log==True:
            plt.semilogy()
    
        plt.show()
        return hist

    def plot_spacetime(self, variable, save=False, close=False):
        """
        Method plots a space time diagram of the variables along the jet axis

            Accepted Variables:
                'bx1' -> Magnetic Field in x1 direction,
                'bx2' -> Magnetic Field in x2 direction,
                'bx3' -> Magnetic Field in x3 direction,
                'vx1' -> Velocity Field in x1 direction,
                'vx2' -> Velocity Field in x2 direction,
                'vx3' -> Velocity Field in x3 direction,
                'prs' -> Pressure Field,
                'rho' -> Density Field,
                'glm' -> General Lagrangian Multiplier,
                'av1' -> Alfvén Velocity Field in x1 direction,
                'av2' -> Alfvén Velocity Field in x2 direction,
                'av3' -> Alfvén Velocity Field in x3 direction]
        """
        names = ['Bx1', 'Bx2', 'Bx3', 'vx1', 'vx2', 'vx3', 'prs', 'rho', 'psi_glm', 'av1', 'av2', 'av3']
        if variable not in names:
            raise ValueError("Please give a valid variable to plot:\n'bx1', 'bx2', 'bx3', 'vx1', 'vx2', 'vx3', 'prs', 'rho', 'glm', 'av1', 'av2', 'av3'")
        
        if variable == 'Bx1':
            axial_array = []
            time_list = range(len(self.data_list))
            for time, data_file in enumerate(self.data_list):
                d_file = h5py.File(self.data_path + data_file, 'r')
                keys = list(d_file.keys())
                data = d_file[keys[0]]['vars']
                transposed = np.reshape(data[variable], self.XZ_shape).T
                data2plot = transposed[0]
                axial_array.append(data2plot)
            
            X, Y = np.meshgrid(self.axial_grid, time_list)
            figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Magnetic Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Magnetic field in x1 direction along Jet axis')
            axes.set_ylabel(r'Time [$time-step$]')
            axes.set_xlabel(r'Axial distance [$R_{jet}$]')

            if close==True:
                plt.close()

            if save==True:
                check_dir = f'{self.data_path}spacetime/bx1'
                if os.path.exists(check_dir) is False:
                    os.mkdir(check_dir)
                bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
                plt.savefig(f'{self.data_path}spacetime/bx1/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
                plt.close()
        
        elif variable == 'Bx2':
            axial_array = []
            time_list = range(len(self.data_list))
            for time, data_file in enumerate(self.data_list):
                d_file = h5py.File(self.data_path + data_file, 'r')
                keys = list(d_file.keys())
                data = d_file[keys[0]]['vars']
                transposed = np.reshape(data[variable], self.XZ_shape).T
                data2plot = transposed[0]
                axial_array.append(data2plot)
            
            X, Y = np.meshgrid(self.axial_grid, time_list)
            figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Magnetic Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Magnetic field in x2 direction along Jet axis')
            axes.set_ylabel(r'Time [$time step$]')
            axes.set_xlabel(r'Axial distance [$R_{jet}$]')

            if close==True:
                plt.close()

            if save==True:
                check_dir = f'{self.data_path}spacetime/bx2'
                if os.path.exists(check_dir) is False:
                    os.mkdir(check_dir)
                bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
                plt.savefig(f'{self.data_path}spacetime/bx2/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
                plt.close()

        elif variable == 'Bx3':
            axial_array = []
            time_list = range(len(self.data_list))
            for time, data_file in enumerate(self.data_list):
                d_file = h5py.File(self.data_path + data_file, 'r')
                keys = list(d_file.keys())
                data = d_file[keys[0]]['vars']
                transposed = np.reshape(data[variable], self.XZ_shape).T
                data2plot = transposed[0]
                axial_array.append(data2plot)
            
            X, Y = np.meshgrid(self.axial_grid, time_list)
            figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Magnetic Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Magnetic field in x3 direction along Jet axis')
            axes.set_ylabel(r'Time [$time step$]')
            axes.set_xlabel(r'Axial distance [$R_{jet}$]')

            if close==True:
                plt.close()

            if save==True:
                check_dir = f'{self.data_path}spacetime/bx3'
                if os.path.exists(check_dir) is False:
                    os.mkdir(check_dir)
                bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
                plt.savefig(f'{self.data_path}spacetime/bx3/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
                plt.close()

        elif variable == 'psi_glm':
            axial_array = []
            time_list = range(len(self.data_list))
            for time, data_file in enumerate(self.data_list):
                d_file = h5py.File(self.data_path + data_file, 'r')
                keys = list(d_file.keys())
                data = d_file[keys[0]]['vars']
                transposed = np.reshape(data[variable], self.XZ_shape).T
                data2plot = transposed[0]
                axial_array.append(data2plot)
            
            X, Y = np.meshgrid(self.axial_grid, time_list)
            figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Magnetic Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the GLM along Jet axis')
            axes.set_ylabel(r'Time [$time step$]')
            axes.set_xlabel(r'Axial distance [$R_{jet}$]')

            if close==True:
                plt.close()

            if save==True:
                check_dir = f'{self.data_path}spacetime/psi_glm'
                if os.path.exists(check_dir) is False:
                    os.mkdir(check_dir)
                bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
                plt.savefig(f'{self.data_path}spacetime/psi_glm/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
                plt.close()

        elif variable == 'prs':
            axial_array = []
            time_list = range(len(self.data_list))
            for time, data_file in enumerate(self.data_list):
                d_file = h5py.File(self.data_path + data_file, 'r')
                keys = list(d_file.keys())
                data = d_file[keys[0]]['vars']
                transposed = np.reshape(data[variable], self.XZ_shape).T
                data2plot = transposed[0]
                axial_array.append(data2plot)
            
            X, Y = np.meshgrid(self.axial_grid, time_list)
            figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
            pl = axes.contourf(X, Y, np.log(axial_array), cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Magnetic Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the log(Pressure) along Jet axis')
            axes.set_ylabel(r'Time [$time step$]')
            axes.set_xlabel(r'Axial distance [$R_{jet}$]')

            if close==True:
                plt.close()

            if save==True:
                check_dir = f'{self.data_path}spacetime/prs'
                if os.path.exists(check_dir) is False:
                    os.mkdir(check_dir)
                bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
                plt.savefig(f'{self.data_path}spacetime/prs/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
                plt.close()

        elif variable == 'rho':
            axial_array = []
            time_list = range(len(self.data_list))
            for time, data_file in enumerate(self.data_list):
                d_file = h5py.File(self.data_path + data_file, 'r')
                keys = list(d_file.keys())
                data = d_file[keys[0]]['vars']
                transposed = np.reshape(data[variable], self.XZ_shape).T
                data2plot = transposed[0]
                axial_array.append(data2plot)
            
            X, Y = np.meshgrid(self.axial_grid, time_list)
            figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Magnetic Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Density along Jet axis')
            axes.set_ylabel(r'Time [$time step$]')
            axes.set_xlabel(r'Axial distance [$R_{jet}$]')

            if close==True:
                plt.close()

            if save==True:
                check_dir = f'{self.data_path}spacetime/rho'
                if os.path.exists(check_dir) is False:
                    os.mkdir(check_dir)
                bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
                plt.savefig(f'{self.data_path}spacetime/rho/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
                plt.close()

        elif variable == 'vx1':
            axial_array = []
            time_list = range(len(self.data_list))
            for time, data_file in enumerate(self.data_list):
                d_file = h5py.File(self.data_path + data_file, 'r')
                keys = list(d_file.keys())
                data = d_file[keys[0]]['vars']
                transposed = np.reshape(data[variable], self.XZ_shape).T
                data2plot = transposed[0]
                axial_array.append(data2plot)
            
            X, Y = np.meshgrid(self.axial_grid, time_list)
            figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Velocity Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Velocity in x1 direction along Jet axis')
            axes.set_ylabel(r'Time [$time-step$]')
            axes.set_xlabel(r'Axial distance [$R_{jet}$]')

            if close==True:
                plt.close()

            if save==True:
                check_dir = f'{self.data_path}spacetime/vx1'
                if os.path.exists(check_dir) is False:
                    os.mkdir(check_dir)
                bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
                plt.savefig(f'{self.data_path}spacetime/vx1/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
                plt.close()
        
        elif variable == 'vx2':
            axial_array = []
            time_list = range(len(self.data_list))
            for time, data_file in enumerate(self.data_list):
                d_file = h5py.File(self.data_path + data_file, 'r')
                keys = list(d_file.keys())
                data = d_file[keys[0]]['vars']
                transposed = np.reshape(data[variable], self.XZ_shape).T
                data2plot = transposed[0]
                axial_array.append(data2plot)
            
            X, Y = np.meshgrid(self.axial_grid, time_list)
            figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Velocity Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Velocity in x2 direction along Jet axis')
            axes.set_ylabel(r'Time [$time step$]')
            axes.set_xlabel(r'Axial distance [$R_{jet}$]')

            if close==True:
                plt.close()

            if save==True:
                check_dir = f'{self.data_path}spacetime/vx2'
                if os.path.exists(check_dir) is False:
                    os.mkdir(check_dir)
                bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
                plt.savefig(f'{self.data_path}spacetime/vx2/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
                plt.close()

        elif variable == 'vx3':
            axial_array = []
            time_list = range(len(self.data_list))
            for time, data_file in enumerate(self.data_list):
                d_file = h5py.File(self.data_path + data_file, 'r')
                keys = list(d_file.keys())
                data = d_file[keys[0]]['vars']
                transposed = np.reshape(data[variable], self.XZ_shape).T
                data2plot = transposed[0]
                axial_array.append(data2plot)
            
            X, Y = np.meshgrid(self.axial_grid, time_list)
            figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Velocity Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Velocity in x3 direction along Jet axis')
            axes.set_ylabel(r'Time [$time step$]')
            axes.set_xlabel(r'Axial distance [$R_{jet}$]')

            if close==True:
                plt.close()

            if save==True:
                check_dir = f'{self.data_path}spacetime/vx3'
                if os.path.exists(check_dir) is False:
                    os.mkdir(check_dir)
                bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
                plt.savefig(f'{self.data_path}spacetime/vx3/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
                plt.close()
        
        elif variable == 'av1':
            magfield = []
            density = []
            time_list = range(len(self.data_list))
            for time, data_file in enumerate(self.data_list):
                d_file = h5py.File(self.data_path + data_file, 'r')
                keys = list(d_file.keys())
                data = d_file[keys[0]]['vars']
                transposedB = np.reshape(data['Bx1'], self.XZ_shape).T
                transposedD = np.reshape(data['rho'], self.XZ_shape).T
                b2plot = transposedB[0]
                rho2plot = transposedD[0]
                magfield.append(b2plot)
                density.append(rho2plot)

            
            X, Y = np.meshgrid(self.axial_grid, time_list)
            figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
            pl = axes.contourf(X, Y, alfven_velocity(B=magfield, density=density), cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Alfvén Velocity magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Alfvén Velocity in x1 direction along Jet axis')
            axes.set_ylabel(r'Time [$time step$]')
            axes.set_xlabel(r'Axial distance [$R_{jet}$]')

            if close==True:
                plt.close()

            if save==True:
                check_dir = f'{self.data_path}spacetime/av2'
                if os.path.exists(check_dir) is False:
                    os.mkdir(check_dir)
                bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
                plt.savefig(f'{self.data_path}spacetime/av2/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
                plt.close()

        elif variable == 'av2':
            magfield = []
            density = []
            time_list = range(len(self.data_list))
            for time, data_file in enumerate(self.data_list):
                d_file = h5py.File(self.data_path + data_file, 'r')
                keys = list(d_file.keys())
                data = d_file[keys[0]]['vars']
                transposedB = np.reshape(data['Bx2'], self.XZ_shape).T
                transposedD = np.reshape(data['rho'], self.XZ_shape).T
                b2plot = transposedB[0]
                rho2plot = transposedD[0]
                magfield.append(b2plot)
                density.append(rho2plot)

            
            X, Y = np.meshgrid(self.axial_grid, time_list)
            figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
            pl = axes.contourf(X, Y, alfven_velocity(B=magfield, density=density), cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Alfvén Velocity magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Alfvén Velocity in x2 direction along Jet axis')
            axes.set_ylabel(r'Time [$time step$]')
            axes.set_xlabel(r'Axial distance [$R_{jet}$]')

            if close==True:
                plt.close()

            if save==True:
                check_dir = f'{self.data_path}spacetime/av3'
                if os.path.exists(check_dir) is False:
                    os.mkdir(check_dir)
                bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
                plt.savefig(f'{self.data_path}spacetime/av3/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
                plt.close()

        elif variable == 'av3':
            magfield = []
            density = []
            time_list = range(len(self.data_list))
            for time, data_file in enumerate(self.data_list):
                d_file = h5py.File(self.data_path + data_file, 'r')
                keys = list(d_file.keys())
                data = d_file[keys[0]]['vars']
                transposedB = np.reshape(data['Bx3'], self.XZ_shape).T
                transposedD = np.reshape(data['rho'], self.XZ_shape).T
                b2plot = transposedB[0]
                rho2plot = transposedD[0]
                magfield.append(b2plot)
                density.append(rho2plot)

            
            X, Y = np.meshgrid(self.axial_grid, time_list)
            figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
            pl = axes.contourf(X, Y, alfven_velocity(B=magfield, density=density), cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Alfvén Velocity magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Alfvén Velocity in x3 direction along Jet axis')
            axes.set_ylabel(r'Time [$time step$]')
            axes.set_xlabel(r'Axial distance [$R_{jet}$]')

            if close==True:
                plt.close()

            if save==True:
                check_dir = f'{self.data_path}spacetime/av3'
                if os.path.exists(check_dir) is False:
                    os.mkdir(check_dir)
                bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
                plt.savefig(f'{self.data_path}spacetime/av3/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
                plt.close()

        return
