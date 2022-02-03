#%%
from posixpath import split
from turtle import color
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
from pluto_python.calculator import get_magnitude, alfven_velocity
from pluto_python.calculator import magneto_acoustic_velocity, mach_number

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
        mirrored = False
        
    ):

        self.data_path = data_path
        self.dpi = dpi
        self.image_size = image_size
        self.selector = select_variable
        self.time_step = time_step
        self.global_limits_bool = global_limits
        self.cmap = cmap
        self.ini_path = ini_path
        self.mirrored = mirrored
        self.closure = False
        self.simulation_title = self.data_path.split('/')[-2]

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
        self.shape_limits = self.read_ini()
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

    def read_ini(self):
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
            '[Boundary]' : {},
            '[Static Grid Output]' : {},
            '[Particles]' : {},
            '[Parameters]' : {},
            '[Chombo HDF5 output]' : {},
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
            split_data = [element.split(' ') for element in filtr2_data]
            
            no_whitespace_elements = [[y for y in x if y != ''] for x in split_data if x != '']
            with_headers_tidy = []
            for line in no_whitespace_elements:
                if '[' in line[0]:
                    with_headers_tidy.append(' '.join(line))
                else:
                    with_headers_tidy.append(line)

            block = None

            for line in with_headers_tidy:
                if type(line) is not list:
                    block = line
                
                elif block == '[Grid]':
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

                else:
                    self.ini_content[block].update({line[0] : line[1:]})



            ### Define axis grid size via subgrids 2nd element values
            grid_size = {}
            keys = self.ini_content['[Grid]'].keys()
            for grid in keys:
                x_grid_size = 0
                for subgrid in self.ini_content['[Grid]'][grid]['Subgrids Data']:
                    x_grid_size += int(subgrid[1])
                grid_size.update({grid : x_grid_size})
        
        self.tstop = float(self.ini_content['[Time]']['tstop'][0])
        self.grid_size = grid_size
        return grid_size

    def reader(self):
        path = os.getcwd()
        files  = os.listdir(self.data_path)
        extension = '.h5'
        self.data_list = [file for file in files if extension in file and 'out' not in file]
        return self.data_list
        
    def classifier(self, delta_time, output_selector=None):
        """
        delta_time: is the time step of which is number of the file by pluto
        output_selector: None or 'all' <- will be removed soon
        """
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
        self.initial_radial_grid = self.radial_grid

        if output_selector == None:
            self.data = data[self.variables[self.selector]] # data[z][phi][rad]
            self.data_reshape = np.reshape(self.data, self.XZ_shape).T
            return [self.data_reshape, self.axial_grid, self.axial_grid]

        elif output_selector == 'all':
            return data

    def _flip_multiply(self, array):
        """
        This method makes a single quadrant mirrored along the axial direction
        """
        x = self.axial_grid
        y = self.radial_grid
        xi = np.flip(x, 0)
        yi = np.flip(y, 0)

        if np.shape(self.initial_radial_grid) == np.shape(self.radial_grid):
            self.radial_grid = np.concatenate((-yi[:-1], y), axis=0)
        array_inverse = np.flip(array, 0)
        new_array = np.concatenate((array_inverse[:-1], array),axis=0)
        self.ylim = (-self.ylim[1], self.ylim[1])
        return new_array


    def get_limits(self,log=False):
        """
        This method runs through all available data to set the colour limits up
        for each variable globally, across all time steps.
        """
        limits = {}
        ### Gets the last data file to find the data limits through the entire data range
        max_file = self.data_list[-1].replace('.dbl.h5', '').replace('data.', '')
        self.max_file = int(max_file)
        ### loops through everything to find limits for the data
        for step in range(-1, self.time_step):
            # read files in here
            print(f'Reading file number {step}')
            data = self.classifier(output_selector='all', delta_time=step)
            keys = list(data.keys())
            # set step to a valid value when its -1 for the classifier
            if step == -1:
                for key in keys:
                    limits[key] = {'min' : 0, 'max' : 0}
                    step = 0
            else:
                for index, variable in enumerate(keys):
                    if log == True:
                        var_min = np.min(np.log(list(data[variable])))
                        var_max = np.max(np.log(list(data[variable])))
                    elif log == False:
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
        if self.mirrored == True:
            data2plot = self._flip_multiply(data2plot)
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

        axes.set_title(f'Magnetic field magnitude at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
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
        if self.mirrored == True:
            data2plot = self._flip_multiply(data2plot)
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
        
        axes.set_title(f'General Lagrangian Multiplier at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
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
        if self.mirrored == True:
            data2plot = self._flip_multiply(data2plot)

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
        
        axes.set_title(f'Magnetic field in x1 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
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
        if self.mirrored == True:
            data2plot = self._flip_multiply(data2plot)
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
        
        axes.set_title(f'Magnetic field in x2 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
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
        if self.mirrored == True:
            data2plot = self._flip_multiply(data2plot)
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
        
        axes.set_title(f'Magnetic field in x3 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
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
        if self.mirrored == True:
            data2plot = self._flip_multiply(data2plot)
        levels = levels = self.set_levels(self.pressure)
        
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        pl = axes.contourf(self.axial_grid, self.radial_grid, data2plot, cmap='hot', levels=levels)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Pressure', 
            format='%.2f'
            )
        
        axes.set_title(f'Pressure at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
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
        data2plot  = np.reshape(data[self.pressure], self.XZ_shape).T
        if self.mirrored == True:
            data2plot = self._flip_multiply(data2plot)
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
        
        axes.set_title(f'Log Pressure at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
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
        if self.mirrored == True:
            data2plot = self._flip_multiply(data2plot)
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
        
        axes.set_title(f'Mass Density at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
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

    def plot_log_density(self,close=False,save=False):
        """
        Plots the mass density field.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        data2plot = np.log(np.reshape(data[self.density], self.XZ_shape).T)
        if self.mirrored == True:
            data2plot = self._flip_multiply(data2plot)
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
        
        axes.set_title(f'Log(Density) at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
        axes.set_ylabel(r'Radial distance [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}logdensity'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}logdensity/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        return data2plot

    def plot_tracer(self,close=False,save=False):
        """
        Plots the tracer.
        
        Returns the data set that is being plotted
        """   
        data = self.classifier(output_selector='all', delta_time=self.time_step)
        data2plot = np.reshape(data[self.tracer1], self.XZ_shape).T
        if self.mirrored == True:
            data2plot = self._flip_multiply(data2plot)
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
        
        axes.set_title(f'Tracer at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
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
        if self.mirrored == True:
            data2plot = self._flip_multiply(data2plot)
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
        
        axes.set_title(f'Velocity field in x1 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
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
        if self.mirrored == True:
            data2plot = self._flip_multiply(data2plot)
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
        
        axes.set_title(f'Velocity field in x2 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
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
        if self.mirrored == True:
            data2plot = self._flip_multiply(data2plot)
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
        
        axes.set_title(f'Velocity field in x3 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
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
        #if self.mirrored == True:
        #    data2plot = self._flip_multiply(data2plot)
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

        axes.set_title(f'Velocity field magnitude at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
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
        density_data = self.plot_density(close=True)
        pressure_data = self.plot_pressure(close=True)

        figure, axes = plt.subplots(2,1,figsize=self.image_size, dpi=self.dpi)
        self.cmap = 'hot'
        density_levels = self.set_levels(self.density)
        
        axes[0].contourf(self.axial_grid, self.radial_grid, np.log(density_data), cmap=self.cmap, levels=density_levels)
        im1 = axes[0].imshow(np.log(density_data), cmap=self.cmap)

        divider = mal(axes[0])
        cax = divider.append_axes('right',size='5%',pad=0.05)
        plt.colorbar(im1,cax,ticks=np.linspace(np.min(np.log(density_data)),np.max(np.log(density_data)), 6))
        axes[0].set_title(f'Log of Mass Density at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[0].set_xlim(self.xlim[0], self.xlim[1])
        axes[0].set_ylim(self.ylim[0], self.ylim[1])
        axes[0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0].set_xlabel(r'Axial distance [$R_{jet}$]')
        
        pressure_levels = self.set_levels(self.pressure)
        axes[1].contourf(self.axial_grid, self.radial_grid, np.log(pressure_data), cmap=self.cmap, levels=128)
        im1 = axes[1].imshow(np.log(pressure_data), cmap=self.cmap)
        divider = mal(axes[1])
        cax = divider.append_axes('right',size='5%',pad=0.05)
        plt.colorbar(im1,cax,ticks=np.linspace(np.min(np.log(pressure_data)),np.max(np.log(pressure_data)), 6))
        axes[1].set_title(f'Log of Pressure at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[1].set_xlim(self.xlim[0], self.xlim[1])
        axes[1].set_ylim(self.ylim[0], self.ylim[1])
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

        self.magnetic_field_levels = levels
                
        figure, axes = plt.subplots(2,2,figsize=(self.image_size[0]*1.2, self.image_size[1]*1.8), dpi=self.dpi)
        cax = plt.axes([1, 0.05, 0.05, 0.9], label='magnetic field')
        # x1 direction 
        axes[0][0].contourf(self.axial_grid, self.radial_grid, data1, cmap=self.cmap, levels=levels)
        axes[0][0].set_title(f'Magnetic field in x1 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[0][0].set_xlim(self.xlim[0], self.xlim[1])
        axes[0][0].set_ylim(self.ylim[0], self.ylim[1])
        axes[0][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][0].set_xlabel(r'Axial distance [$R_{jet}$]')
        # x2 direction 
        axes[0][1].contourf(self.axial_grid, self.radial_grid, data2, cmap=self.cmap, levels=levels)
        axes[0][1].set_title(f'Magnetic field in x2 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[0][1].set_xlim(self.xlim[0], self.xlim[1])
        axes[0][1].set_ylim(self.ylim[0], self.ylim[1])
        axes[0][1].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][1].set_xlabel(r'Axial distance [$R_{jet}$]')
        # x3 direction 
        axes[1][0].contourf(self.axial_grid, self.radial_grid, data3, cmap=self.cmap, levels=levels)
        axes[1][0].set_title(f'Magnetic field in x3 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[1][0].set_xlim(self.xlim[0], self.xlim[1])
        axes[1][0].set_ylim(self.ylim[0], self.ylim[1])
        axes[1][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[1][0].set_xlabel(r'Axial distance [$R_{jet}$]')
        # magnitude
        axes[1][1].contourf(self.axial_grid, self.radial_grid, data4, cmap=self.cmap, levels=levels)
        axes[1][1].set_title(f'Magnetic field magnitude at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[1][1].set_xlim(self.xlim[0], self.xlim[1])
        axes[1][1].set_ylim(self.ylim[0], self.ylim[1])
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
        return [data1, data2, data3, data4]

    def velocity_quad(self,close=False,save=False):
        """
        Plots a graph with 4 plots of the velocity components and overall magnitude.

        Returns a matlpotlib figure object
        """
        data1 = self.plot_vx1(close=True)
        data2 = self.plot_vx2(close=True)
        data3 = self.plot_vx3(close=True)
        data4 = self.plot_velocity_field_magnitude(close=True)
        
        if self.mirrored == True:
            data4 = self._flip_multiply(data4)

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
        axes[0][0].set_title(f'Velocity field in x1 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[0][0].set_xlim(self.xlim[0], self.xlim[1])
        axes[0][0].set_ylim(self.ylim[0], self.ylim[1])
        axes[0][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][0].set_xlabel(r'Axial distance [$R_{jet}$]')
       
        axes[0][1].contourf(self.axial_grid, self.radial_grid, data2, cmap=self.cmap, levels=levels)
        axes[0][1].set_title(f'Velocity field in x2 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[0][1].set_xlim(self.xlim[0], self.xlim[1])
        axes[0][1].set_ylim(self.ylim[0], self.ylim[1])
        axes[0][1].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][1].set_xlabel(r'Axial distance [$R_{jet}$]')

        axes[1][0].contourf(self.axial_grid, self.radial_grid, data3, cmap=self.cmap, levels=levels)
        axes[1][0].set_title(f'Velocity field in x3 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[1][0].set_xlim(self.xlim[0], self.xlim[1])
        axes[1][0].set_ylim(self.ylim[0], self.ylim[1])
        axes[1][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[1][0].set_xlabel(r'Axial distance [$R_{jet}$]')
        
        axes[1][1].contourf(self.axial_grid, self.radial_grid, data4, cmap=self.cmap, levels=levels)
        axes[1][1].set_title(f'Velocity field magnitude at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[1][1].set_xlim(self.xlim[0], self.xlim[1])
        axes[1][1].set_ylim(self.ylim[0], self.ylim[1])
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
        axes[0][0].set_title(f'Alfvén Velocity field in x1 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[0][0].set_xlim(self.xlim[0], self.xlim[1])
        axes[0][0].set_ylim(self.ylim[0], self.ylim[1])
        axes[0][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][0].set_xlabel(r'Axial distance [$R_{jet}$]')
       
        axes[0][1].contourf(self.axial_grid, self.radial_grid, data2, cmap=self.cmap, levels=levels)
        axes[0][1].set_title(f'Alfvén Velocity field in x2 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[0][1].set_xlim(self.xlim[0], self.xlim[1])
        axes[0][1].set_ylim(self.ylim[0], self.ylim[1])
        axes[0][1].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][1].set_xlabel(r'Axial distance [$R_{jet}$]')

        axes[1][0].contourf(self.axial_grid, self.radial_grid, data3, cmap=self.cmap, levels=levels)
        axes[1][0].set_title(f'Alfvén Velocity field in x3 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[1][0].set_xlim(self.xlim[0], self.xlim[1])
        axes[1][0].set_ylim(self.ylim[0], self.ylim[1])
        axes[1][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[1][0].set_xlabel(r'Axial distance [$R_{jet}$]')
        
        axes[1][1].contourf(self.axial_grid, self.radial_grid, data4, cmap=self.cmap, levels=levels)
        axes[1][1].set_title(f'Alfvén Velocity field magnitude at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[1][1].set_xlim(self.xlim[0], self.xlim[1])
        axes[1][1].set_ylim(self.ylim[0], self.ylim[1])
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

    def magnetic_streamlines(self,close=False,save=False,levels=None,levels2=None):
        """
        Plots a vector plot of the magnetic field lines in the axial-radial plane,
        while plots the true magnitude of the magnetic field accounted in all 3 directions.
        """
        data =self.classifier(output_selector='all', delta_time=self.time_step)
        
        data1 = np.reshape(data[self.b_radial], self.XZ_shape).T
        data2 = np.reshape(data[self.b_azimuthal], self.XZ_shape).T
        data3 = np.reshape(data[self.b_axial], self.XZ_shape).T
        b_mag = np.sqrt(data1**2 + data2**2 + data3**2)

        cmp='jet'
        scmap = 'seismic'
        density=4

        subgrid_x_low = float(self.ini_content['[Grid]']['X3-grid']['Subgrids Data'][0][0])
        subgrid_y_low = float(self.ini_content['[Grid]']['X1-grid']['Subgrids Data'][0][0])
        subgrid_x = float(self.ini_content['[Grid]']['X3-grid']['Subgrids Data'][1][0])
        subgrid_y = float(self.ini_content['[Grid]']['X1-grid']['Subgrids Data'][1][0])
        subgrid_x_res = int(self.ini_content['[Grid]']['X3-grid']['Subgrids Data'][0][1])
        subgrid_y_res = int(self.ini_content['[Grid]']['X1-grid']['Subgrids Data'][0][1])

        ### taking the correct indecies for the unbiform subgrid
        if self.mirrored == False:
            subgrid_y_start = 0
            subgrid_y_end = subgrid_y_res
            figsize = (10,8)
            X2, Y2 = np.meshgrid(np.linspace(subgrid_x_low, subgrid_x, subgrid_x_res),
                                 np.linspace(subgrid_y_low, subgrid_y, subgrid_y_res))
        elif self.mirrored == True:
            data1 = self._flip_multiply(data1)
            data3 = self._flip_multiply(data3)
            b_mag = self._flip_multiply(b_mag)
            data2 = np.concatenate((-1*np.flip(data2), data2[1:]), axis=0)

            total_y = len(b_mag)
            mid_plane = int(total_y/2)
            subgrid_y_start = mid_plane - subgrid_y_res
            subgrid_y_end = mid_plane + subgrid_y_res
            figsize = (10,10)
            X2, Y2 = np.meshgrid(np.linspace(subgrid_x_low, subgrid_x, subgrid_x_res),
                                np.linspace(-1*subgrid_y, subgrid_y, subgrid_y_res*2))

        x1_comp = np.asarray(data1)[subgrid_y_start:subgrid_y_end,:subgrid_x_res]
        x2_comp = np.asarray(data2)[subgrid_y_start:subgrid_y_end,:subgrid_x_res]
        x3_comp = np.asarray(data3)[subgrid_y_start:subgrid_y_end,:subgrid_x_res]
        magnitude = np.asarray(b_mag)[subgrid_y_start:subgrid_y_end,:subgrid_x_res]

        if self.global_limits_bool == True:
            bx1_levels = self.set_levels(self.b_radial)
            bx2_levels = self.set_levels(self.b_azimuthal)
            bx3_levels = self.set_levels(self.b_axial)

            magnitude_levels = np.linspace(
                            0,
                            np.sqrt(bx1_levels[-1]**2 + bx2_levels[-1]**2 + bx3_levels[-1]**2),
                            128)

        if type(levels) == None :
            levels = np.linspace(0,2,128)
        
        if type(levels2) == None:
            levels2 = np.linspace(-2,2,128)

        X, Y = np.meshgrid(self.axial_grid[:subgrid_x_res],
                            self.radial_grid[subgrid_y_start:subgrid_y_end])

        figure, axes = plt.subplots(1,1,figsize=figsize,dpi=self.dpi)        
        axes.contourf(
                    X,
                    Y,
                    magnitude,
                    cmap=cmp,
                    alpha=0.95,
                    levels=levels)

        axes.streamplot(X2, 
                        Y2,
                        x3_comp,
                        x1_comp,
                        density=density,
                        color=x2_comp,
                        cmap=scmap,
                        integration_direction='both',
                        maxlength=20,
                        arrowsize=1.25,
                        arrowstyle='->',
                        linewidth=0.75,
                        norm=matplotlib.colors.Normalize(vmin=np.min(x2_comp), vmax=np.max(x2_comp)),
                        )
        norm = matplotlib.colors.Normalize(vmin=np.min(levels), vmax=np.max(levels))
        linenorm = matplotlib.colors.Normalize(vmin=np.min(levels2), vmax=np.max(levels2))
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmp), ax=axes, format='%.2f', label='Magnetic Field Strength',  location='bottom', pad=0.01)
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=linenorm, cmap=scmap), ax=axes, format='%.2f', label='Magnetic Field in direction x2',  location='bottom', pad=0.05)
        
        plt.xlim(subgrid_x_low, subgrid_x)
        plt.title(f'Magnetic field with field line direction at {self.time_step} {self.simulation_title}')
        
        if self.mirrored == False:
            plt.ylim(subgrid_y_low, subgrid_y)
        else:
            plt.ylim(-subgrid_y, subgrid_y)

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
        axes.set_title(f'Histogram data for {title} value (normalised) at {self.time_step} {self.simulation_title}')
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
        self.classifier(self.time_step, output_selector='all')
        self.cmap='hot'
        
        check = False
        if self.mirrored == True:
            self.mirrored = False
            check = True

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
            figure, axes = plt.subplots(figsize=(self.image_size[1], self.image_size[1]), dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Magnetic Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Magnetic field in x1 direction along Jet axis {self.simulation_title}')
            axes.set_xlim(self.xlim[0], self.xlim[1])
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
            figure, axes = plt.subplots(figsize=(self.image_size[1], self.image_size[1]), dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Magnetic Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Magnetic field in x2 direction along Jet axis {self.simulation_title}')
            axes.set_xlim(self.xlim[0], self.xlim[1])
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
            figure, axes = plt.subplots(figsize=(self.image_size[1], self.image_size[1]), dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Magnetic Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Magnetic field in x3 direction along Jet axis {self.simulation_title}')
            axes.set_xlim(self.xlim[0], self.xlim[1])
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
            figure, axes = plt.subplots(figsize=(self.image_size[1], self.image_size[1]), dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='General Lagrangian Multiplier', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the GLM along Jet axis {self.simulation_title}')
            axes.set_xlim(self.xlim[0], self.xlim[1])
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
            figure, axes = plt.subplots(figsize=(self.image_size[1], self.image_size[1]), dpi=self.dpi)
            pl = axes.contourf(X, Y, np.log(axial_array), cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='log(Pressure) Field', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the log(Pressure) along Jet axis {self.simulation_title}')
            axes.set_xlim(self.xlim[0], self.xlim[1])
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
            figure, axes = plt.subplots(figsize=(self.image_size[1], self.image_size[1]), dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Density Field', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Density along Jet axis {self.simulation_title}')
            axes.set_xlim(self.xlim[0], self.xlim[1])
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
            figure, axes = plt.subplots(figsize=(self.image_size[1], self.image_size[1]), dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Velocity Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Velocity in x1 direction along Jet axis {self.simulation_title}')
            axes.set_xlim(self.xlim[0], self.xlim[1])
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
            figure, axes = plt.subplots(figsize=(self.image_size[1], self.image_size[1]), dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Velocity Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Velocity in x2 direction along Jet axis {self.simulation_title}')
            axes.set_xlim(self.xlim[0], self.xlim[1])
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
            figure, axes = plt.subplots(figsize=(self.image_size[1], self.image_size[1]), dpi=self.dpi)
            pl = axes.contourf(X, Y, axial_array, cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Velocity Field magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Velocity in x3 direction along Jet axis {self.simulation_title}')
            axes.set_xlim(self.xlim[0], self.xlim[1])
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
            figure, axes = plt.subplots(figsize=(self.image_size[1], self.image_size[1]), dpi=self.dpi)
            pl = axes.contourf(X, Y, alfven_velocity(B=magfield, density=density), cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Alfvén Velocity magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Alfvén Velocity in x1 direction along Jet axis {self.simulation_title}')
            axes.set_xlim(self.xlim[0], self.xlim[1])
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
            figure, axes = plt.subplots(figsize=(self.image_size[1], self.image_size[1]), dpi=self.dpi)
            pl = axes.contourf(X, Y, alfven_velocity(B=magfield, density=density), cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Alfvén Velocity magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Alfvén Velocity in x2 direction along Jet axis {self.simulation_title}')
            axes.set_xlim(self.xlim[0], self.xlim[1])
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
            figure, axes = plt.subplots(figsize=(self.image_size[1], self.image_size[1]), dpi=self.dpi)
            pl = axes.contourf(X, Y, alfven_velocity(B=magfield, density=density), cmap=self.cmap, levels=128)
            
            figure.colorbar(pl, 
                location='right', 
                shrink=0.95, 
                aspect=20,
                pad=0.02, 
                label='Alfvén Velocity magnitude', 
                format='%.2f'
                )
            
            axes.set_title(f'Space-Time diagram of the Alfvén Velocity in x3 direction along Jet axis {self.simulation_title}')
            axes.set_xlim(self.xlim[0], self.xlim[1])
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

        if self.mirrored == False and check == True:
            self.mirrored = True

        return


    def plot_magnetoacoustics(self,close=False,save=False,density_contour=False,levels=None):
        """
        This method shows the fast and slow magneto acoustic waves and their components

        Returns a list of wave velocity component arrays in the following order: 
            [slow1, slow2, slow3, fast1, fast2, fast3]
        """
        if levels == None:
            cms_levels = 128
        elif type(levels) == int:
            cms_levels = levels
        elif type(levels) == float:
            cms_levels = int(levels)
        elif type(levels) == list or type(levels) == tuple:
            if len(levels) == 1:
                cms_levels = np.linspace(0, levels[0], 128)
            elif len(levels) == 2:
                if levels[0] == levels[1]:
                    raise ValueError('The given values for levels are the same, thus cannot equally space them.')
                else:
                    cms_levels = np.linspace(np.min(levels), np.max(levels), 128)
            elif len(levels) == 3:
                if levels[0] == levels[1]:
                    raise ValueError(f'The min() and max() values for levels are the same, thus cannot equally space them to {levels[2]} steps.')
                else:
                    cms_levels = np.linspace(levels[0], levels[1], levels[2])

        data = self.classifier(delta_time=self.time_step ,output_selector='all')
        bx1 = np.reshape(data[self.b_radial], self.XZ_shape).T
        bx2 = np.reshape(data[self.b_azimuthal], self.XZ_shape).T
        bx3 = np.reshape(data[self.b_axial], self.XZ_shape).T
        prs = np.reshape(data[self.pressure], self.XZ_shape).T
        rho = np.reshape(data[self.density], self.XZ_shape).T
        gamma = 5/3

        slow1, fast1 = magneto_acoustic_velocity(bx1, prs, rho, gamma)
        slow2, fast2 = magneto_acoustic_velocity(bx2, prs, rho, gamma)
        slow3, fast3 = magneto_acoustic_velocity(bx3, prs, rho, gamma)

        if self.mirrored == True:
            slow1 = self._flip_multiply(slow1)
            slow2 = self._flip_multiply(slow2)
            slow3 = self._flip_multiply(slow3)
            fast1 = self._flip_multiply(fast1)
            fast2 = self._flip_multiply(fast2)
            fast3 = self._flip_multiply(fast3)

        figure, axes = plt.subplots(3,2,figsize=(self.image_size[0]*2, self.image_size[0]*3), dpi=self.dpi)
        
        divider = mal(axes[0][0])
        cax = divider.append_axes('right',size='5%',pad=0.25)
        pls1 = axes[0][0].contourf(self.axial_grid, self.radial_grid, slow1, cmap='jet', levels=cms_levels, alpha=0.95)
        plt.colorbar(pls1,cax,ticks=np.linspace(np.min(slow1),np.max(slow1), 6))
        axes[0][0].set_title(f'Slow wave velocity in x1 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[0][0].set_xlim(self.xlim[0], self.xlim[1])
        axes[0][0].set_ylim(self.ylim[0], self.ylim[1])
        axes[0][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][0].set_xlabel(r'Axial distance [$R_{jet}$]')
        
        divider = mal(axes[1][0])
        cax = divider.append_axes('right',size='5%',pad=0.25)
        pls2 = axes[1][0].contourf(self.axial_grid, self.radial_grid, slow2, cmap='jet', levels=cms_levels, alpha=0.95)
        plt.colorbar(pls2,cax,ticks=np.linspace(np.min(slow2),np.max(slow2), 6))
        axes[1][0].set_title(f'Slow wave velocity in x2 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[1][0].set_xlim(self.xlim[0], self.xlim[1])
        axes[1][0].set_ylim(self.ylim[0], self.ylim[1])
        axes[1][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[1][0].set_xlabel(r'Axial distance [$R_{jet}$]')
        
        divider = mal(axes[2][0])
        cax = divider.append_axes('right',size='5%',pad=0.25)
        pls3 = axes[2][0].contourf(self.axial_grid, self.radial_grid, slow3, cmap='jet', levels=cms_levels, alpha=0.95)
        plt.colorbar(pls3,cax,ticks=np.linspace(np.min(slow3),np.max(slow3), 6))
        axes[2][0].set_title(f'Slow wave velocity in x3 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[2][0].set_xlim(self.xlim[0], self.xlim[1])
        axes[2][0].set_ylim(self.ylim[0], self.ylim[1])
        axes[2][0].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[2][0].set_xlabel(r'Axial distance [$R_{jet}$]')
        
        divider = mal(axes[0][1])
        cax = divider.append_axes('right',size='5%',pad=0.25)
        plf1 = axes[0][1].contourf(self.axial_grid, self.radial_grid, fast1, cmap='jet', levels=cms_levels, alpha=0.95)
        plt.colorbar(plf1,cax,ticks=np.linspace(np.min(fast1),np.max(fast1), 6))
        axes[0][1].set_title(f'Fast wave velocity in x1 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[0][1].set_xlim(self.xlim[0], self.xlim[1])
        axes[0][1].set_ylim(self.ylim[0], self.ylim[1])
        axes[0][1].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[0][1].set_xlabel(r'Axial distance [$R_{jet}$]')
        
        divider = mal(axes[1][1])
        cax = divider.append_axes('right',size='5%',pad=0.25)
        plf2 = axes[1][1].contourf(self.axial_grid, self.radial_grid, fast2, cmap='jet', levels=cms_levels, alpha=0.95)
        plt.colorbar(plf2,cax,ticks=np.linspace(np.min(fast2),np.max(fast2), 6))
        axes[1][1].set_title(f'Fast wave velocity in x2 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[1][1].set_xlim(self.xlim[0], self.xlim[1])
        axes[1][1].set_ylim(self.ylim[0], self.ylim[1])
        axes[1][1].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[1][1].set_xlabel(r'Axial distance [$R_{jet}$]')
        
        divider = mal(axes[2][1])
        cax = divider.append_axes('right',size='5%',pad=0.25)
        plf3 = axes[2][1].contourf(self.axial_grid, self.radial_grid, fast3, cmap='jet', levels=cms_levels, alpha=0.95)
        plt.colorbar(plf3,cax,ticks=np.linspace(np.min(fast3),np.max(fast3), 6))
        axes[2][1].set_title(f'Fast wave velocity in x3 direction at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} {self.simulation_title}')
        axes[2][1].set_xlim(self.xlim[0], self.xlim[1])
        axes[2][1].set_ylim(self.ylim[0], self.ylim[1])
        axes[2][1].set_ylabel(r'Radial distance [$R_{jet}$]')
        axes[2][1].set_xlabel(r'Axial distance [$R_{jet}$]')

        if density_contour == True:
            axes[0][0].contour(self.axial_grid, self.radial_grid, rho, cmap='Greys', levels=16, alpha=0.4)
            axes[1][0].contour(self.axial_grid, self.radial_grid, rho, cmap='Greys', levels=16, alpha=0.4)
            axes[2][0].contour(self.axial_grid, self.radial_grid, rho, cmap='Greys', levels=16, alpha=0.4)
            axes[0][1].contour(self.axial_grid, self.radial_grid, rho, cmap='Greys', levels=16, alpha=0.4)
            axes[1][1].contour(self.axial_grid, self.radial_grid, rho, cmap='Greys', levels=16, alpha=0.4)
            axes[2][1].contour(self.axial_grid, self.radial_grid, rho, cmap='Greys', levels=16, alpha=0.4)
       
        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}cms_plots'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}cms_plots/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()
        
        return [slow1, slow2, slow3, fast1, fast2, fast3]

    def find_shocks(self, plot_mach=False, plot_shock=None):
        """
        This method loops through the spatial grid, and finds MHD shocks

        Return: plot data
        """
        jet_velocity = float(self.ini_content['[Parameters]']['JET_VEL'][0])
        gamma = 5/3
        prs = np.reshape(self.classifier(delta_time=self.time_step, output_selector='all')[self.pressure], self.XZ_shape).T
        rho = np.reshape(self.classifier(delta_time=self.time_step, output_selector='all')[self.density], self.XZ_shape).T
        
        vx1 = np.reshape(self.classifier(delta_time=self.time_step, output_selector='all')[self.radial_velocity], self.XZ_shape).T
        bx1 = np.reshape(self.classifier(delta_time=self.time_step, output_selector='all')[self.b_radial], self.XZ_shape).T
        vx2 = np.reshape(self.classifier(delta_time=self.time_step, output_selector='all')[self.azimuthal_velocity], self.XZ_shape).T
        bx2 = np.reshape(self.classifier(delta_time=self.time_step, output_selector='all')[self.b_azimuthal], self.XZ_shape).T
        vx3 = np.reshape(self.classifier(delta_time=self.time_step, output_selector='all')[self.axial_velocity], self.XZ_shape).T
        bx3 = np.reshape(self.classifier(delta_time=self.time_step, output_selector='all')[self.b_axial], self.XZ_shape).T

        av1 = alfven_velocity(bx1, rho) 
        av2 = alfven_velocity(bx2, rho)
        av3 = alfven_velocity(bx3, rho)

        slow1, fast1 = magneto_acoustic_velocity(bx1, prs, rho, gamma)
        slow2, fast2 = magneto_acoustic_velocity(bx2, prs, rho, gamma)
        slow3, fast3 = magneto_acoustic_velocity(bx3, prs, rho, gamma)

        mach_fast = mach_number(np.sqrt(vx1**2 + vx2**2 + vx3**2), np.sqrt(fast1**2 + fast2**2 + fast3**2))
        mach_alfv = mach_number(np.sqrt(vx1**2 + vx2**2 + vx3**2), np.sqrt(av1**2 + av2**2 + av3**2))
        mach_slow = mach_number(np.sqrt(vx1**2 + vx2**2 + vx3**2), np.sqrt(slow1**2 + slow2**2 + slow3**2))

        #### Super fast MS to Super Alfvénic ####
        
        zero_array = np.zeros_like(mach_fast)
        fast_shock_ax = []
        fast_shock_ra = []
        for j, row in enumerate(zero_array):
            for i, val in enumerate(row):
                if (i+1 == len(row)) or (j+1 == len(zero_array)):
                    pass
                else:
                    if (mach_fast[j,i] > 1) and (mach_fast[j,i+1] < 1) and (mach_alfv[j,i+1] > 1):
                        fast_shock_ax.append(self.axial_grid[i])
                        fast_shock_ra.append(self.radial_grid[j])
                    if (mach_fast[j,i] > 1) and (mach_fast[j+1,i] < 1) and (mach_alfv[j+1,i] > 1):
                        fast_shock_ax.append(self.axial_grid[i])
                        fast_shock_ra.append(self.radial_grid[j])
                    if (mach_fast[j,i] > 1) and (mach_fast[j+1,i+1] < 1) and (mach_alfv[j+1,i+1] > 1):
                        fast_shock_ax.append(self.axial_grid[i])
                        fast_shock_ra.append(self.radial_grid[j])
                
        #### Intermed. Shocks #####

        inter_shock_ax1 = []
        inter_shock_ra1 = []
        inter_shock_ax2 = []
        inter_shock_ra2 = []
        inter_shock_ax3 = []
        inter_shock_ra3 = []
        inter_shock_ax4 = []
        inter_shock_ra4 = []
        

        
        for j, row in enumerate(zero_array):
            for i, val in enumerate(row):
                if (i+1 == len(row)) or (j+1 == len(zero_array)):
                    pass
                else:
                    ### Super Fast to Sub Alfvénic, Super Slow
                    # perpendicular shocks
                    if (mach_fast[j,i] > 1) and (mach_alfv[j,i+1] < 1) and (mach_slow[j,i+1] > 1):
                        inter_shock_ax1.append(self.axial_grid[i])
                        inter_shock_ra1.append(self.radial_grid[j])
                    # parallel shocks
                    if (mach_fast[j,i] > 1) and (mach_alfv[j+1,i] < 1) and (mach_slow[j+1,i] > 1):
                        inter_shock_ax1.append(self.axial_grid[i])
                        inter_shock_ra1.append(self.radial_grid[j])
                    # oblique shocks
                    if (mach_fast[j,i] > 1) and (mach_alfv[j+1,i+1] < 1) and (mach_slow[j+1,i+1] > 1):
                        inter_shock_ax1.append(self.axial_grid[i])
                        inter_shock_ra1.append(self.radial_grid[j])
                    
                    ## Sub Fast, Super Alfvénic to Sub Alfvénic, Super Slow
                    # perpendicular shocks
                    if (mach_alfv[j,i] > 1) and (mach_fast[j,i] < 1) and (mach_alfv[j,i+1] < 1) and (mach_slow[j,i+1] > 1):
                        inter_shock_ax2.append(self.axial_grid[i])
                        inter_shock_ra2.append(self.radial_grid[j])
                    # parallel shocks
                    if (mach_alfv[j,i] > 1) and (mach_fast[j,i] < 1) and (mach_alfv[j+1,i] < 1) and (mach_slow[j+1,i] > 1):
                        inter_shock_ax2.append(self.axial_grid[i])
                        inter_shock_ra2.append(self.radial_grid[j])
                    # oblique shocks
                    if (mach_alfv[j,i] > 1) and (mach_fast[j,i] < 1) and (mach_alfv[j+1,i+1] < 1) and (mach_slow[j+1,i+1] > 1):
                        inter_shock_ax2.append(self.axial_grid[i])
                        inter_shock_ra2.append(self.radial_grid[j])
                   
                    ### Sub Fast, Super Alfvénic to Sub Slow
                    # perpendicular shocks
                    if (mach_alfv[j,i] > 1) and (mach_fast[j,i] < 1) and (mach_slow[j,i+1] < 1):
                        inter_shock_ax3.append(self.axial_grid[i])
                        inter_shock_ra3.append(self.radial_grid[j])
                    # parallel shocks
                    if (mach_alfv[j,i] > 1) and (mach_fast[j,i] < 1) and (mach_slow[j+1,i] < 1):
                        inter_shock_ax3.append(self.axial_grid[i])
                        inter_shock_ra3.append(self.radial_grid[j])
                    # oblique shocks
                    if (mach_alfv[j,i] > 1) and (mach_fast[j,i] < 1) and (mach_slow[j+1,i+1] < 1):
                        inter_shock_ax3.append(self.axial_grid[i])
                        inter_shock_ra3.append(self.radial_grid[j])

                    ### Hydrodynamic
                    # perpendicular shocks
                    if (mach_fast[j,i] > 1) and (mach_slow[j,i+1] < 1):
                        inter_shock_ax4.append(self.axial_grid[i])
                        inter_shock_ra4.append(self.radial_grid[j])
                    # parallel shocks
                    if (mach_fast[j,i] > 1) and (mach_slow[j+1,i] < 1):
                        inter_shock_ax4.append(self.axial_grid[i])
                        inter_shock_ra4.append(self.radial_grid[j])
                    # oblique shocks
                    if (mach_fast[j,i] > 1) and (mach_slow[j+1,i+1] < 1):
                        inter_shock_ax4.append(self.axial_grid[i])
                        inter_shock_ra4.append(self.radial_grid[j])

        #### Slow shocks #####

        slow_shock_ax = []
        slow_shock_ra = []
        for j, row in enumerate(zero_array):
            for i, val in enumerate(row):
                if (i+1 == len(row)) or (j+1 == len(zero_array)):
                    pass
                else:
                    if (mach_slow[j,i] > 1) and (mach_alfv[j,i+1] < 1) and (mach_slow[j,i+1] < 1):
                        slow_shock_ax.append(self.axial_grid[i])
                        slow_shock_ra.append(self.radial_grid[j])
                    if (mach_slow[j,i] > 1) and (mach_alfv[j+1,i] < 1) and (mach_slow[j+1,i] < 1):
                        slow_shock_ax.append(self.axial_grid[i])
                        slow_shock_ra.append(self.radial_grid[j])
                    if (mach_slow[j,i] > 1) and (mach_alfv[j+1,i+1] < 1) and (mach_slow[j+1,i+1] < 1):
                        slow_shock_ax.append(self.axial_grid[i])
                        slow_shock_ra.append(self.radial_grid[j])
        
        if plot_shock != None:
            figureS, axesS = plt.subplots(figsize=(self.image_size[0]*1.25, self.image_size[1]*1.25), dpi=self.dpi*2)
            if plot_shock == True:
                axesS.plot(fast_shock_ax, fast_shock_ra, '+', lw=0.25, color='red', markersize=5, label='Fast 1-2 Shocks', alpha=0.25)
                axesS.plot(inter_shock_ax1, inter_shock_ra1, '+', lw=0.25, color='yellow', markersize=5, label='Inter 1-3 Shocks', alpha=0.25)
                axesS.plot(inter_shock_ax2, inter_shock_ra2, '+', lw=0.25, color='green', markersize=5, label='Inter 2-3 Shocks', alpha=0.25)
                axesS.plot(inter_shock_ax3, inter_shock_ra3, '+', lw=0.25, color='orange', markersize=5, label='Inter 2-4 Shocks', alpha=0.25)
                axesS.plot(inter_shock_ax4, inter_shock_ra4, '+', lw=0.25, color='cyan', markersize=5, label='Hydro 1-4 Shocks', alpha=0.25)
                axesS.plot(slow_shock_ax, slow_shock_ra, '+', lw=0.25, color='blue', markersize=5, label='Slow 3-4 Shocks', alpha=0.25)

            elif plot_shock == 'slow':
                axesS.plot(slow_shock_ax, slow_shock_ra, '+', lw=0.25, color='blue', markersize=5, label='Slow 3-4 Shocks', alpha=0.25)
            elif plot_shock == 'inter':
                axesS.plot(inter_shock_ax1, inter_shock_ra1, '+', lw=0.25, color='yellow', markersize=5, label='Inter 1-3 Shocks', alpha=0.25)
                axesS.plot(inter_shock_ax2, inter_shock_ra2, '+', lw=0.25, color='green', markersize=5, label='Inter 2-3 Shocks', alpha=0.25)
                axesS.plot(inter_shock_ax3, inter_shock_ra3, '+', lw=0.25, color='orange', markersize=5, label='Inter 2-4 Shocks', alpha=0.25)
                axesS.plot(inter_shock_ax4, inter_shock_ra4, '+', lw=0.25, color='cyan', markersize=5, label='Hydro 1-4 Shocks', alpha=0.25)
            elif plot_shock == 'fast':
                axesS.plot(fast_shock_ax, fast_shock_ra, '+', lw=0.25, color='red', markersize=5, label='Fast 1-2 Shocks', alpha=0.25)

            axesS.legend()
            axesS.set_xlim(self.xlim[0], self.xlim[1])
            axesS.set_ylim(self.ylim[0], self.ylim[1])
            axesS.set_ylabel(r'Radial distance [$R_{jet}$]')
            axesS.set_xlabel(r'Axial distance [$R_{jet}$]')


        if plot_mach == True:
            figure, axes = plt.subplots(3,1,figsize=(self.image_size[0], self.image_size[0]*3), dpi=self.dpi)

            divider = mal(axes[0])
            cax = divider.append_axes('right',size='5%',pad=0.25)
            pls1 = axes[0].contourf(self.axial_grid, self.radial_grid, np.log(mach_fast), cmap='jet', levels=48, alpha=0.95)
            plt.colorbar(pls1,cax,ticks=np.linspace(np.min(np.log(mach_fast)),np.max(np.log(mach_fast)), 6))
            axes[0].set_title(f'Fast wave Mach at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f}, $M_m$ = {np.max(mach_fast)} {self.simulation_title}')
            axes[0].set_xlim(self.xlim[0], self.xlim[1])
            axes[0].set_ylim(self.ylim[0], self.ylim[1])
            axes[0].set_ylabel(r'Radial distance [$R_{jet}$]')
            axes[0].set_xlabel(r'Axial distance [$R_{jet}$]')

            divider = mal(axes[1])
            cax = divider.append_axes('right',size='5%',pad=0.25)
            pls1 = axes[1].contourf(self.axial_grid, self.radial_grid, np.log(mach_alfv), cmap='jet', levels=48, alpha=0.95)
            plt.colorbar(pls1,cax,ticks=np.linspace(np.min(np.log(mach_alfv)),np.max(np.log(mach_alfv)), 6))
            axes[1].set_title(f'Alfvén wave Mach at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f}, $M_m$ = {np.max(mach_alfv)} {self.simulation_title}')
            axes[1].set_xlim(self.xlim[0], self.xlim[1])
            axes[1].set_ylim(self.ylim[0], self.ylim[1])
            axes[1].set_ylabel(r'Radial distance [$R_{jet}$]')
            axes[1].set_xlabel(r'Axial distance [$R_{jet}$]')

            divider = mal(axes[2])
            cax = divider.append_axes('right',size='5%',pad=0.25)
            pls1 = axes[2].contourf(self.axial_grid, self.radial_grid, np.log(mach_slow), cmap='jet', levels=48, alpha=0.95)
            plt.colorbar(pls1,cax,ticks=np.linspace(np.min(np.log(mach_slow)),np.max(np.log(mach_slow)), 6))
            axes[2].set_title(f'Slow wave Mach at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f}, $M_m$ = {np.max(mach_slow)} {self.simulation_title}')
            axes[2].set_xlim(self.xlim[0], self.xlim[1])
            axes[2].set_ylim(self.ylim[0], self.ylim[1])
            axes[2].set_ylabel(r'Radial distance [$R_{jet}$]')
            axes[2].set_xlabel(r'Axial distance [$R_{jet}$]')


        return [np.log(mach_slow), np.log(mach_alfv), np.log(mach_fast)]
