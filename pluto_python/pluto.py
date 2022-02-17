#%%
"""
This module contains the super class for the package:
    - all data is stored here
    - all subsidary data is calculated here
"""
from pluto_python.calculator import alfven_velocity
from pluto_python.calculator import magneto_acoustic_velocity
from pluto_python.calculator import mach_number
from pluto_python.calculator import magnetic_pressure
from pluto_python.calculator import energy_density
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.animation as pla
import matplotlib
import os

class py3Pluto:
    """
    This class makes an object from reading in the file at the selected timestep
    with all its simulated and calculated subsidary data ready formated for plotting
    using the subclasses like mhd_jet.
    """
    def __init__(
        self,
        data_path,
        time_step = 0,
        ini_path = None,
        dpi = 300,
        image_size = (10,5),
        ylim = None,
        xlim = None,
        cmap = 'bwr',
        global_limits = False,
        mirrored = False,
        gamma = 5/3,
        
    ):

        self.data_path = data_path
        self.dpi = dpi
        self.image_size = image_size
        self.time_step = time_step
        self.global_limits_bool = global_limits
        self.cmap = cmap
        self.ini_path = ini_path
        self.mirrored = mirrored
        self.closure = False
        self.simulation_title = self.data_path.split('/')[-2]
        self.gamma = gamma

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

        self._reader()
        self.shape_limits = self._read_ini()

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
            
        
        self.XZ_shape = (self.shape_limits['X3-grid'], self.shape_limits['X1-grid'])
        self.calculate_data(self.time_step)

        ### Define axes limits from defaults from the ini file if not given. To see max grid,
        ### If global limit bool is set to False, the limits are not run
        if self.global_limits_bool == True:
            self.global_limits = self._get_limits()

    def _read_ini(self):
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

    def _reader(self):
        path = os.getcwd()
        files  = os.listdir(self.data_path)
        extension = '.h5'
        self.data_list = [file for file in files if extension in file and 'out' not in file]
        return self.data_list
        
    def classifier(self, delta_time):
        """
        delta_time: is the time step of which is number of the file by pluto
        or 'all' <- will be removed soon
        """
        self._reader()
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

    def _get_limits(self,log=False):
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
            data = self.classifier(delta_time=step)
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

    def calculate_data(self, time):
        """
        This method is to calculate all subsidary data sets from the simulation data.
        sets global variables that are accessible by sub classes to use
        """
        ############################## Magnetic fields ##############################
        self.bx1 = np.reshape(
                    self.classifier(
                        delta_time=time,
                        )[self.b_radial],
                        self.XZ_shape).T
        self.bx2 = np.reshape(
                    self.classifier(
                        delta_time=time,
                        )[self.b_azimuthal],
                        self.XZ_shape).T
        self.bx3 = np.reshape(
                    self.classifier(
                        delta_time=time,
                        )[self.b_axial],
                        self.XZ_shape).T
        self.magnetic_field_magnitude = np.sqrt(self.bx1**2 + self.bx2**2 + self.bx3**2)
        ############################## Velocities ##############################
        self.vx1 = np.reshape(
                    self.classifier(
                        delta_time=time,
                        )[self.radial_velocity],
                        self.XZ_shape).T
        self.vx2 = np.reshape(
                    self.classifier(
                        delta_time=time,
                        )[self.azimuthal_velocity],
                        self.XZ_shape).T
        self.vx3 = np.reshape(
                    self.classifier(
                        delta_time=time,
                        )[self.axial_velocity],
                        self.XZ_shape).T
        self.velocity_magnitude = np.sqrt(self.vx1**2 + self.vx2**2 + self.vx3**2)
        ############################## Pressure, Density and Tracer ##############################
        self.prs = np.reshape(
                    self.classifier(
                        delta_time=time,
                        )[self.pressure],
                        self.XZ_shape).T
        self.rho = np.reshape(
                    self.classifier(
                        delta_time=time,
                        )[self.density],
                        self.XZ_shape).T
        self.tr1 = np.reshape(
                    self.classifier(
                        delta_time=time,
                        )[self.tracer1],
                        self.XZ_shape).T
        ############################## General Lagrangian Multiplier ##############################
        self.glm = np.reshape(
                    self.classifier(
                        delta_time=time,
                        )[self.psi_glm],
                        self.XZ_shape).T
        ############################## Alfvén Velocities ##############################
        self.avx1 = alfven_velocity(self.bx1, self.rho)
        self.avx2 = alfven_velocity(self.bx2, self.rho)
        self.avx3 = alfven_velocity(self.bx3, self.rho)
        self.alfvén_velocity_magnitude = np.sqrt(self.avx1**2 + self.avx2**2 + self.avx3**2)
        ############################## Magneto acoustic Waves ##############################
        self.slow_ms_x1, self.fast_ms_x1 = magneto_acoustic_velocity(self.bx1, self.prs, self.rho, self.gamma)
        self.slow_ms_x2, self.fast_ms_x2 = magneto_acoustic_velocity(self.bx2, self.prs, self.rho, self.gamma)
        self.slow_ms_x3, self.fast_ms_x3 = magneto_acoustic_velocity(self.bx3, self.prs, self.rho, self.gamma)
        self.fast_ms_velocity_magnitude = np.sqrt(self.fast_ms_x1**2 + self.fast_ms_x2**2 + self.fast_ms_x3**2)
        self.slow_ms_velocity_magnitude = np.sqrt(self.slow_ms_x1**2 + self.slow_ms_x2**2 + self.slow_ms_x3**2)
        ############################## Mach numbers ##############################
        self.mach_fast = mach_number(self.velocity_magnitude, self.fast_ms_velocity_magnitude)
        self.mach_slow = mach_number(self.velocity_magnitude, self.slow_ms_velocity_magnitude)
        self.mach_alfvén = mach_number(self.velocity_magnitude, self.alfvén_velocity_magnitude)
        ############################## Plasma Beta ##############################
        self.beta = (self.prs / self.magnetic_field_magnitude)
        ############################## Magnetic pressure ##############################
        self.magnetic_prs = magnetic_pressure(self.magnetic_field_magnitude)
        ############################## Energy density ##############################
        self.kinetic_energy_density, \
        self.thermal_energy_density, \
        self.magnetic_energy_density = energy_density(
                                                self.prs,
                                                self.rho,
                                                self.velocity_magnitude,
                                                self.magnetic_field_magnitude,
                                                self.gamma
        )
        self.total_energy_density = self.kinetic_energy_density + self.thermal_energy_density + self.magnetic_energy_density
        ############################## Energy ##############################
        # Axial element lengths
        axial_differences = []
        for z in self.axial_grid:
            if len(axial_differences) == 0:
                axial_differences.append(z)
            else:
                axial_differences.append(z-axial_differences[-1])
        # Radial element area
        radial_scaler = []
        for index, rad in enumerate(self.radial_grid):
            if index == 0:
                dsurf = np.pi*rad**2
            else:
                dsurf = np.pi * ((rad)**2 - (self.radial_grid[index-1])**2)
            radial_scaler.append(dsurf)
        # Volume elements
        volumes = np.transpose([x*np.asarray(radial_scaler) for x in axial_differences])
        # System energies
        self.kinetic_energy_sys = self.kinetic_energy_density * volumes
        self.thermal_energy_sys = self.thermal_energy_density * volumes
        self.magnetic_energy_sys = self.magnetic_energy_density * volumes
        self.total_energy_sys = self.kinetic_energy_sys + self.thermal_energy_sys + self.magnetic_energy_sys 
        # Jet energies
        self.kinetic_energy_jet = self.kinetic_energy_sys * self.tr1
        self.thermal_energy_jet = self.thermal_energy_sys * self.tr1
        self.magnetic_energy_jet = self.magnetic_energy_sys * self.tr1
        self.total_energy_jet = self.total_energy_sys * self.tr1
        ############################## Power ##############################
        # System power
        self.kinetic_power_sys = np.asarray([x/axial_differences for x in self.kinetic_energy_sys]) * self.vx3
        self.thermal_power_sys = np.asarray([x/axial_differences for x in self.thermal_energy_sys]) * self.vx3
        self.magnetic_power_sys = np.asarray([x/axial_differences for x in self.magnetic_energy_sys]) * self.vx3
        self.total_power_sys = self.kinetic_power_sys + self.thermal_power_sys + self.magnetic_power_sys
        # Jet power
        self.kinetic_power_jet = self.kinetic_power_sys * self.tr1
        self.thermal_power_jet = self.thermal_power_sys * self.tr1
        self.magnetic_power_jet = self.magnetic_power_sys * self.tr1
        self.total_power_jet = self.kinetic_power_jet + self.thermal_power_jet + self.magnetic_power_jet


        if self.mirrored == True:
            self.bx1 = self._flip_multiply(self.bx1)
            self.bx2 = self._flip_multiply(self.bx2)
            self.bx3 = self._flip_multiply(self.bx3)
            self.magnetic_field_magnitude = self._flip_multiply(self.magnetic_field_magnitude)
            self.vx1 = self._flip_multiply(self.vx1)
            self.vx2 = self._flip_multiply(self.vx2)
            self.vx3 = self._flip_multiply(self.vx3)
            self.velocity_magnitude = self._flip_multiply(self.velocity_magnitude)
            self.prs = self._flip_multiply(self.prs)
            self.rho = self._flip_multiply(self.rho)
            self.tr1 = self._flip_multiply(self.tr1)
            self.glm = self._flip_multiply(self.glm)
            self.avx1 = self._flip_multiply(self.avx1)
            self.avx2 = self._flip_multiply(self.avx2)
            self.avx3 = self._flip_multiply(self.avx3)
            self.alfvén_velocity_magnitude = self._flip_multiply(self.alfvén_velocity_magnitude)
            self.slow_ms_x1 = self._flip_multiply(self.slow_ms_x1)
            self.fast_ms_x1 = self._flip_multiply(self.fast_ms_x1)
            self.slow_ms_x2 = self._flip_multiply(self.slow_ms_x2)
            self.fast_ms_x2 = self._flip_multiply(self.fast_ms_x2)
            self.slow_ms_x3 = self._flip_multiply(self.slow_ms_x3)
            self.fast_ms_x3 = self._flip_multiply(self.fast_ms_x3)
            self.fast_ms_velocity_magnitude = self._flip_multiply(self.fast_ms_velocity_magnitude)
            self.slow_ms_velocity_magnitude = self._flip_multiply(self.slow_ms_velocity_magnitude)
            self.mach_fast = self._flip_multiply(self.mach_fast)
            self.mach_slow = self._flip_multiply(self.mach_slow)
            self.mach_alfvén = self._flip_multiply(self.mach_alfvén)
            self.beta = self._flip_multiply(self.beta)
            self.magnetic_prs = self._flip_multiply(self.magnetic_prs)
            
            