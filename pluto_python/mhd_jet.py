#%%
from .pluto import py3Pluto
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal

import numpy as np
import matplotlib.pyplot as plt
import time

class mhd_jet(py3Pluto):
    """
    This class is a sub-class of py3Pluto, which stores all information from the simulations
    and claculations, which can be plotted using the plot or hist methods within this class.
    """
    def __init__(self,
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

        
        super().__init__(
            data_path = data_path,
            time_step = time_step,
            ini_path = ini_path,
            dpi = dpi,
            image_size = image_size,
            ylim = ylim,
            xlim = xlim,
            cmap = cmap,
            global_limits = global_limits,
            mirrored = mirrored,
            gamma = gamma
        )

    def _data(self, data2plot=None, log=False, close=False, save=False):
        """
        Method plots individual data sets that are given as an argument.
        """
        if data2plot == None:
            text = """
            Select one of the following arguments to plot them:
                Magnetic field at x1 axis: 'bx1'
                Magnetic field at x2 axis: 'bx2'
                Magnetic field at x3 axis: 'bx3'
                Magnetic Field magnitude:  'bxs'

                Velocity at x1 axis: 'vx1'
                Velocity at x2 axis: 'vx2'
                Velocity at x3 axis: 'vx3'
                Velocity magnitude:  'vxs'

                Pressure field:     'prs'
                Density field:      'rho'
                Jet Tracer:         'tr1'

                General Lagrangian Multiplier: 'glm'

                Alfvén Wave Velocity along x1 axis: 'avx1'
                Alfvén Wave Velocity along x2 axis: 'avx2'
                Alfvén Wave Velocity along x3 axis: 'avx3'
                Alfvén Wave Velocity Magnitude:     'avxs'

                Fast Magneto-acoustic Wave along x1 axis:   'msfx1'
                Fast Magneto-acoustic Wave along x2 axis:   'msfx2'
                Fast Magneto-acoustic Wave along x3 axis:   'msfx3'
                Fast Magneto-acoustic Wave magnitude:       'msfs'

                Slow Magneto-acoustic Wave along x1 axis:   'mssx1'
                Slow Magneto-acoustic Wave along x2 axis:   'mssx2'
                Slow Magneto-acoustic Wave along x3 axis:   'mssx3'
                Slow Magneto-acoustic Wave magnitude:       'msss'

                Fast MS Wave Mach number: 'fmach'
                Alfvén Wave Mach number:  'amach'
                Slow MS Wave Mach number: 'smach'

                Plasma Beta:    'beta'

                Magnetic Pressure:      'b_prs'  
            """
            print(text)
        ### Magnetic Fields ###
        elif data2plot == 'bx1':
            variable_name = 'Magnetic Field in x1 direction'
            if log == True:
                data = np.log(self.bx1)
            else:
                data = self.bx1
        elif data2plot == 'bx2':
            variable_name = 'Magnetic Field in x2 direction'
            if log == True:
                data = np.log(self.bx2)
            else:
                data = self.bx2
        elif data2plot == 'bx3':
            variable_name = 'Magnetic Field in x3 direction'
            if log == True:
                data = np.log(self.bx3)
            else:
                data = self.bx3
        elif data2plot == 'bxs':
            variable_name = 'Magnetic Field magnitude'
            if log == True:
                data = np.log(self.magnetic_field_magnitude)
            else:
                data = self.magnetic_field_magnitude
        ### Velocities ###
        elif data2plot == 'vx1':
            variable_name = 'Velocity in x1 direction'
            if log == True:
                data = np.log(self.vx1)
            else:
                data = self.vx1
        elif data2plot == 'vx2':
            variable_name = 'Velocity in x2 direction'
            if log == True:
                data = np.log(self.vx2)
            else:
                data = self.vx2
        elif data2plot == 'vx3':
            variable_name = 'Velocity in x3 direction'
            if log == True:
                data = np.log(self.vx3)
            else:
                data = self.vx3
        elif data2plot == 'vxs':
            variable_name = 'Velocity Magnitude'
            if log == True:
                data = np.log(self.velocity_magnitude)
            else:
                data = self.velocity_magnitude
        ### Pressure, Density, Tracer and GLM ###
        elif data2plot == 'prs':
            variable_name = 'Pressure'
            if log == True:
                data = np.log(self.prs)
            else:
                data = self.prs
        elif data2plot == 'rho':
            variable_name = 'Density'
            if log == True:
                data = np.log(self.rho)
            else:
                data = self.rho
        elif data2plot == 'tr1':
            variable_name = 'Density weighted tracer'
            if log == True:
                data = np.log(self.tr1)
            else:
                data = self.tr1
        elif data2plot == 'glm':
            variable_name = 'General Lagrangian Multiplier'
            if log == True:
                data = np.log(self.glm)
            else:
                data = self.glm
        ### Alfvén Velocities ###
        elif data2plot == 'avx1':
            variable_name = 'Alfvén velocity in x1 direction'
            if log == True:
                data = np.log(self.avx1)
            else:
                data = self.avx1
        elif data2plot == 'avx2':
            variable_name = 'Alfvén velocity in x2 direction'
            if log == True:
                data = np.log(self.avx2)
            else:
                data = self.avx2
        elif data2plot == 'avx3':
            variable_name = 'Alfvén velocity in x3 direction'
            if log == True:
                data = np.log(self.avx3)
            else:
                data = self.avx3
        elif data2plot == 'avxs':
            variable_name = 'Alfvén velocity Magnitude'
            if log == True:
                data = np.log(self.alfvén_velocity_magnitude)
            else:
                data = self.alfvén_velocity_magnitude
        ### Magneto-acoustic wave velocities ###
        ### Fast ###
        elif data2plot == 'msfx1':
            variable_name = 'Fast Magneto-acoustic Wave velocity in x1 direction'
            if log == True:
                data = np.log(self.fast_m)
            else:
                data = self.fast_ms_x1
        elif data2plot == 'msfx2':
            variable_name = 'Fast Magneto-acoustic Wave velocity in x2 direction'
            if log == True:
                data = np.log(self.fast_m)
            else:
                data = self.fast_ms_x2
        elif data2plot == 'msfx3':
            variable_name = 'Fast Magneto-acoustic Wave velocity in x3 direction'
            if log == True:
                data = np.log(self.fast_m)
            else:
                data = self.fast_ms_x3
        elif data2plot == 'msfs':
            variable_name = 'Fast Magneto-acoustic Wave velocity magnitude'
            if log == True:
                data = np.log(self.fast_m)
            else:
                data = self.fast_ms_velocity_magnitude
        ### Slow ###
        elif data2plot == 'mssx1':
            variable_name = 'Slow Magneto-acoustic Wave velocity in x1 direction'
            if log == True:
                data = np.log(self.slow_m)
            else:
                data = self.slow_ms_x1
        elif data2plot == 'mssx2':
            variable_name = 'Slow Magneto-acoustic Wave velocity in x2 direction'
            if log == True:
                data = np.log(self.slow_m)
            else:
                data = self.slow_ms_x2
        elif data2plot == 'mssx3':
            variable_name = 'Slow Magneto-acoustic Wave velocity in x3 direction'
            if log == True:
                data = np.log(self.slow_m)
            else:
                data = self.slow_ms_x3
        elif data2plot == 'msss':
            variable_name = 'Slow Magneto-acoustic Wave velocity magnitude'
            if log == True:
                data = np.log(self.slow_m)
            else:
                data = self.slow_ms_velocity_magnitude
        ### Mach numbers ###
        elif data2plot == 'fmach':
            variable_name = 'Fast wave Mach number'
            if log == True:
                data = np.log(self.mach_f)
            else:
                data = self.mach_fast
        elif data2plot == 'amach':
            variable_name = 'Alfvén wave Mach number'
            if log == True:
                data = np.log(self.mach_a)
            else:
                data = self.mach_alfvén
        elif data2plot == 'smach':
            variable_name = 'Slow wave Mach number'
            if log == True:
                data = np.log(self.mach_s)
            else:
                data = self.mach_slow
        ### Plasma Beta ###
        elif data2plot == 'beta':
            variable_name = 'Plasma Beta'
            if log == True:
                data = np.log(self.beta)
            else:
                data = self.beta
        ### Magnetic pressure ###
        elif data2plot == 'b_prs':
            variable_name = 'Magnetic pressure'
            if log == True:
                data = np.log(self.magnetic_prs)
            else:
                data = self.magnetic_prs


        self.data = data

        prefix = ''
        if log == True:
            prefix = 'Log of'
        self.title = f'{prefix} {variable_name}'

    def plot(self, data2plot=None, log=False, close=False, save=False):
        """
        This method plots the simulated data sets output by PLUTO, contained within the h5 file
        while additionally also plots the wave velocities, machnumbers and other calculated values.
        Focuses on spatial distributon of the data.
        """
        self._data(data2plot=data2plot, log=log, close=close, save=save)
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        
        divider = mal(axes)
        cax = divider.append_axes('right',size='5%',pad=0.25)
        pl = axes.contourf(self.axial_grid, self.radial_grid, self.data, cmap=self.cmap, levels=128, alpha=0.95)
        plt.colorbar(pl,cax,ticks=np.linspace(np.min(self.data),np.max(self.data), 9))
        axes.set_title(self.title)
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
        axes.set_ylabel(r'Radial distnace [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            folder = title.replace(' ', '_')
            check_dir = f'{self.data_path}plot/{folder}'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}plot/{folder}/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()
    
    def hist(self,  data2plot=None, data2log=False, close=False, save=False, bins=None, log=False):
        """
        Method to plot histogram of the data which is passed in as the argument
        """
        self._data(data2plot=data2plot, log=log, close=close, save=save)
        title = self.title
        shape = self.data.shape
        x_shape = shape[0]*shape[1]
        new_shape = (x_shape, 1)
        data_plot = np.reshape(self.data, new_shape)
        plt.rc('font', size=8)
        fig, axes = plt.subplots(1,1,figsize=self.image_size, dpi=self.dpi)
        axes.set_title(f'Histogram data for {title} at {self.time_step} {self.simulation_title}')
        hist = axes.hist(data_plot, bins=bins, align='mid', edgecolor='white')
        axes.set_xlabel('Value')
        axes.set_ylabel('Frequency')
        cols = axes.patches
        labels = [f'{int(x)/x_shape*100:.2f}%' for x in hist[0]]
        
        for col, label in zip(cols, labels):
            height = col.get_height()
            axes.text(col.get_x() + col.get_width() / 2, height+0.01, label, ha='center', va='bottom')
    
        if log==True:
            plt.semilogy()
        
        if close==True:
            plt.close()

        if save==True:
            folder = title.replace(' ', '_')
            check_dir = f'{self.data_path}hist/{folder}'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}hist/{folder}/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()
        
    def shocks(self, plot_shock=True, save=False, close=False):
        """
        method to plot MHD shocks
        """
        #### Super fast MS to Super Alfvénic ####
        
        zero_array = np.zeros_like(self.mach_fast)
        fast_shock_ax = []
        fast_shock_ra = []
        for j, row in enumerate(zero_array):
            for i, val in enumerate(row):
                if (i+1 == len(row)) or (j+1 == len(zero_array)):
                    pass
                else:
                    if (self.mach_fast[j,i] > 1) and (self.mach_fast[j,i+1] < 1) and (self.mach_alfvén[j,i+1] > 1):
                        fast_shock_ax.append(self.axial_grid[i])
                        fast_shock_ra.append(self.radial_grid[j])
                    if (self.mach_fast[j,i] > 1) and (self.mach_fast[j+1,i] < 1) and (self.mach_alfvén[j+1,i] > 1):
                        fast_shock_ax.append(self.axial_grid[i])
                        fast_shock_ra.append(self.radial_grid[j])
                    if (self.mach_fast[j,i] > 1) and (self.mach_fast[j+1,i+1] < 1) and (self.mach_alfvén[j+1,i+1] > 1):
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
                    ### Super Fast to Sub Alfvénic, Super Slow (1-3)
                    # perpendicular shocks
                    if (self.mach_fast[j,i] > 1) and (self.mach_alfvén[j,i+1] < 1) and (self.mach_slow[j,i+1] > 1):
                        inter_shock_ax1.append(self.axial_grid[i])
                        inter_shock_ra1.append(self.radial_grid[j])
                    # parallel shocks
                    if (self.mach_fast[j,i] > 1) and (self.mach_alfvén[j+1,i] < 1) and (self.mach_slow[j+1,i] > 1):
                        inter_shock_ax1.append(self.axial_grid[i])
                        inter_shock_ra1.append(self.radial_grid[j])
                    # oblique shocks
                    if (self.mach_fast[j,i] > 1) and (self.mach_alfvén[j+1,i+1] < 1) and (self.mach_slow[j+1,i+1] > 1):
                        inter_shock_ax1.append(self.axial_grid[i])
                        inter_shock_ra1.append(self.radial_grid[j])
                    
                    ## Sub Fast, Super Alfvénic to Sub Alfvénic, Super Slow (2-3)
                    # perpendicular shocks
                    if (self.mach_alfvén[j,i] > 1) and (self.mach_fast[j,i] < 1) and (self.mach_alfvén[j,i+1] < 1) and (self.mach_slow[j,i+1] > 1):
                        inter_shock_ax2.append(self.axial_grid[i])
                        inter_shock_ra2.append(self.radial_grid[j])
                    # parallel shocks
                    if (self.mach_alfvén[j,i] > 1) and (self.mach_fast[j,i] < 1) and (self.mach_alfvén[j+1,i] < 1) and (self.mach_slow[j+1,i] > 1):
                        inter_shock_ax2.append(self.axial_grid[i])
                        inter_shock_ra2.append(self.radial_grid[j])
                    # oblique shocks
                    if (self.mach_alfvén[j,i] > 1) and (self.mach_fast[j,i] < 1) and (self.mach_alfvén[j+1,i+1] < 1) and (self.mach_slow[j+1,i+1] > 1):
                        inter_shock_ax2.append(self.axial_grid[i])
                        inter_shock_ra2.append(self.radial_grid[j])
                   
                    ### Sub Fast, Super Alfvénic to Sub Slow (2-4)
                    # perpendicular shocks
                    if (self.mach_alfvén[j,i] > 1) and (self.mach_fast[j,i] < 1) and (self.mach_slow[j,i+1] < 1):
                        inter_shock_ax3.append(self.axial_grid[i])
                        inter_shock_ra3.append(self.radial_grid[j])
                    # parallel shocks
                    if (self.mach_alfvén[j,i] > 1) and (self.mach_fast[j,i] < 1) and (self.mach_slow[j+1,i] < 1):
                        inter_shock_ax3.append(self.axial_grid[i])
                        inter_shock_ra3.append(self.radial_grid[j])
                    # oblique shocks
                    if (self.mach_alfvén[j,i] > 1) and (self.mach_fast[j,i] < 1) and (self.mach_slow[j+1,i+1] < 1):
                        inter_shock_ax3.append(self.axial_grid[i])
                        inter_shock_ra3.append(self.radial_grid[j])

                    ### Hydrodynamic (1-4)
                    # perpendicular shocks
                    if (self.mach_fast[j,i] > 1) and (self.mach_slow[j,i+1] < 1):
                        inter_shock_ax4.append(self.axial_grid[i])
                        inter_shock_ra4.append(self.radial_grid[j])
                    # parallel shocks
                    if (self.mach_fast[j,i] > 1) and (self.mach_slow[j+1,i] < 1):
                        inter_shock_ax4.append(self.axial_grid[i])
                        inter_shock_ra4.append(self.radial_grid[j])
                    # oblique shocks
                    if (self.mach_fast[j,i] > 1) and (self.mach_slow[j+1,i+1] < 1):
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
                    if (self.mach_slow[j,i] > 1) and (self.mach_alfvén[j,i+1] < 1) and (self.mach_slow[j,i+1] < 1):
                        slow_shock_ax.append(self.axial_grid[i])
                        slow_shock_ra.append(self.radial_grid[j])
                    if (self.mach_slow[j,i] > 1) and (self.mach_alfvén[j+1,i] < 1) and (self.mach_slow[j+1,i] < 1):
                        slow_shock_ax.append(self.axial_grid[i])
                        slow_shock_ra.append(self.radial_grid[j])
                    if (self.mach_slow[j,i] > 1) and (self.mach_alfvén[j+1,i+1] < 1) and (self.mach_slow[j+1,i+1] < 1):
                        slow_shock_ax.append(self.axial_grid[i])
                        slow_shock_ra.append(self.radial_grid[j])
        
        ### Plots ###

        if plot_shock != None:
            figureS, axesS = plt.subplots(figsize=(self.image_size[0]*1.25, self.image_size[1]*1.25), dpi=self.dpi*2)
            if plot_shock == True:
                axesS.plot(fast_shock_ax, fast_shock_ra, '^', lw=0.25, color='red', markersize=3.5, label=f'Fast 1-2 Shocks ({len(set(fast_shock_ax))})', alpha=0.5)
                axesS.plot(inter_shock_ax1, inter_shock_ra1, 's', lw=0.25, color='magenta', markersize=3.5, label=f'Inter 1-3 Shocks ({len(set(inter_shock_ax1))})', alpha=0.5)
                axesS.plot(inter_shock_ax2, inter_shock_ra2, 'v', lw=0.25, color='green', markersize=3.5, label=f'Inter 2-3 Shocks ({len(set(inter_shock_ax2))})', alpha=0.5)
                axesS.plot(inter_shock_ax3, inter_shock_ra3, 'H', lw=0.25, color='orange', markersize=3.5, label=f'Inter 2-4 Shocks ({len(set(inter_shock_ax3))})', alpha=0.5)
                axesS.plot(inter_shock_ax4, inter_shock_ra4, 'D', lw=0.25, color='cyan', markersize=3.5, label=f'Hydro 1-4 Shocks ({len(set(inter_shock_ax4))})', alpha=0.5)
                axesS.plot(slow_shock_ax, slow_shock_ra, '+', lw=0.25, color='blue', markersize=3.5, label=f'Slow 3-4 Shocks ({len(set(slow_shock_ax))})', alpha=0.5)

            elif plot_shock == 'slow':
                axesS.plot(slow_shock_ax, slow_shock_ra, '+', lw=0.25, color='blue', markersize=3.5, label=f'Slow 3-4 Shocks ({len(set(slow_shock_ax))})', alpha=0.5)
            elif plot_shock == 'inter':
                axesS.plot(inter_shock_ax1, inter_shock_ra1, 's', lw=0.25, color='magenta', markersize=3.5, label=f'Inter 1-3 Shocks ({len(set(inter_shock_ax1))})', alpha=0.5)
                axesS.plot(inter_shock_ax2, inter_shock_ra2, 'v', lw=0.25, color='green', markersize=3.5, label=f'Inter 2-3 Shocks ({len(set(inter_shock_ax2))})', alpha=0.5)
                axesS.plot(inter_shock_ax3, inter_shock_ra3, 'H', lw=0.25, color='orange', markersize=3.5, label=f'Inter 2-4 Shocks ({len(set(inter_shock_ax3))})', alpha=0.5)
                axesS.plot(inter_shock_ax4, inter_shock_ra4, 'D', lw=0.25, color='cyan', markersize=3.5, label=f'Hydro 1-4 Shocks ({len(set(inter_shock_ax4))})', alpha=0.5)
            elif plot_shock == 'fast':
                axesS.plot(fast_shock_ax, fast_shock_ra, '^', lw=0.25, color='red', markersize=3.5, label=f'Fast 1-2 Shocks ({len(set(fast_shock_ax))})', alpha=0.5)

            axesS.legend()
            axesS.set_title(f'MHD Shocks from plasma-state transition at {self.time_step}')
            axesS.set_xlim(self.xlim[0], self.xlim[1])
            axesS.set_ylim(self.ylim[0], self.ylim[1])
            axesS.set_ylabel(r'Radial distance [$R_{jet}$]')
            axesS.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            folder = title.replace(' ', '_')
            check_dir = f'{self.data_path}shock/{folder}'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}shock/{folder}/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

    def plot_spacetime(self, data2plot=None, begin=0, end=-1, radial_step=0, log=False, close=False, save=False):
        """
        Space-time diagram plotting method
        """
        ### Deal with mirrored images ###
        check_mirr = self.mirrored
        self.mirrored = False
        
        ### Begin loop through dataset ###
        self.list2plot = []
        time_list = range(len(self.data_list))
        
        print(f'Begin space-time diagram - {time.ctime()}')
        if begin != 0:
            print(f'Started at time-step {begin}!')
        for index, val in enumerate(self.data_list[begin:end]):
            index += begin
            self.calculate_data(index)
            self._data(data2plot=data2plot, log=log, close=close, save=save)
            self.list2plot.append(self.data[radial_step])
            print(f'{index}/{len(self.data_list)-1} files loaded', end='\r')

        print(f'Done! {time.ctime()}')

        
        X, Y = np.meshgrid(self.axial_grid, range(begin, begin+len(self.list2plot)))
        figure, axes = plt.subplots(figsize=(self.image_size[1], self.image_size[1]), dpi=self.dpi)
        pl = axes.contourf(X, Y, self.list2plot, cmap='hot', levels=128)
        
        figure.colorbar(pl, 
            location='right', 
            shrink=0.95, 
            aspect=20,
            pad=0.02, 
            label='Magnetic Field magnitude', 
            format='%.2f'
            )
        
        if end == -1:
            end_tick = len(self.data_list)
        else:
            end_tick = end

        ### Set tick spacing ###
        if end_tick - begin >= 500:
            spacing = 100
        elif end_tick - begin >= 100:
            spacing = 50
        elif end_tick - begin >= 50:
            spacing = 10

        if (begin == 0) == False and (end == -1) == False:
            axes.set_yticks(np.arange(begin,end_tick,spacing))
    
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylabel(r'Time [$time-step$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')

        

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}spacetime/{data2plot}'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}spacetime/{data2plot}/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()