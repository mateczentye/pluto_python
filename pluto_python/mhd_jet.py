#%%
from .pluto import py3Pluto
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import os

class mhd_jet(py3Pluto):
    """
    This class is a sub-class of py3Pluto, which stores all information from the simulations
    and claculations, which can be plotted using the plot or hist methods within this class.
    """
    def __init__(self,
        data_path,
        time_step = 0,
        dpi = 300,
        image_size = (10,5),
        ylim = None,
        xlim = None,
        cmap = 'bwr',
        global_limits = False,
        mirrored = False,
        gamma = 5/3,
        title=''
    ):

        super().__init__(
            data_path = data_path,
            time_step = time_step,
            dpi = dpi,
            image_size = image_size,
            ylim = ylim,
            xlim = xlim,
            cmap = cmap,
            global_limits = global_limits,
            mirrored = mirrored,
            gamma = gamma
        )
        self.data = None
        self.figure = None
        self.simulation_title = title

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
            raise StopIteration('Please give a variable to plot!')
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
            variable_name = 'Alfvén speed'
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
            variable_name = 'Fast Magneto-acoustic speed'
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
            variable_name = 'Slow Magneto-acoustic speed'
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
        ### Sound Speed ###
        elif data2plot == 'cs':
            variable_name = 'Ideal Sound Speed'
            if log == True:
                data = np.log(self.sound_speed)
            else:
                data = self.sound_speed
        


        self.data = data
        self.variable_name = variable_name

        prefix = ''
        if log == True:
            prefix = 'Log of'
        self.title = f'{prefix} {variable_name}'

    def plot(self, data2plot=None, log=False, close=False, save=False, title=''):
        """
        This method plots the simulated data sets output by PLUTO, contained within the h5 file
        while additionally also plots the wave velocities, machnumbers and other calculated values.
        Focuses on spatial distributon of the data.
        """
        self._data(data2plot=data2plot, log=log, close=close, save=save)
        figure, axes = plt.subplots(figsize=self.image_size, dpi=self.dpi)
        plt.tight_layout()
        divider = mal(axes)
        cax = divider.append_axes('right',size='5%',pad=0.25)
        pl = axes.contourf(self.axial_grid, self.radial_grid, self.data, cmap=self.cmap, levels=128, alpha=0.95)
        plt.colorbar(pl,cax,ticks=np.linspace(np.min(self.data),np.max(self.data), 9))
        #axes.set_title(self.title)
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylim(self.ylim[0], self.ylim[1])
        axes.set_ylabel(r'Radial distnace [$R_{jet}$]')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')
        

        if close==True:
            plt.close()

        if save==True:
            #title = self.data_path.split('/')[-2]
            folder = title.replace(' ', '_')
            check_dir = f'{self.data_path}plot'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir, 755)
                chck_subdir = check_dir + f'/{data2plot}'
                if os.path.exists(chck_subdir) is False:
                    os.mkdir(chck_subdir, 755)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}plot/{data2plot}/{self.time_step}_{data2plot}_{title}.jpeg', bbox_inches='tight', pad_inches=0.5)
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
        axes.set_title(f'Histogram data for {title} at {self.time_step}')
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
            title = self.data_path.split('/')[-2]
            folder = title.replace(' ', '_')
            check_dir = f'{self.data_path}hist'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir, 755)
                chck_subdir = check_dir + f'/{data2plot}'
                if os.path.exists(chck_subdir) is False:
                    os.mkdir(chck_subdir, 755)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}hist/{data2plot}/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()
        
    def shocks(self, plot_shock='12, 13, 14, 23, 24, 34', save=False, close=False):
        """
        method to plot MHD shocks
        """
        self.plot_shock = plot_shock
        #### Super fast MS to Super Alfvénic ####
        zero_array = np.zeros_like(self.mach_fast)
        self.fast_shock_ax = []
        self.fast_shock_ra = []
        for j, row in enumerate(zero_array):
            for i, val in enumerate(row):
                if (i+1 == len(row)) or (j+1 == len(zero_array)):
                    pass
                else:
                    if (self.mach_fast[j,i] > 1) and (self.mach_fast[j,i+1] < 1) and (self.mach_alfvén[j,i+1] > 1):
                        self.fast_shock_ax.append(self.axial_grid[i])
                        self.fast_shock_ra.append(self.radial_grid[j])
                    if (self.mach_fast[j,i] > 1) and (self.mach_fast[j+1,i] < 1) and (self.mach_alfvén[j+1,i] > 1):
                        self.fast_shock_ax.append(self.axial_grid[i])
                        self.fast_shock_ra.append(self.radial_grid[j])
                    if (self.mach_fast[j,i] > 1) and (self.mach_fast[j+1,i+1] < 1) and (self.mach_alfvén[j+1,i+1] > 1):
                        self.fast_shock_ax.append(self.axial_grid[i])
                        self.fast_shock_ra.append(self.radial_grid[j])
                
        #### Intermed. Shocks #####
        self.inter_shock_ax1 = []
        self.inter_shock_ra1 = []
        self.inter_shock_ax2 = []
        self.inter_shock_ra2 = []
        self.inter_shock_ax3 = []
        self.inter_shock_ra3 = []
        self.inter_shock_ax4 = []
        self.inter_shock_ra4 = []
        
        for j, row in enumerate(zero_array):
            for i, val in enumerate(row):
                if (i+1 == len(row)) or (j+1 == len(zero_array)):
                    pass
                else:
                    ### Super Fast to Sub Alfvénic, Super Slow (1-3)
                    # perpendicular shocks
                    if (self.mach_fast[j,i] > 1) and (self.mach_alfvén[j,i+1] < 1) and (self.mach_slow[j,i+1] > 1):
                        self.inter_shock_ax1.append(self.axial_grid[i])
                        self.inter_shock_ra1.append(self.radial_grid[j])
                    # parallel shocks
                    if (self.mach_fast[j,i] > 1) and (self.mach_alfvén[j+1,i] < 1) and (self.mach_slow[j+1,i] > 1):
                        self.inter_shock_ax1.append(self.axial_grid[i])
                        self.inter_shock_ra1.append(self.radial_grid[j])
                    # oblique shocks
                    if (self.mach_fast[j,i] > 1) and (self.mach_alfvén[j+1,i+1] < 1) and (self.mach_slow[j+1,i+1] > 1):
                        self.inter_shock_ax1.append(self.axial_grid[i])
                        self.inter_shock_ra1.append(self.radial_grid[j])
                    
                    ## Sub Fast, Super Alfvénic to Sub Alfvénic, Super Slow (2-3)
                    # perpendicular shocks
                    if (self.mach_alfvén[j,i] > 1) and (self.mach_fast[j,i] < 1) and (self.mach_alfvén[j,i+1] < 1) and (self.mach_slow[j,i+1] > 1):
                        self.inter_shock_ax2.append(self.axial_grid[i])
                        self.inter_shock_ra2.append(self.radial_grid[j])
                    # parallel shocks
                    if (self.mach_alfvén[j,i] > 1) and (self.mach_fast[j,i] < 1) and (self.mach_alfvén[j+1,i] < 1) and (self.mach_slow[j+1,i] > 1):
                        self.inter_shock_ax2.append(self.axial_grid[i])
                        self.inter_shock_ra2.append(self.radial_grid[j])
                    # oblique shocks
                    if (self.mach_alfvén[j,i] > 1) and (self.mach_fast[j,i] < 1) and (self.mach_alfvén[j+1,i+1] < 1) and (self.mach_slow[j+1,i+1] > 1):
                        self.inter_shock_ax2.append(self.axial_grid[i])
                        self.inter_shock_ra2.append(self.radial_grid[j])
                   
                    ### Sub Fast, Super Alfvénic to Sub Slow (2-4)
                    # perpendicular shocks
                    if (self.mach_alfvén[j,i] > 1) and (self.mach_fast[j,i] < 1) and (self.mach_slow[j,i+1] < 1):
                        self.inter_shock_ax3.append(self.axial_grid[i])
                        self.inter_shock_ra3.append(self.radial_grid[j])
                    # parallel shocks
                    if (self.mach_alfvén[j,i] > 1) and (self.mach_fast[j,i] < 1) and (self.mach_slow[j+1,i] < 1):
                        self.inter_shock_ax3.append(self.axial_grid[i])
                        self.inter_shock_ra3.append(self.radial_grid[j])
                    # oblique shocks
                    if (self.mach_alfvén[j,i] > 1) and (self.mach_fast[j,i] < 1) and (self.mach_slow[j+1,i+1] < 1):
                        self.inter_shock_ax3.append(self.axial_grid[i])
                        self.inter_shock_ra3.append(self.radial_grid[j])

                    ### Hydrodynamic (1-4)
                    # perpendicular shocks
                    if (self.mach_fast[j,i] > 1) and (self.mach_slow[j,i+1] < 1):
                        self.inter_shock_ax4.append(self.axial_grid[i])
                        self.inter_shock_ra4.append(self.radial_grid[j])
                    # parallel shocks
                    if (self.mach_fast[j,i] > 1) and (self.mach_slow[j+1,i] < 1):
                        self.inter_shock_ax4.append(self.axial_grid[i])
                        self.inter_shock_ra4.append(self.radial_grid[j])
                    # oblique shocks
                    if (self.mach_fast[j,i] > 1) and (self.mach_slow[j+1,i+1] < 1):
                        self.inter_shock_ax4.append(self.axial_grid[i])
                        self.inter_shock_ra4.append(self.radial_grid[j])

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
        
        ### Check array for unit tests
        self.shocks_list = [
            [slow_shock_ax, slow_shock_ra],
            [self.fast_shock_ax, self.fast_shock_ra],
            [self.inter_shock_ax1,self.inter_shock_ra1],
            [self.inter_shock_ax2, self.inter_shock_ra2],
            [self.inter_shock_ax3, self.inter_shock_ra3],
            [self.inter_shock_ax4, self.inter_shock_ra4]
        ]

        ### Plots ###
        figureS, axesS = plt.subplots(figsize=(self.image_size[0]*1.25, self.image_size[1]*1.25), dpi=self.dpi*2)
        
        if 'slow' in self.plot_shock:
            axesS.plot(slow_shock_ax, slow_shock_ra, '+', lw=0.25, color='blue', markersize=3.5, label=f'Slow 3-4 Shocks ({len((slow_shock_ax))})', alpha=0.5)
        if 'inter' in self.plot_shock:
            axesS.plot(self.inter_shock_ax1, self.inter_shock_ra1, 's', lw=0.25, color='magenta', markersize=3.5, label=f'Inter 1-3 Shocks ({len((self.inter_shock_ax1))})', alpha=0.5)
            axesS.plot(self.inter_shock_ax2, self.inter_shock_ra2, 'v', lw=0.25, color='green', markersize=3.5, label=f'Inter 2-3 Shocks ({len((self.inter_shock_ax2))})', alpha=0.5)
            axesS.plot(self.inter_shock_ax3, self.inter_shock_ra3, 'H', lw=0.25, color='orange', markersize=3.5, label=f'Inter 2-4 Shocks ({len((self.inter_shock_ax3))})', alpha=0.5)
            axesS.plot(self.inter_shock_ax4, self.inter_shock_ra4, 'D', lw=0.25, color='cyan', markersize=3.5, label=f'Hydro 1-4 Shocks ({len((self.inter_shock_ax4))})', alpha=0.5)
        if 'fast' in self.plot_shock:
            axesS.plot(self.fast_shock_ax, self.fast_shock_ra, '^', lw=0.25, color='red', markersize=3.5, label=f'Fast 1-2 Shocks ({len((self.fast_shock_ax))})', alpha=0.5)

        if '12' in self.plot_shock:
            axesS.plot(self.fast_shock_ax, self.fast_shock_ra, '^', lw=0.25, color='red', markersize=3.5, label=f'Fast 1-2 Shocks ({len((self.fast_shock_ax))})', alpha=0.5)
        
        if '13' in self.plot_shock:
            axesS.plot(self.inter_shock_ax1, self.inter_shock_ra1, 's', lw=0.25, color='magenta', markersize=3.5, label=f'Inter 1-3 Shocks ({len((self.inter_shock_ax1))})', alpha=0.5)
        
        if '14' in self.plot_shock:
            axesS.plot(self.inter_shock_ax4, self.inter_shock_ra4, 'D', lw=0.25, color='cyan', markersize=3.5, label=f'Hydro 1-4 Shocks ({len((self.inter_shock_ax4))})', alpha=0.5)

        if '23' in self.plot_shock:
            axesS.plot(self.inter_shock_ax2, self.inter_shock_ra2, 'v', lw=0.25, color='green', markersize=3.5, label=f'Inter 2-3 Shocks ({len((self.inter_shock_ax2))})', alpha=0.5)

        if '24' in self.plot_shock:
            axesS.plot(self.inter_shock_ax3, self.inter_shock_ra3, 'H', lw=0.25, color='orange', markersize=3.5, label=f'Inter 2-4 Shocks ({len((self.inter_shock_ax3))})', alpha=0.5)

        if '34' in self.plot_shock:
            axesS.plot(slow_shock_ax, slow_shock_ra, '+', lw=0.25, color='blue', markersize=3.5, label=f'Slow 3-4 Shocks ({len((slow_shock_ax))})', alpha=0.5)
        

        axesS.legend()
        axesS.set_title(f'MHD Shocks from plasma-state transition at {self.time_step}')
        axesS.set_xlim(self.xlim[0], self.xlim[1])
        axesS.set_ylim(self.ylim[0], self.ylim[1])
        axesS.set_ylabel(r'Radial distance [$R_{jet}$]')
        axesS.set_xlabel(r'Axial distance [$R_{jet}$]')

        if close==True:
            plt.close()

        if save==True:
            title = self.data_path.split('/')[-2]
            folder = title.replace(' ', '_')
            check_dir = f'{self.data_path}shocks'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir, 755)
                chck_subdir = check_dir + f'/{self.plot_shock}'
                if os.path.exists(chck_subdir) is False:
                    os.mkdir(chck_subdir, 755)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}shocks/{self.plot_shock}/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
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
            label=f'{self.variable_name} magnitude', 
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
            title = self.data_path.split('/')[-2]
            folder = title.replace(' ', '_')
            check_dir = f'{self.data_path}space_time'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir, 755)
                chck_subdir = check_dir + f'/{data2plot}'
                if os.path.exists(chck_subdir) is False:
                    os.mkdir(chck_subdir, 755)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}space_time/{data2plot}/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

        if check_mirr == True:
            self.mirrored = True

    def plot_power(self, save=False, close=False):
        """
        Plots the power curves for the jet
        """
        self.calculate_data(self.time_step)
        total_sys = [np.sum(x) for x in np.transpose(self.total_power_sys)]
        total_jet = [np.sum(x) for x in np.transpose(self.total_power_jet)]
        kinetic_jet = [np.sum(x) for x in np.transpose(self.kinetic_power_jet)]
        enthalpy_jet = [np.sum(x) for x in np.transpose(self.thermal_power_jet)]
        magnetic_jet = [np.sum(x) for x in np.transpose(self.magnetic_power_jet)]

        self.list_power = [
            total_sys,
            total_jet,
            kinetic_jet,
            enthalpy_jet,
            magnetic_jet
            ]

        figure, axes = plt.subplots(figsize=(self.image_size[0], self.image_size[1]), dpi=self.dpi)
        #plt0 = axes.plot(self.axial_grid, total_sys, '-', color='black', ms=2.5, label='Total System Power')
        plt1 = axes.plot(self.axial_grid, total_jet, '-', color='blue', ms=2.5, label='Total Jet Power')
        plt2 = axes.plot(self.axial_grid, kinetic_jet, '-.', color='green', ms=2.5, label='Kinetic Jet Power')
        plt3 = axes.plot(self.axial_grid, enthalpy_jet, ':', color='orange', ms=1.5, label='Thermal Jet Power')
        plt4 = axes.plot(self.axial_grid, magnetic_jet, '--', color='red', ms=1.5, label='Magnetic Jet Power')
        axes.set_title(f'Power at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} of {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylabel(r'Power')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')
        axes.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize='small', markerscale=2)

        if close==True:
            plt.close()

        if save==True:
            title = self.data_path.split('/')[-2]
            folder = title.replace(' ', '_')
            check_dir = f'{self.data_path}power'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir, 755)
                chck_subdir = check_dir
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}power/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()
        


    def plot_energy(self, save=False, close=False):
        """
        Plots the energy curves for the jet
        """
        self.calculate_data(self.time_step)
        total_sys = [np.sum(x) for x in np.transpose(self.total_energy_sys)]
        total_jet = [np.sum(x) for x in np.transpose(self.total_energy_jet)]
        kinetic_jet = [np.sum(x) for x in np.transpose(self.kinetic_energy_jet)]
        enthalpy_jet = [np.sum(x) for x in np.transpose(self.thermal_energy_jet)]
        magnetic_jet = [np.sum(x) for x in np.transpose(self.magnetic_energy_jet)]

        self.list_energy = [
            total_sys,
            total_jet,
            kinetic_jet,
            enthalpy_jet,
            magnetic_jet,
        ]

        figure, axes = plt.subplots(figsize=(self.image_size[0], self.image_size[1]), dpi=self.dpi)
        #plt0 = axes.plot(self.axial_grid, total_sys, '-', color='black', ms=2.5, label='Total System Energy')
        plt1 = axes.plot(self.axial_grid, total_jet, '-', color='blue', ms=2.5, label='Total Jet Energy')
        plt2 = axes.plot(self.axial_grid, kinetic_jet, '-.', color='green', ms=2.5, label='Kinetic Jet Energy')
        plt3 = axes.plot(self.axial_grid, enthalpy_jet, ':', color='orange', ms=1.5, label='Thermal Jet Energy')
        plt4 = axes.plot(self.axial_grid, magnetic_jet, '--', color='red', ms=1.5, label='Magnetic Jet Energy')
        axes.set_title(f'Energy at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} of {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylabel(r'Energy')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')
        axes.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize='small', markerscale=2)

        if close==True:
            plt.close()

        if save==True:
            title = self.data_path.split('/')[-2]
            folder = title.replace(' ', '_')
            check_dir = f'{self.data_path}energy'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir, 755)
                chck_subdir = check_dir
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}energy/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

    def plot_energy_density(self, save=False, close=False):
        """
        Plots the energy density curves for the jet
        """
        self.calculate_data(self.time_step)
        total_jet = [np.sum(x) for x in np.transpose(self.total_energy_density)]
        kinetic_jet = [np.sum(x) for x in np.transpose(self.kinetic_energy_density)]
        enthalpy_jet = [np.sum(x) for x in np.transpose(self.thermal_energy_density)]
        magnetic_jet = [np.sum(x) for x in np.transpose(self.magnetic_energy_density)]

        self.list_E_dens = [
            total_jet,
            kinetic_jet,
            enthalpy_jet,
            magnetic_jet,
        ]

        figure, axes = plt.subplots(figsize=(self.image_size[0], self.image_size[1]), dpi=self.dpi)
        plt1 = axes.plot(self.axial_grid, total_jet, '-', color='blue', ms=2.5, label='Total System Energy Density')
        plt2 = axes.plot(self.axial_grid, kinetic_jet, '-.', color='green', ms=2.5, label='Kinetic System Energy Density')
        plt3 = axes.plot(self.axial_grid, enthalpy_jet, ':', color='orange', ms=1.5, label='Thermal System Energy Density')
        plt4 = axes.plot(self.axial_grid, magnetic_jet, '--', color='red', ms=1.5, label='Magnetic System Energy Density')
        axes.set_title(f'Energy density at time = {self.tstop * int(self.timestep.replace("Timestep_", "")) / 1001 :.1f} of {self.simulation_title}')
        axes.set_xlim(self.xlim[0], self.xlim[1])
        axes.set_ylabel(r'Energy density')
        axes.set_xlabel(r'Axial distance [$R_{jet}$]')
        axes.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize='small', markerscale=2)

        if close==True:
            plt.close()

        if save==True:
            title = self.data_path.split('/')[-2]
            folder = title.replace(' ', '_')
            check_dir = f'{self.data_path}energy_density'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir, 755)
                chck_subdir = check_dir
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}energy_density/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()

    def plot_fieldlines(self, save=False, close=False, levels=128, min_bxs=None, max_bxs=None, min_bx2=None, max_bx2=None):
        """
        Plots a vector plot of the magnetic field lines in the axial-radial plane,
        while plots the true magnitude of the magnetic field accounted in all 3 directions.
        """
        b_mag = self.magnetic_field_magnitude
        cmp='jet'
        scmap = 'seismic'
        density=4

        subgrid_x_low = float(self.ini_content['[Grid]']['X3-grid']['Subgrids Data'][0][0])
        subgrid_y_low = float(self.ini_content['[Grid]']['X1-grid']['Subgrids Data'][0][0])
        subgrid_x = float(self.ini_content['[Grid]']['X3-grid']['Subgrids Data'][1][0])
        subgrid_y = float(self.ini_content['[Grid]']['X1-grid']['Subgrids Data'][1][0])
        subgrid_x_res = int(self.ini_content['[Grid]']['X3-grid']['Subgrids Data'][0][1])
        subgrid_y_res = int(self.ini_content['[Grid]']['X1-grid']['Subgrids Data'][0][1])

        subgrid_y_start = 0
        subgrid_y_end = subgrid_y_res

        x1_comp = np.asarray(self.bx1)[subgrid_y_start:subgrid_y_end,:subgrid_x_res]
        x2_comp = np.asarray(self.bx2)[subgrid_y_start:subgrid_y_end,:subgrid_x_res]
        x3_comp = np.asarray(self.bx3)[subgrid_y_start:subgrid_y_end,:subgrid_x_res]
        magnitude = np.asarray(b_mag)[subgrid_y_start:subgrid_y_end,:subgrid_x_res]

        X, Y = np.meshgrid(self.axial_grid[:subgrid_x_res],
                            self.radial_grid[subgrid_y_start:subgrid_y_end])

        X2, Y2 = np.meshgrid(np.linspace(subgrid_x_low, subgrid_x, subgrid_x_res), 
                                 np.linspace(subgrid_y_low, subgrid_y, subgrid_y_res))

        ### Colourbar scaling

        if min_bxs == None:
            min_bxs = 0
        if max_bxs == None:
            max_bxs = np.max(magnitude)
        if min_bx2 == None:
            min_bx2 = np.min(x2_comp)
        if max_bxs == None:
            max_bx2 = np.max(x2_comp)

        figure, axes = plt.subplots(1,1,figsize=(self.image_size[0], self.image_size[0]),dpi=self.dpi)
        plt.tight_layout()
        norm = matplotlib.colors.Normalize(vmin=min_bxs, vmax=max_bxs)
        linenorm = matplotlib.colors.Normalize(vmin=min_bx2, vmax=max_bx2)
        
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
                        norm=linenorm,
                        )
        
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmp), ax=axes, format='%.2f', label='Magnetic Field Strength',  location='bottom', pad=0.01), 
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=linenorm, cmap=scmap), ax=axes, format='%.2f', label='Magnetic Field in direction x2',  location='bottom', pad=0.05)
        
        ### set up limits
        #if subgrid_x_low > self.xlim[0]

        plt.xlim(self.xlim[0], self.xlim[1])
        #plt.title(f'Magnetic field with field line direction at {self.time_step} {self.simulation_title}')
        
        if self.mirrored == False:
            plt.ylim(self.ylim[0], self.ylim[1])
        else:
            plt.ylim(-self.ylim[1], self.ylim[1])

        if close==True:
            plt.close()

        if save==True:
            check_dir = f'{self.data_path}field_line'
            if os.path.exists(check_dir) is False:
                os.mkdir(check_dir)
            bbox = matplotlib.transforms.Bbox([[0,0], [12,9]])
            plt.savefig(f'{self.data_path}field_line/{self.time_step}.jpeg', bbox_inches='tight', pad_inches=0.5)
            plt.close()
