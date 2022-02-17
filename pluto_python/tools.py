#%%
import numpy as np

def rearrange_axial_grid(initial_list, reference_list):
    return_list = []
    if type(initial_list) != list and type(initial_list) != tuple and type(initial_list) != np.ndarray:
        raise ValueError('Initial list must be a list or a tuple')
    elif type(reference_list) != list and type(reference_list) != tuple and type(initial_list) != np.ndarray:
        raise ValueError('Reference list must be a list or a tuple')
    else:
        previous_slice = 0
        for sub_list in reference_list:
            subl_len = len(sub_list)
            return_list.append(np.asanyarray(initial_list[previous_slice:previous_slice + subl_len]))

    return return_list

def data(self, data2plot=None, log=False, close=False, save=False):
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
