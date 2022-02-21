#%%
import pytest
import h5py
import os
import numpy as np
import sys
sys.path.insert(0,'..')
from pluto_python.mhd_jet import mhd_jet as mj

path = os.getcwd()+'/tests/test_files/'
test_object = mj(
    data_path=path,
    time_step=0,
    cmap='seismic',
    global_limits=False,
    xlim=(0,30),
    ylim=(0,15),
    mirrored=False,
)
@pytest.mark.util
def test_init_dat():
    assert test_object.data == None

@pytest.mark.util    
def test_init_fig():
    assert test_object.figure == None

@pytest.mark.util
def test_plot():
    test_object.plot('vx1')
    assert type(test_object.data) == np.ndarray

@pytest.mark.util
def test_test_hist():
    test_object.hist('vx1')
    assert type(test_object.data) == np.ndarray

@pytest.mark.util
def test_shocks():
    test_object.shocks()
    assert type(test_object.shocks_list) == list

@pytest.mark.util
def test_spacetime():
    test_object.plot_spacetime('prs')
    assert type(test_object.list2plot) == list

@pytest.mark.util
def test_power():
    test_object.plot_power()
    assert type(test_object.list_power) == list

@pytest.mark.util
def test_energy():
    test_object.plot_energy()
    assert type(test_object.list_energy) == list

@pytest.mark.util
def test_E_dens():
    test_object.plot_energy_density()
    assert type(test_object.list_E_dens) == list

@pytest.mark.util
def test_streamlines():
    test_object.plot_fieldlines()
    assert type(test_object.streamline_check) == np.ndarray