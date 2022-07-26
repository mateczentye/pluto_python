#%%
import pytest
import h5py
import os
import numpy as np
import sys
sys.path.insert(0,'../')
from pluto_python.pluto import py3Pluto

path = os.getcwd()+'/tests/test_files/'

test_object = py3Pluto(
    data_path=path,
    time_step=0,
    dpi=300,
    image_size=(20,10),
    cmap='seismic',
    global_limits=False,
    xlim=(0,30),
    ylim=(0,15),
    mirrored=False,
)

@pytest.mark.input
def test_path():
    with pytest.raises(TypeError):
        test_object = py3Pluto(
            data_path=None,
            time_step=0,
            cmap='seismic',
            global_limits=False,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=False,
        )

@pytest.mark.input
def test_timestep():
    with pytest.raises(TypeError):
        test_object = py3Pluto(
            data_path=path,
            time_step=10.1,
            cmap='seismic',
            global_limits=False,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=False,
        )

@pytest.mark.input
def test_dpi():
    with pytest.raises(TypeError):
        test_object = py3Pluto(
            data_path=path,
            time_step=0,
            dpi='str',
            cmap='seismic',
            global_limits=False,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=False,
        )

@pytest.mark.input
def test_image_size_type():
    with pytest.raises(TypeError):
        test_object = py3Pluto(
            data_path=path,
            time_step=0,
            dpi=300,
            image_size=0,
            cmap='seismic',
            global_limits=False,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=False,
        )

@pytest.mark.input
def test_image_size_len():
    with pytest.raises(ValueError):
        test_object = py3Pluto(
            data_path=path,
            time_step=0,
            dpi=300,
            image_size=(0,0,0),
            cmap='seismic',
            global_limits=False,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=False,
        )

@pytest.mark.input
def test_xlim():
    with pytest.raises(ValueError):
        test_object = py3Pluto(
            data_path=path,
            time_step=0,
            dpi=300,
            image_size=(20,10),
            cmap='seismic',
            global_limits=False,
            xlim='(0,30)',
            ylim=(0,15),
            mirrored=False,
        )

@pytest.mark.input
def test_ylim():
    with pytest.raises(ValueError):
        test_object = py3Pluto(
            data_path=path,
            time_step=0,
            dpi=300,
            image_size=(20,10),
            cmap='seismic',
            global_limits=False,
            xlim='(0,30)',
            ylim=(0,15),
            mirrored=False,
        )

@pytest.mark.input
def test_glob_lims():
    with pytest.raises(TypeError):
        test_object = py3Pluto(
            data_path=path,
            time_step=0,
            dpi=300,
            image_size=(20,10),
            cmap='seismic',
            global_limits=50,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=False,
        )

@pytest.mark.input
def test_mirror():
    with pytest.raises(TypeError):
        test_object = py3Pluto(
            data_path=path,
            time_step=0,
            dpi=300,
            image_size=(20,10),
            cmap='seismic',
            global_limits=False,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=50,
        )

@pytest.mark.input
def test_gamma():
    with pytest.raises(TypeError):
        test_object = py3Pluto(
            data_path=path,
            time_step=0,
            dpi=300,
            image_size=(20,10),
            cmap='seismic',
            global_limits=False,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=False,
            gamma='string'
        )

@pytest.mark.util
def test_classifier():
    assert type(test_object.classifier(0)) == h5py._hl.group.Group

@pytest.mark.util
def test_reader():
    assert type(test_object.data_list) == list

@pytest.mark.util
def test_calculate_data():
    test_object.calculate_data(0)
    assert np.shape(test_object.vx1) == (300,600)

    