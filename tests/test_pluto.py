#%%
import pytest
import sys
sys.path.insert(0,'..')
from pluto_python.pluto import py3Pluto


test_object_mir = py3Pluto(
    data_path='test_files/',
    time_step=0,
    cmap='seismic',
    global_limits=False,
    xlim=(0,30),
    ylim=(0,15),
    mirrored=True,
)

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

def test_timestep():
    with pytest.raises(TypeError):
        test_object = py3Pluto(
            data_path='test_files',
            time_step=10.1,
            cmap='seismic',
            global_limits=False,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=False,
        )

def test_ini_path():
    with pytest.raises(IsADirectoryError):
        test_object = py3Pluto(
            data_path='test_files/',
            ini_path='test_files/test_inis2',
            time_step=0,
            cmap='seismic',
            global_limits=False,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=False,
        )

def test_ini_count():
    with pytest.raises(FileExistsError):
        test_object = py3Pluto(
            data_path='test_files/',
            ini_path='test_files/test_inis',
            time_step=0,
            cmap='seismic',
            global_limits=False,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=False,
        )

def test_dpi():
    with pytest.raises(TypeError):
        test_object = py3Pluto(
            data_path='test_files/',
            time_step=0,
            dpi='str',
            cmap='seismic',
            global_limits=False,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=False,
        )

def test_image_size_type():
    with pytest.raises(TypeError):
        test_object = py3Pluto(
            data_path='test_files/',
            time_step=0,
            dpi=300,
            image_size=0,
            cmap='seismic',
            global_limits=False,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=False,
        )

def test_image_size_len():
    with pytest.raises(ValueError):
        test_object = py3Pluto(
            data_path='test_files/',
            time_step=0,
            dpi=300,
            image_size=(0,0,0),
            cmap='seismic',
            global_limits=False,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=False,
        )

def test_xlim():
    with pytest.raises(ValueError):
        test_object = py3Pluto(
            data_path='test_files/',
            time_step=0,
            dpi=300,
            image_size=(20,10),
            cmap='seismic',
            global_limits=False,
            xlim='(0,30)',
            ylim=(0,15),
            mirrored=False,
        )

def test_ylim():
    with pytest.raises(ValueError):
        test_object = py3Pluto(
            data_path='test_files/',
            time_step=0,
            dpi=300,
            image_size=(20,10),
            cmap='seismic',
            global_limits=False,
            xlim='(0,30)',
            ylim=(0,15),
            mirrored=False,
        )

def test_glob_lims():
    with pytest.raises(TypeError):
        test_object = py3Pluto(
            data_path='test_files/',
            time_step=0,
            dpi=300,
            image_size=(20,10),
            cmap='seismic',
            global_limits=50,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=False,
        )

def test_mirror():
    with pytest.raises(TypeError):
        test_object = py3Pluto(
            data_path='test_files/',
            time_step=0,
            dpi=300,
            image_size=(20,10),
            cmap='seismic',
            global_limits=False,
            xlim=(0,30),
            ylim=(0,15),
            mirrored=50,
        )

def test_gamma():
    with pytest.raises(TypeError):
        test_object = py3Pluto(
            data_path='test_files/',
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
