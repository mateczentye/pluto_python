#%%
import pytest
import h5py
import os
import numpy as np
import sys
sys.path.insert(0,'..')
import pluto_python.calculator as c

@pytest.mark.phys
def test_magnetic_field():
    assert c.magnetic_field(1,1) == pytest.approx(1.783655867)

@pytest.mark.phys
def test_ms_speed():
    assert c.magnetosonic_speed(0.1, 2.0, 1.783655867) == pytest.approx(7.198213843)

@pytest.mark.phys
def test_alfven_velocity():
    assert c.alfven_velocity(1.783655867, 1) == 1.783655867

@pytest.mark.phys
def test_magneto_acoustic_velocity_unity():
    assert c.magneto_acoustic_velocity(1, 1, 1, 1) == [pytest.approx(1), pytest.approx(1)]

@pytest.mark.phys
def test_magneto_acoustic_velocity():
    assert c.magneto_acoustic_velocity(2, 1, 1, 1) == [1, 2]

@pytest.mark.phys
def test_magneto_acoustic_velocity_res():
    assert c.magneto_acoustic_velocity(1.78, 0.6, 1, 5/3) == [1, 1.78]

@pytest.mark.phys
def test_machnumber():
    assert c.mach_number(16.45860550417676, c.magnetosonic_speed(0.1,2.0, c.magnetic_field(1.5,1.5))) == 2

@pytest.mark.phys
def test_energy_density():
    assert c.energy_density(0.6, 1, 16.4586, 2.1845, 5/3) == [pytest.approx(0.9), pytest.approx(135.442757), pytest.approx(2.38602)]

@pytest.mark.phys
def test_magnetic_pressure():
    assert c.magnetic_pressure(5) == 12.5

@pytest.mark.phys
def test_soundspeed():
    assert c.sound_speed(5/3, 0.6, 1) == 1