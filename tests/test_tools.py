#%%
import pytest
import h5py
import os
import numpy as np
import sys
sys.path.insert(0,'..')
from pluto_python.tools import rearrange_axial_grid as rag

@pytest.mark.util
def test_rag():
    oneD = [1]*10
    twoD = [oneD for x in range(5)]

    print(np.shape(twoD))
    test = [
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1],
        [1,1,1],
        [1,1,1],
        [1,1,1],
        [1,1,1],
        [1,1,1,1,1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1,1,1,1,1,1,1,1]
    ]

    assert np.shape(rag(twoD, test)) == np.shape(test)