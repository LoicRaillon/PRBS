import pylfsr
from pylfsr import LFSR
import numpy as np
import itertools


def prbs(Tmax, Tmin, initstate='random'):
    '''Pseudo Random Binary signal (PRBS)

    Args:
        Tmax: maximum number of samples in one state
        Tmin: minimum number of samples in one state
        initstate: initial state of the linear feedback state register
            'ones': binary numpy array of ones, dimension = (length register,)
            'random': random binary numpy array, dimension = (length register,)

    Returns:
        PRBS as a numpy array

    Notes:
        The Linear Feedback Shift Register (LFSR) can be installed from PyPi: https://pypi.org/project/pylfsr/
        or from the source:  https://github.com/Nikeshbajaj/Linear_Feedback_Shift_Register
    '''
    if not isinstance(Tmax, int):
        raise TypeError('`Tmax` must be an integer')

    if Tmax < 2:
        raise ValueError('`Tmax` must be > 2')

    if not isinstance(Tmin, int):
        raise TypeError('`Tmax` must be an integer')

    if Tmin < 1:
        raise ValueError('`Tmin` must be > 1')

    if Tmin >= Tmax:
        raise ValueError('`Tmax` must be strictly superior to `Tmin`')

    __init_availabble__ = ['random', 'ones']
    if initstate not in __init_availabble__:
        raise ValueError(f'`initstate` must be either {__init_availabble__}')

    # get the register length
    n = np.ceil(Tmax / Tmin)
    if n < 2 or n > 31:
        raise ValueError('The PRBS cannot be generated, '
                         'decompose the signal in two sequences')

    # Linear feedback register up to 32 bits
    fpoly = {
        2: [2, 1],
        3: [3, 1],
        4: [4, 1],
        5: [5, 2],
        6: [6, 1],
        7: [7, 1],
        8: [8, 4, 3, 2],
        9: [9, 4],
        10: [10, 3],
        11: [11, 2],
        12: [12, 6, 4, 1],
        13: [13, 4, 3, 1],
        14: [14, 8, 6, 1],
        15: [15, 1],
        16: [16, 12, 3, 1],
        17: [17, 3],
        18: [18, 7],
        19: [19, 5, 2, 1],
        20: [20, 3],
        21: [21, 2],
        22: [22, 1],
        23: [23, 5],
        24: [24, 7, 2, 1],
        25: [25, 3],
        26: [26, 6, 2, 1],
        27: [27, 5, 2, 1],
        28: [28, 3],
        29: [29, 2],
        30: [30, 23, 2, 1],
        31: [31, 3]
    }

    L = LFSR(fpoly=fpoly[n], initstate=initstate, verbose=False)

    seq = []
    for n in range(L.expectedPeriod):
        L.next()
        seq.append(L.state[0])

    seq_padded = np.repeat(seq, Tmin)

    # check generated PRBS
    assert seq_padded.shape[0] == L.expectedPeriod * Tmin
    assert max(len(list(v)) for g, v in itertools.groupby(seq_padded)) == Tmax
    assert min(len(list(v)) for g, v in itertools.groupby(seq_padded)) == Tmin

    return seq_padded
