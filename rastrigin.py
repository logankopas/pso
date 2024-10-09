""" 
Rastrigin Test function for optimization tests
Details can be found here: https://en.wikipedia.org/wiki/Rastrigin_function
Expected global minimum is 0 at (0,0) 
"""

from typing import Union
import math
import plot
import numpy.typing as npt
import numpy as np


@np.vectorize
def rastrigin_2d(x: Union[float, npt.NDArray], y: Union[float, npt.NDArray], a: int = 10
                 ) -> Union[float, npt.NDArray]:
    """Rastigin function for testing optimization algorithms

    Rastigin is a non-convex, multimodal function perfect for testing optimization algorithms 
    in regards to local minima. This function only supports the 2D version of Rastigin.

    Parameters
    ----------
    x : float
        The X value, must be in [-5.12, 5.12]
    y : float
        The y value, must be in [-5.12, 5.12]
    a : int, optional
        The A parameter is a constant that modifies the Rastigin function.

    Returns
    -------
        The value of the Rastigin function at (x, y)

    Raises
    ------
    AssertionError
        An assertion error is raised if (x,y) fall outside of the bounds of the function

    """
    # Function restrictions
    assert -5.12 <= x <= 5.12
    assert -5.12 <= y <= 5.12

    _sum = sum((_i**2) - a * math.cos(2 * math.pi * _i) for _i in (x, y))
    return a * 2 + _sum


def almost_equals(a: float, b: float, epsilon: float = 0.01) -> bool:
    """ Basic float equality function
    """
    return abs(b - a) < epsilon


if __name__ == "__main__":
    # Development testing
    assert almost_equals(rastrigin_2d(0, 0), 0)
    assert almost_equals(rastrigin_2d(0.5, 1), 21.25)
    try:
        rastrigin_2d(-5.13, 0)
    except AssertionError:
        pass
    else:
        raise Exception("Assertion Failed")

    plot.plot_function(rastrigin_2d, (-5.12, 5.12), (-5.12, 5.12), 0.12)
