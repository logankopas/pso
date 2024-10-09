"""
Various plotting functions for display and debugging
"""

import typing
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from cycler import cycler
import imageio


import numpy as np


FILE_DIR = 'tmp'


def plot_function(
        func: typing.Callable[[float, float], float],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        tick_distance: float,
        additional_points: typing.Optional[list] = None,
        filename: typing.Optional[str] = None
) -> None:
    """Plots the provided function as a 3d surface across the provided range
    """
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    # Set up grid
    x_min, x_max = x_range
    y_min, y_max = y_range
    X = np.arange(x_min, x_max, tick_distance)
    Y = np.arange(y_min, y_max, tick_distance)
    X, Y = np.meshgrid(X, Y)

    # Get function values in numpy format
    Z = np.array(func(X, Y))

    surface = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, alpha=0.5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.view_init(75, 105)

    # Add a color bar which maps values to colors.
    fig.colorbar(surface, shrink=0.5, aspect=5)

    if additional_points is not None:
        _add_points(ax, additional_points)


    if not os.path.exists(FILE_DIR):
        os.makedirs(FILE_DIR)
    
    image_path = os.path.join(FILE_DIR, filename or "out.png")
    plt.savefig(image_path)
    plt.close()
    

def _add_points(ax: plt.Axes, points: list[tuple[float, float, float]]):
    # Each point should get a unique (and consistent) colour
    colour_map = plt.get_cmap('gist_rainbow')
    colours=[colour_map(1.0*i/len(points)) for i in range(len(points))]

    X, Y, Z = zip(*points)
    ax.scatter(X, Y, Z, c=colours)

def create_animation():
    """Creates an animation from all plots"""
    files = sorted(os.listdir(FILE_DIR))
    with imageio.get_writer(os.path.join(FILE_DIR, 'out.gif'), 
                            mode='i', duration=0.2) as writer:
        for file in files:
            image = imageio.imread(os.path.join(FILE_DIR, file))
            writer.append_data(image)
    writer.close()
