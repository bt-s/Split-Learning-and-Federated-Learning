#!/usr/bin/python3

"""plotting.py Contains some plotting functions

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def generate_simple_plot(x: List, y: List, title: str="", x_label: str="",
        y_label: str="", y_lim: List[float]=[0.0, 1.0], save: bool=True,
        fname: str=""):
    """Create a simple plot

    Args:
        x: x-coordinates
        y: y-coordinates
        titel: Plot title
        x_label, y_label: Plot labels
        y_lim: Y-axis limits
        save: Whether to save the plot
        fname: Name for saving
    """
    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel=x_label, ylabel=y_label, ylim=y_lim, title=title)

    if save:
        fig.savefig("../plots/" + fname)
    else:
        plt.show()
