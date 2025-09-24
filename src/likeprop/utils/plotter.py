'''This module defines some basic plotting functions and classes'''
import itertools
from typing import Optional, Any
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot
import matplotlib.tri
import matplotlib.colors
import numpy as np
from likeprop.utils.typing import TimeStateVector, SliceT, ArrayLikeT


def get_vector_field(x_discretization: np.ndarray,
                     y_discretization: np.ndarray,
                     ode_dynamics: TimeStateVector,
                     projection: SliceT = slice(0, 2, None),
                     active_indices: ArrayLikeT = np.array([0, 1]),
                     dim_of_state: int = 2,
                     default_state: Optional[np.ndarray] = None) -> tuple:
    r"""returns the vector field (\dot{x}, \dot{y}) at the 
    point (x,y)

    Args:
        x_discretization (np.ndarray): descretization of the x coordinate of the state
        y_discretization (np.ndarray): descretization of the y coordinate of the state
        ode_dynamics (TimeStateVector): OdeModel.F
        projection (SliceT, optional): the coordinates to project on. Defaults to slice(0,2,None).
        active_indices (ArrayLikeT, optional): indices for x and y coordinates of the state. 
        Defaults to [0,1].
        dim_of_state (int, optional): dimension of the state. Defaults to 2.
        default_state (np.ndarray): the default state vector for holding values that are not the
        x and y coordinates

    Returns:
        tuple: (grid_in_x, grid_in_y, \dot{x}, \dot{y})
    """

    # create a state vector with zero defaults. By default if the dynamics are in dimension > 2 then
    # other coordinates are set to 0.

    if default_state is None:
        default_state = np.zeros(dim_of_state)
    # create a meshgrid object
    x_grid, y_grid = np.meshgrid(x_discretization, y_discretization)

    # store the return type of the dynamics in the meshgrid
    u_vec = np.zeros(x_grid.shape)
    v_vec = np.zeros(x_grid.shape)

    for i, j in itertools.product(range(x_grid.shape[0]), range(x_grid.shape[1])):
        xcoord = x_grid[i, j]
        ycoord = y_grid[i, j]
        # my_ode.F(time, state, params)
        default_state[active_indices] = [xcoord, ycoord]
        vec = ode_dynamics(0, default_state, None)
        # project the vector to the plane of x and y
        vec = vec[projection]
        # store the components of vec in meshgrid
        u_vec[i, j] = vec[0]
        v_vec[i, j] = vec[1]

    # return the vector field
    return (x_grid, y_grid, u_vec, v_vec)


def plot_contour(triangulization: matplotlib.tri.Triangulation,
                 density_points: ArrayLikeT,
                 xlim: Optional[tuple] = None,
                 ylim: Optional[tuple] = None,
                 stream_lines: Optional[tuple] = None,
                 lower_threshold: Optional[float] = None,
                 upper_threshold: Optional[float] = None,
                 xticks: Optional[ArrayLikeT] = None,
                 yticks: Optional[ArrayLikeT] = None,
                 cmap: str = 'coolwarm',
                 streamline_kwargs:Optional[dict[str, Any]]=None,
                 streamline_alpha:float = 0.3
                 ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """generates contour plot of the probability density on a plane

    Args:
        triangulization (matplotlib.tri.Triangulation): a matplotlib Triangulization of the plane
        density_points (ArrayLikeT): points of pdf
        xlim (Optional[tuple], optional): range of x coordinate. Defaults to None.
        ylim (Optional[tuple], optional): range of y coordinate. Defaults to None.
        stream_lines (Optional[tuple], optional): streamline tuple on this plane. Defaults to None.
        lower_threshold (Optional[float], optional): lower cutoff for density. Defaults to None.
        upper_threshold (Optional[float], optional): upper cutoff for density. Defaults to None.
        xticks (Optional[ArrayLikeT], optional): array of x tick. Defaults to None.
        yticks (Optional[ArrayLikeT], optional): array of y tick. Defaults to None.
        cmap (str, optional): matplotlib colormap string. Defaults to 'coolwarm'.

    Returns:
        matplotlib.figure.Figure: Figure with the contour plot
    """

    # set the lower and upper values
    if lower_threshold is None:
        lower_threshold = np.floor(np.log10(np.min(density_points)) - 1)
    else:
        lower_threshold = np.log10(lower_threshold)
    if upper_threshold is None:
        upper_threshold = np.ceil(np.log10(np.max(density_points)) + 1)
    else:
        upper_threshold = np.log10(upper_threshold)

    # create the levels in log10 scale
    lev_exp = np.arange(lower_threshold, upper_threshold)
    levs = np.power(10, lev_exp)

    fig, axes = matplotlib.pyplot.subplots()
    # plot stream plot with this level
    if stream_lines is not None:
        streamlines = axes.streamplot(*stream_lines, **streamline_kwargs)
        streamlines.lines.set_alpha(streamline_alpha)

    # get the contour
    contourf = axes.tricontourf(triangulization,
                                density_points,
                                levs,
                                norm=matplotlib.colors.LogNorm(),
                                cmap=cmap)
    # make equal aspect ratio
    axes.set_aspect(float(np.diff(axes.get_xlim())/np.diff(axes.get_ylim())))

    # limit the plots to a good box
    if xlim is not None and ylim is not None:
        axes.set(xlim=xlim, ylim=ylim)
    elif xlim is not None:
        axes.set(xlim=xlim)
    elif ylim is not None:
        axes.set(ylim=ylim)
    # control the x,y ticks
    if xticks is not None:
        axes.set_xticks(xticks)
    if yticks is not None:
        axes.set_yticks(yticks)

    # set colorbar
    fig.colorbar(contourf, location='bottom')

    return (fig,axes)
