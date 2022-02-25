__author__ = 'Justin Panchula'
__copyright__ = 'Copyright 2022'
__credits__ = 'Justin Panchula'
__license__ = 'N/A'
__version__ = '1.1.0'
__status__ = 'Dev'
__doc__ = """This file is used to store all the methods necessary to generate any aircraft from an STL file and simulate its movement."""

# Imports
from matplotlib.animation import FuncAnimation
import numpy as np
import control as ctrl
from stl import mesh

# Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits import mplot3d as mpl
from matplotlib.widgets import Slider

# Typing imports
from typing_extensions import Self
from typing import Tuple

class _Framing():
    def _body2inertial(phi: float, theta: float, psi: float) -> np.ndarray:
        R = np.array([[np.cos(theta) * np.cos(psi), np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), np.cos(phi) *
                       np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
                      [np.cos(theta) * np.sin(psi), np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), np.cos(phi) *
                       np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
                      [-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]])
        return R

    def _body2stability(alpha: float) -> np.ndarray:
        R = np.array([[np.cos(alpha, 0, np.sin(alpha))],
                      [0, 1, 0],
                      [-np.sin(alpha), 0, np.cos(alpha)]])
        return R

    def _stability2wind(beta: float) -> np.ndarray:
        R = np.array([[np.cos(beta), np.sin(beta), 0],
                      [-np.sin(beta), np.cos(beta), 0],
                      [0, 0, 1]])
        return R

class _environment():
    def _wind_gust(phi: float, theta: float, psi: float, Va: float, dt: float) -> np.ndarray:
        return

class Plotting():
    def generate_sliders(fig: Figure) -> np.ndarray:
        """Generates aircraft control sliders on the provided figure.

        Args:
            fig (Figure): The figure to add sliders to.

        Returns:
            np.ndarray: The sliders as an ordered numpy array.
        """
        sliders = np.empty(6, dtype=Slider)

        # Fx
        sliders[0] = Slider(
            ax=fig.add_axes([0.05, 0.25, 0.0225, 0.63]),  # Left, Bottom, Width, Height
            label='Fx',
            valmin=-50,
            valmax=50,
            valstep=1,
            valinit=0,
            orientation='vertical'
        )
        # Fy
        sliders[1] = Slider(
            ax=fig.add_axes([0.11, 0.25, 0.0225, 0.63]),
            label='Fy',
            valmin=-50,
            valmax=50,
            valstep=1,
            valinit=0,
            orientation='vertical'
        )
        # Fz
        sliders[2] = Slider(
            ax=fig.add_axes([0.17, 0.25, 0.0225, 0.63]),
            label='Fz',
            valmin=-50,
            valmax=50,
            valstep=1,
            valinit=0,
            orientation='vertical'
        )
        # Mx
        sliders[3] = Slider(
            ax=fig.add_axes([0.25, 0.08, 0.65, 0.03]),
            label='Mx',
            valmin=-50,
            valmax=50,
            valstep=1,
            valinit=0
        )
        # My
        sliders[4] = Slider(
            ax=fig.add_axes([0.25, 0.05, 0.65, 0.03]),
            label='My',
            valmin=-50,
            valmax=50,
            valstep=1,
            valinit=0
        )
        # Mz
        sliders[5] = Slider(
            ax=fig.add_axes([0.25, 0.02, 0.65, 0.03]),
            label='Mz',
            valmin=-50,
            valmax=50,
            valstep=1,
            valinit=0
        )
        return sliders

class UAV():
    def __init__(self, meshFile: str, mass: float = 25.0, Jx: float = 0.8244, Jy: float = 1.135, Jz: float = 1.759, Jxz: float = 0.1204) -> None:
        """Instantiates the object and sets values.

        Args:
            meshFile (str): The .stl file of the UAV.
            mass (float): The mass of the UAV. Defaults to 25 kg.
            Jx (float): The moment of inertia about the x-axis. Defaults to 0.8244 __
            Jy (float): The moment of inertia about the y-axis. Defaults to 1.1350 __
            Jz (float): The moment of inertia about the z-axis. Defaults to 1.7590 __
            Jxz (float): The product of inertia about the x-, and z-axis: this is the coupling between roll and yaw. Defaults to 0.1204
        """
        # Set mesh
        self.Mesh = mesh.Mesh.from_file(meshFile)

        # Set positional values
        self.north = np.array([0.0], float)
        self._north = 0.0
        self.east = np.array([0.0], float)
        self._east = 0.0
        self.down = np.array([0.0], float)
        self._down = 0.0
        self.roll = np.array([0.0], float)
        self._roll = 0.0
        self.pitch = np.array([0.0], float)
        self._pitch = 0.0
        self.yaw = np.array([0.0], float)
        self._yaw = 0.0

        # Set property values
        self.mass = mass
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.Jxz = Jxz

    def _print_coordinates(self: Self) -> None:
        """Print the coordinates of the self object.

        Args:
            self (Self): Self object.
        """
        print('\n---------Coordinates---------')
        print('North: {} meters'.format(self.north.sum()))
        print('East: {} meters'.format(self.east.sum()))
        print('Down: {} meters'.format(self.down.sum()))
        print('Pitch: {:.4f} radians'.format(self.pitch.sum()))
        print('Roll: {:.4f} radians'.format(self.roll.sum()))
        print('Yaw: {:.4f} radians'.format(self.yaw.sum()))
        print('-----------------------------')

    def plot(self: Self, title: str, scaleFactor: float = 1/6) -> Tuple[Figure, Axes]:
        """Plots a mesh
        Args:
            self (Self): The UAV.
            title (string): The name of the plot.
            scaleFactor (float, optional): The scale factor of the object resolved in meteres. Defaults to 1/6.

        Returns:
            Tuple[Figure, Axes]: The figure and axes generated.
        """
        # Define the figure
        fig = plt.figure(figsize=(1.6*6, 0.9*6))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        # Add mesh to plot
        collection = mpl.art3d.Poly3DCollection(self.Mesh.vectors * scaleFactor, edgecolor='black', linewidth=0.2)
        ax.add_collection3d(collection)

        # Auto scale to mesh size
        scale = self.Mesh.points.flatten() * scaleFactor
        ax.auto_scale_xyz(scale, scale, scale)

        # Format the plot
        ax.set_title(title)

        return fig, ax

    def update_uav(self: Self, sliders: np.ndarray) -> None:
        """Updates the self paramters based on the slider changes

        Args:
            self (Self): Self object.
            sliders (np.ndarray): The numpy array of sliders.
        """
        def _update_sliders(val) -> None:
            """Private function used to satisfy "Slider.on_changed()" method.

            Args:
                val (private): Inherent to "Slider.on_changed()" method.
            """
            self._north = sliders[0].val
            self._east = sliders[1].val
            self._down = sliders[2].val
            self._pitch = sliders[3].val
            self._roll = sliders[4].val
            self._yaw = sliders[5].val
            self._print_coordinates()

        sliders[0].on_changed(_update_sliders)
        sliders[1].on_changed(_update_sliders)
        sliders[2].on_changed(_update_sliders)
        sliders[3].on_changed(_update_sliders)
        sliders[4].on_changed(_update_sliders)
        sliders[5].on_changed(_update_sliders)

if __name__ == '__main__':
    meshFile = 'F117.stl'
    title = 'F117 Nighthawk (1:1)'
    uav = UAV(meshFile)
    fig, ax = uav.plot(title)
    sliders = Plotting.generate_sliders(fig)
    uav.update_uav(sliders)
    anim = FuncAnimation(fig, Plotting.update_plot, frames=60, blit=False, fargs=[uav, ax, title])
    plt.show()
    anim.save