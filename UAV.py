__author__ = 'Justin Panchula'
__copyright__ = 'Copyright 2022'
__credits__ = 'Justin Panchula'
__license__ = 'N/A'
__version__ = '1.1.0'
__status__ = 'Dev'
__doc__ = """This file is used to store all the methods necessary to generate any aircraft from an STL file and simulate its movement."""

# Imports
import numpy as np
from stl import mesh

# Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits import mplot3d as mpl
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

# Typing imports
from typing_extensions import Self
from typing import Tuple
from collections.abc import Callable

class _Framing():
    def _body2inertial(phi: float, theta: float, psi: float) -> np.ndarray:
        R = np.array([[np.cos(theta) * np.cos(psi), np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), np.cos(phi) *
                       np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
                      [np.cos(theta) * np.sin(psi), np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), np.cos(phi) *
                       np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
                      [-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]])
        return R

class Plotting():
    def generate_sliders(fig: Figure) -> np.ndarray:
        """Generates aircraft control sliders on the provided figure.

        Args:
            fig (Figure): The figure to add sliders to.

        Returns:
            np.ndarray: The sliders as an ordered numpy array.
        """
        sliders = np.empty(6, dtype=Slider)

        # North
        sliders[0] = Slider(
            ax=fig.add_axes([0.05, 0.25, 0.0225, 0.63]),  # Left, Bottom, Width, Height
            label='North',
            valmin=-1,
            valmax=1,
            valstep=0.01,
            valinit=0,
            orientation='vertical'
        )
        # East
        sliders[1] = Slider(
            ax=fig.add_axes([0.11, 0.25, 0.0225, 0.63]),
            label='East',
            valmin=-1,
            valmax=1,
            valstep=0.01,
            valinit=0,
            orientation='vertical'
        )
        # Down
        sliders[2] = Slider(
            ax=fig.add_axes([0.17, 0.25, 0.0225, 0.63]),
            label='Down',
            valmin=-1,
            valmax=1,
            valstep=0.01,
            valinit=0,
            orientation='vertical'
        )
        # Pitch
        sliders[3] = Slider(
            ax=fig.add_axes([0.25, 0.08, 0.65, 0.03]),
            label='Pitch',
            valmin=np.radians(-90),
            valmax=np.radians(90),
            valstep=np.radians(1),
            valinit=0
        )
        # Roll
        sliders[4] = Slider(
            ax=fig.add_axes([0.25, 0.05, 0.65, 0.03]),
            label='Roll',
            valmin=np.radians(-90),
            valmax=np.radians(90),
            valstep=np.radians(1),
            valinit=0
        )
        # Yaw
        sliders[5] = Slider(
            ax=fig.add_axes([0.25, 0.02, 0.65, 0.03]),
            label='Yaw',
            valmin=np.radians(-90),
            valmax=np.radians(90),
            valstep=np.radians(1),
            valinit=0
        )
        return sliders

    def update_plot(t: float, uav: object, planeAx: Axes, title: str, scaleFactor: float = 1/6) -> None:
        uav.theMesh.y -= uav._north * 50
        uav.north = np.append(uav.north, uav._north)
        uav.theMesh.x += uav._east * 50
        uav.east = np.append(uav.east, uav._east)
        uav.theMesh.z -= uav._down * 50
        uav.down = np.append(uav.down, uav._down)
        uav.theMesh.rotate([0.5, 0.0, 0.0], -uav._pitch)
        uav.pitch = np.append(uav.pitch, uav._pitch)
        uav.theMesh.rotate([0.0, 0.5, 0.0], -uav._roll)
        uav.roll = np.append(uav.roll, uav._roll)
        uav.theMesh.rotate([0.0, 0.0, 0.5], uav._yaw)
        uav.yaw = np.append(uav.yaw, uav._yaw)

        # Clear the axis
        planeAx.clear()

        # Re-add collection
        collection = mpl.art3d.Poly3DCollection(uav.theMesh.vectors * scaleFactor, edgecolor='black', linewidth=0.2)
        planeAx.add_collection3d(collection)

        # Auto scale to mesh size
        scale = uav.theMesh.points.flatten() * scaleFactor
        planeAx.auto_scale_xyz(scale, scale, scale)

        # Format the plot
        planeAx.set_title(title)
        return planeAx

class UAV():
    def __init__(self, meshFile: str, north: float = 0.0, east: float = 0.0, down: float = 0.0, pitch: float = 0.0, roll: float = 0.0, yaw: float = 0.0) -> None:
        """Instantiates the object and sets values.

        Args:
            meshFile (str): The .stl file of the uav.
            north (float, optional): The position north. Defaults to 0.0.
            east (float, optional): The position east. Defaults to 0.0.
            down (float, optional): The position down. Defaults to 0.0.
            pitch (float, optional): The pitch angle in radians. Defaults to 0.0.
            roll (float, optional): The roll angle in radians. Defaults to 0.0.
            yaw (float, optional): The yaw angle in radians. Defaults to 0.0.
        """
        # Get mesh file data
        theMesh = mesh.Mesh.from_file(meshFile)

        # Rotate file if default
        if meshFile == 'Assignment 05\\F117.stl':
            theMesh.rotate([0, 0, 0.5], np.radians(90+47.5))
        # Set values
        self.theMesh = theMesh
        self.north = np.array([north])
        self._north = 0.0
        self.east = np.array([east])
        self._east = 0.0
        self.down = np.array([down])
        self._down = 0.0
        self.pitch = np.array([pitch])
        self._pitch = 0.0
        self.roll = np.array([roll])
        self._roll = 0.0
        self.yaw = np.array([yaw])
        self._yaw = 0.0

    def _print_coordinates(self: Self) -> None:
        """Print the coordinates of the self object.

        Args:
            self (Self): Self object.
        """
        print('\n---------Coordinates---------')
        print('North: {0:.4f} meters'.format(self.north.sum()))
        print('East: {0:.4f} meters'.format(self.east.sum()))
        print('Down: {0:.4f} meters'.format(self.down.sum()))
        print('Pitch: {0:.4f} radians'.format(self.pitch.sum()))
        print('Roll: {0:.4f} radians'.format(self.roll.sum()))
        print('Yaw: {0:.4f} radians'.format(self.yaw.sum()))
        print('-----------------------------')

    def plot(self: Self, title: str, scaleFactor: float = 1/6) -> Tuple[Figure, Axes]:
        """Plots a mesh
        Args:
            theMesh (mesh.Mesh): The mesh to plot.
            title (string): The name of the plot.
            scaleFactor (float, optional): The scale factor of the object resolved in meteres. Defaults to 1/6.

        Returns:
            Tuple[Figure, Axes]: The figure and axes generated.
        """
        # Define the figure
        fig = plt.figure(figsize=(1.6*6, 0.9*6))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        # Add mesh to plot
        collection = mpl.art3d.Poly3DCollection(self.theMesh.vectors * scaleFactor, edgecolor='black', linewidth=0.2)
        ax.add_collection3d(collection)

        # Auto scale to mesh size
        scale = self.theMesh.points.flatten() * scaleFactor
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
    uav = UAV('F117.stl')
    fig, ax = uav.plot('F117 Nighthawk (1:1)')
    uav.update_uav(Plotting.generate_sliders(fig))
    anim = FuncAnimation(fig, Plotting.update_plot, frames=60, blit=False, fargs=[uav, ax, 'F117 Nighthawk (1:1)'])
    plt.show()
    anim.save