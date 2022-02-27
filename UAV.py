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
from stl import mesh
from scipy.integrate import solve_ivp

# Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits import mplot3d as mpl
from matplotlib.widgets import Slider

# Typing imports
from typing_extensions import Self
from typing import Any, Tuple

class _Framing():
    def _vehicle2body(phi: float, theta: float, psi: float) -> np.ndarray:
        """Generates a matrix to rotate vectors from vehicle frame to body frame.

        Args:
            phi (float): The roll angle in radians.
            theta (float): The pitch angle in radians.
            psi (float): The yaw angle in radians.

        Returns:
            np.ndarray: The appropriate rotation matrix.
        """
        R = np.array([[np.cos(theta) * np.cos(psi), np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), np.cos(phi) *
                       np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
                      [np.cos(theta) * np.sin(psi), np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), np.cos(phi) *
                       np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
                      [-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]], float)
        return R

    def _body2stability(alpha: float) -> np.ndarray:
        """Generates a matrix to rotate vectors from body frame to stability frame.

        Args:
            alpha (float): The AoA in radians.

        Returns:
            np.ndarray: The rotation matrix.
        """
        R = np.array([[np.cos(alpha, 0, np.sin(alpha))],
                      [0, 1, 0],
                      [-np.sin(alpha), 0, np.cos(alpha)]], float)
        return R

    def _stability2wind(beta: float) -> np.ndarray:
        """Generates a matrix to rotate vectors from stability frame to wind frame.

        Args:
            beta (float): UNKOWN. FIXME: What is beta?

        Returns:
            np.ndarray: The rotation matrix.
        """
        R = np.array([[np.cos(beta), np.sin(beta), 0],
                      [-np.sin(beta), np.cos(beta), 0],
                      [0, 0, 1]], float)
        return R

    def _pqr2phithetapsi(phi: float, theta: float, psi: float) -> np.ndarray:
        """Generates a matrix to rotate vectors from p-q-r to phi-theta-psi

        Args:
            phi (float): The roll angle in radians.
            theta (float): The pitch angle in radians.
            psi (float): The yaw angle in radians.

        Returns:
            np.ndarray: The appropriate rotation matrix.
        """
        R = np.array([[1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                      [0, np.cos(phi), -np.sin(phi)],
                      [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]], float)
        return R

class _Environment():
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

    def update_plot(t: float, uav: object, planeAx: Axes, title: str, scaleFactor: float = 1/6) -> Axes:
        # Update Coordinates
        uav.uav_dynamics()

        # Add to time
        uav.t = np.append(uav.t, t)

        # Update mesh
        uav.Mesh.x += uav._north * 50
        uav.Mesh.y += uav._east * 50
        uav.Mesh.z -= uav._down * 50
        uav.Mesh.rotate([0.5, 0.0, 0.0], -uav._phi)
        uav.Mesh.rotate([0.0, 0.5, 0.0], -uav._theta)
        uav.Mesh.rotate([0.0, 0.0, 0.5], uav._psi)

        # Update state lists
        uav.north = np.append(uav.north, uav._north)
        uav.east = np.append(uav.east, uav._east)
        uav.down = np.append(uav.down, uav._down)
        uav.u = np.append(uav.u, uav._u)
        uav.v = np.append(uav.v, uav._v)
        uav.w = np.append(uav.w, uav._w)
        uav.phi = np.append(uav.phi, uav._phi)
        uav.theta = np.append(uav.theta, uav._theta)
        uav.psi = np.append(uav.psi, uav._psi)
        uav.p = np.append(uav.p, uav._p)
        uav.q = np.append(uav.q, uav._q)
        uav.r = np.append(uav.r, uav._r)

        # Clear the axis
        planeAx.clear()

        # Re-add collection
        collection = mpl.art3d.Poly3DCollection(uav.Mesh.vectors * scaleFactor, edgecolor='black', linewidth=0.2)
        planeAx.add_collection3d(collection)

        # Auto scale to mesh size
        scale = uav.Mesh.points.flatten() * scaleFactor
        planeAx.auto_scale_xyz(scale, scale, scale)

        # Format the plot
        planeAx.set_title(title)
        return planeAx

    def plot_states(uav: object) -> None:
        # Create figure
        fig = plt.figure(figsize=(1.6*6, 0.9*6))

        # Create axs list
        axs = np.empty(12, Axes)

        # Populate axs list
        for i in range(1, 13):
            axs[i-1] = fig.add_subplot(4, 3, i)
            axs[i-1].set_xlabel('Time (sec)')
            axs[i-1].set_title(uav.state_names[i-1])

        # Plot each axs
        axs[0].plot(uav.t, uav.north)
        axs[0].set_ylabel('(m)')
        axs[1].plot(uav.t, uav.east)
        axs[1].set_ylabel('(m)')
        axs[2].plot(uav.t, uav.down)
        axs[2].set_ylabel('(m)')
        axs[3].plot(uav.t, uav.u)
        axs[3].set_ylabel('(m/s)')
        axs[4].plot(uav.t, uav.v)
        axs[4].set_ylabel('(m/s)')
        axs[5].plot(uav.t, uav.w)
        axs[5].set_ylabel('(m/s)')
        axs[6].plot(uav.t, uav.phi)
        axs[6].set_ylabel('(radians)')
        axs[7].plot(uav.t, uav.theta)
        axs[7].set_ylabel('(radians)')
        axs[8].plot(uav.t, uav.psi)
        axs[8].set_ylabel('(radians)')
        axs[9].plot(uav.t, uav.p)
        axs[9].set_ylabel('(radians/sec)')
        axs[10].plot(uav.t, uav.q)
        axs[10].set_ylabel('(radians/sec)')
        axs[11].plot(uav.t, uav.r)
        axs[11].set_ylabel('(radians/sec)')

        # Format figure
        fig.set_tight_layout(True)

        plt.show()
        return

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

        # Set state values
        self.north = np.array([0.0], float)     # Position North
        self._north = 0.0
        self.east = np.array([0.0], float)      # Position East
        self._east = 0.0
        self.down = np.array([0.0], float)      # Position South
        self._down = 0.0
        self.u = np.array([0.0], float)         # Velocity along i
        self._u = 0.0
        self.v = np.array([0.0], float)         # Velocity along j
        self._v = 0.0
        self.w = np.array([0.0], float)         # Velocity along k
        self._w = 0.0
        self.phi = np.array([0.0], float)       # Roll
        self._phi = 0.0
        self.theta = np.array([0.0], float)     # Pitch
        self._theta = 0.0
        self.psi = np.array([0.0], float)       # Yaw
        self._psi = 0.0
        self.p = np.array([0.0], float)         # Roll rate
        self._p = 0.0
        self.q = np.array([0.0], float)         # Pitch rate
        self._q = 0.0
        self.r = np.array([0.0], float)         # Yaw rate
        self._r = 0.0

        # List state names
        self.state_names = np.array(['North', 'East', 'Down', 'u', 'v', 'w', 'Phi', 'Theta', 'Psi', 'l', 'm', 'n'], str)

        # Set intrisic values
        self.mass = mass
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.Jxz = Jxz

        # Set gamma values
        self._G0 = Jx * Jz - Jxz ** 2
        self._G1 = (Jxz * (Jx - Jy + Jz))/self._G0
        self._G2 = (Jz * (Jz - Jy) + Jxz ** 2)/self._G0
        self._G3 = Jz/self._G0
        self._G4 = Jxz/self._G0
        self._G5 = (Jz - Jx) / Jy
        self._G6 = Jxz/Jy
        self._G7 = (Jx * (Jx - Jy) + Jxz ** 2) / self._G0
        self._G8 = Jx/self._G0

        # Set slider value storage
        self._fx = 0.0
        self._fy = 0.0
        self._fz = 0.0
        self._l = 0.0
        self._m = 0.0
        self._n = 0.0

        # Set simulation parameters
        self.t = np.array([0.0], float)
        self.dt = 0.01
        self.duration = 60
        return

    def _print_coordinates(self: Self) -> None:
        """Print the coordinates of the self object.

        Args:
            self (Self): Self object.
        """
        print('\n---------Coordinates---------')
        print('North: {} meters'.format(self.north.sum()))
        print('East: {} meters'.format(self.east.sum()))
        print('Down: {} meters'.format(self.down.sum()))
        print('Pitch: {:.4f} radians'.format(self.phi.sum()))
        print('Roll: {:.4f} radians'.format(self.theta.sum()))
        print('Yaw: {:.4f} radians'.format(self.psi.sum()))
        print('-----------------------------')
        return

    def plot(self: Self, title: str, scaleFactor: float = 1/6) -> Tuple[Figure, Axes]:
        """Plots a mesh.

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
        """Updates the self paramters based on the slider changes.

        Args:
            self (Self): Self object.
            sliders (np.ndarray): The numpy array of sliders.
        """
        def _update_sliders(val) -> None:
            """Private function used to satisfy "Slider.on_changed()" method.

            Args:
                val (private): Inherent to "Slider.on_changed()" method.
            """
            self._fx = sliders[0].val
            self._fy = sliders[1].val
            self._fz = sliders[2].val
            self._l = sliders[4].val
            self._m = sliders[3].val
            self._n = sliders[5].val
            self._print_coordinates()
            return

        sliders[0].on_changed(_update_sliders)
        sliders[1].on_changed(_update_sliders)
        sliders[2].on_changed(_update_sliders)
        sliders[3].on_changed(_update_sliders)
        sliders[4].on_changed(_update_sliders)
        sliders[5].on_changed(_update_sliders)
        return

    def uav_dynamics(self: Self) -> None:
        def _dynamics(t, y, integrand: Any) -> Any:
            """Returns the integrand for "solve_ivp()".

            Args:
                t (private): Inherent to "solve_ivp()" method.
                y (private): Inherent to "solve_ivp()" method.
                integrand (Any): The integrand.

            Returns:
                Any: The value of the integrand.
            """
            return integrand

        def _lmn2pqr(self: Self) -> None:
            """Transforms moment inputs to pqr.

            Args:
                self (Self): The UAV object.
            """
            # p
            p_prime = (self._G1 * self._p * self._q - self._G2 * self._q * self._r) + (self._G3 * self._l + self._G4 * self._n)
            s = solve_ivp(lambda t, y: _dynamics(t, y, p_prime), [0, self.dt], [self._p], t_eval=np.linspace(0, self.dt, self.duration))
            ans = s.y[:, -1].T
            self._p, = ans

            # q
            q_prime = (self._G5 * self._p * self._r - self._G6 * (self._p**2 - self._r**2)) + ((1/self.Jy) * self._m)
            s = solve_ivp(lambda t, y: _dynamics(t, y, q_prime), [0, self.dt], [self._q], t_eval=np.linspace(0, self.dt, self.duration))
            ans = s.y[:, -1].T
            self._q, = ans

            # r
            r_prime = (self._G7 * self._p * self._q - self._G1 * self._q * self._r) + (self._G4 * self._l + self._G8 * self._n)
            s = solve_ivp(lambda t, y: _dynamics(t, y, r_prime), [0, self.dt], [self._r], t_eval=np.linspace(0, self.dt, self.duration))
            ans = s.y[:, -1].T
            self._r, = ans
            return

        def _pqr2phithetapsi(self: Self, R: np.ndarray) -> None:
            """Transforms angluar rates into angles.

            Args:
                self (Self): The UAV object.
                R (np.ndarray): The rotation matrix.
            """
            phithetapsi_prime = np.dot(R, np.array([self._p, self._q, self._r]))
            s = solve_ivp(lambda t, y: _dynamics(t, y, phithetapsi_prime), [0, self.dt], [self._p, self._q, self._r], t_eval=np.linspace(0, self.dt, self.duration))
            ans = s.y[:, -1].T
            self._phi, self._theta, self._psi = ans
            return

        def _f2uvw(self: Self) -> None:
            """Transforms forces to linear velocities.

            Args:
                self (Self): The UAV object.
            """
            # u
            u_prime = (self._r * self._v - self._q * self._w) + (self._fx/self.mass)
            s = solve_ivp(lambda t, y: _dynamics(t, y, u_prime), [0, self.dt], [self._u], t_eval=np.linspace(0, self.dt, self.duration))
            ans = s.y[:, -1].T
            self._u, = ans

            # v
            v_prime = (self._p * self._w - self._r * self._u) + (self._fy/self.mass)
            s = solve_ivp(lambda t, y: _dynamics(t, y, v_prime), [0, self.dt], [self._v], t_eval=np.linspace(0, self.dt, self.duration))
            ans = s.y[:, -1].T
            self._v, = ans

            # w
            w_prime = (self._q * self._u - self._p * self._v) + (self._fz/self.mass)
            s = solve_ivp(lambda t, y: _dynamics(t, y, w_prime), [0, self.dt], [self._w], t_eval=np.linspace(0, self.dt, self.duration))
            ans = s.y[:, -1].T
            self._w, = ans
            return

        def _uvw2ned(self: Self, R: np.ndarray) -> None:
            """Transforms linear velocity to positions

            Args:
                self (Self): The UAV object
                R (np.ndarray): The rotation matrix.
            """
            ned_prime = np.dot(R, np.array([self._u, self._v, self._w]))
            s = solve_ivp(lambda t, y: _dynamics(t, y, ned_prime), [0, self.dt], [self._north, self._east, self._down], t_eval=np.linspace(0, self.dt, self.duration))
            ans = s.y[:, -1].T
            self._north, self._east, self._down = ans
            return

        # Moments
        _lmn2pqr(self)

        # pqr
        _pqr2phithetapsi(self, _Framing._pqr2phithetapsi(self._phi, self._theta, self._psi))

        # Forces
        _f2uvw(self)

        # Velocities
        _uvw2ned(self, _Framing._vehicle2body(self._phi, self._theta, self._psi))
        return

if __name__ == '__main__':
    meshFile = 'F117.stl'
    title = 'F117 Nighthawk (1:1)'
    uav = UAV(meshFile)
    fig, ax = uav.plot(title)
    sliders = Plotting.generate_sliders(fig)
    uav.update_uav(sliders)
    anim = FuncAnimation(fig, Plotting.update_plot, frames=60, blit=False, fargs=[uav, ax, title])
    plt.show()
    Plotting.plot_states(uav)
    anim.save