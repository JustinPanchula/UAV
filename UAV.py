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
from scipy.integrate import solve_ivp
from control.matlab import lsim
import control as ctrl
from simple_pid import PID

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
        R = np.array([[np.cos(alpha), 0, np.sin(alpha)],
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
    def _wind(phi: float, theta: float, psi: float, Va: float, dt: float) -> np.ndarray:
        """Generates wind gusts

        Args:
            phi (float): The pitch angle in radians.
            theta (float): The roll angle in radians.
            psi (float): The yaw angle in radians.
            Va (float): The wind speed.
            dt (float): The time step.

        Returns:
            np.ndarray: The wind vector in body frame.
        """
        wn = 0
        we = 0
        wd = 0

        lu = 200
        lv = 200
        lw = 50

        sigma_u = 0.01
        sigma_v = sigma_u
        sigma_w = 0.01

        au = sigma_u * np.sqrt((2 * Va)/lu)
        av = sigma_v * np.sqrt((3 * Va)/lv)
        aw = sigma_w * np.sqrt((3 * Va)/lw)

        # Transfer functions
        num_u = [0, au]
        den_u = [1, Va/lu]
        sys_u = ctrl.tf(num_u, den_u)

        num_v = [av, (av * Va)/(np.sqrt(3) * lv)]
        den_v = [1, (2 * Va)/lv, (Va/lv)**2]
        sys_v = ctrl.tf(num_v, den_v)

        num_w = [aw, (aw * Va)/(np.sqrt(3) * lw)]
        den_w = [1, (2 * Va)/lw, (Va/lw)**2]
        sys_w = ctrl.tf(num_w, den_w)

        # Noise generation
        T = [0, dt]
        X0 = 0.0
        white_noise_u = np.random.normal(0, 1, 1)
        white_noise_v = np.random.normal(0, 1, 1)
        white_noise_w = np.random.normal(0, 1, 1)

        y_u, T, x_u = lsim(sys_u, white_noise_u[0], T, X0)
        y_v, T, x_v = lsim(sys_v, white_noise_v[0], T, X0)
        y_w, T, x_w = lsim(sys_w, white_noise_w[0], T, X0)

        # Gust components
        wg_u = y_u[1]
        wg_v = y_v[1]
        wg_w = y_w[1]

        Ws_v = np.array([wn, we, wd])
        R = _Framing._vehicle2body(phi, theta, psi)
        Ws_b = np.transpose(np.matmul(np.transpose(R), np.transpose(Ws_v)))
        Wg_b = np.array([wg_u, wg_v, wg_w])
        Vw = Wg_b + Ws_b

        return Vw

class Plotting():
    def generate_sliders(fig: Figure) -> np.ndarray:
        """Generates aircraft control sliders on the provided figure.

        Args:
            fig (Figure): The figure to add sliders to.

        Returns:
            np.ndarray: The sliders as an ordered numpy array.
        """
        sliders = np.empty(4, Slider)

        # Thrust
        sliders[0] = Slider(
            ax=fig.add_axes([0.11, 0.25, 0.0225, 0.63]),
            label='Thrust',
            valmin=0.0,
            valmax=1,
            valstep=0.1,
            valinit=0.0,
            orientation='vertical'
        )
        # Aileron
        sliders[1] = Slider(
            ax=fig.add_axes([0.25, 0.08, 0.65, 0.03]),
            label='Aileron',
            valmin=-0.3,
            valmax=0.3,
            valstep=0.01,
            valinit=0.0
        )
        # Elevator
        sliders[2] = Slider(
            ax=fig.add_axes([0.25, 0.05, 0.65, 0.03]),
            label='Elevator',
            valmin=-0.3,
            valmax=0.3,
            valstep=0.01,
            valinit=0.0
        )
        # Rudder
        sliders[3] = Slider(
            ax=fig.add_axes([0.25, 0.02, 0.65, 0.03]),
            label='Rudder',
            valmin=-0.3,
            valmax=0.3,
            valstep=0.01,
            valinit=0.0
        )
        return sliders

    def update_plot(t: float, uav: object, planeAx: Axes, title: str, scaleFactor: float = 1/6) -> Axes:
        """Updates the uav to model the simulation

        Args:
            t (float): Inherent to FuncAnimation.
            uav (object): The UAV.
            planeAx (Axes): The axis to draw the UAV on.
            title (str): The tile of the plot.
            scaleFactor (float, optional): The scale factor of the model. Defaults to 1/6.

        Returns:
            Axes: The redrawn UAV axis.
        """
        # Update Coordinates
        uav.uav_dynamics()

        # Add to time
        uav._track_t = np.append(uav._track_t, uav._track_t[-1] + uav._dt)

        # Update mesh
        uav._Mesh = mesh.Mesh.from_file(uav._meshFile)
        uav._Mesh.x += uav._north
        uav._Mesh.y -= uav._east
        uav._Mesh.z -= uav._down
        uav._Mesh.rotate([0.5, 0.0, 0.0], -uav._phi)
        uav._Mesh.rotate([0.0, 0.5, 0.0], uav._theta)
        uav._Mesh.rotate([0.0, 0.0, 0.5], uav._psi)

        # Update state lists
        uav._track_north = np.append(uav._track_north, uav._north)
        uav._track_east = np.append(uav._track_east, uav._east)
        uav._track_down = np.append(uav._track_down, uav._down)
        uav._track_u = np.append(uav._track_u, uav._u)
        uav._track_v = np.append(uav._track_v, uav._v)
        uav._track_w = np.append(uav._track_w, uav._w)
        uav._track_phi = np.append(uav._track_phi, uav._phi)
        uav._track_theta = np.append(uav._track_theta, uav._theta)
        uav._track_psi = np.append(uav._track_psi, uav._psi)
        uav._track_p = np.append(uav._track_p, uav._p)
        uav._track_q = np.append(uav._track_q, uav._q)
        uav._track_r = np.append(uav._track_r, uav._r)

        # Clear the axis
        planeAx.clear()

        # Re-add collection
        collection = mpl.art3d.Poly3DCollection(uav._Mesh.vectors, edgecolor='black', linewidth=0.2)
        planeAx.add_collection3d(collection)

        # Auto scale to mesh size
        scale = uav._Mesh.points.flatten()
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
            axs[i-1].set_title(uav._state_names[i-1])

        # Plot each axs
        axs[0].plot(uav._track_t, uav._track_north); axs[0].set_ylabel('(m)')
        axs[1].plot(uav._track_t, uav._track_east); axs[1].set_ylabel('(m)')
        axs[2].plot(uav._track_t, uav._track_down); axs[2].set_ylabel('(m)')
        axs[3].plot(uav._track_t, uav._track_u); axs[3].set_ylabel('(m/s)')
        axs[4].plot(uav._track_t, uav._track_v); axs[4].set_ylabel('(m/s)')
        axs[5].plot(uav._track_t, uav._track_w); axs[5].set_ylabel('(m/s)')
        axs[6].plot(uav._track_t, uav._track_phi); axs[6].set_ylabel('(radians)')
        axs[7].plot(uav._track_t, uav._track_theta); axs[7].set_ylabel('(radians)')
        axs[8].plot(uav._track_t, uav._track_psi); axs[8].set_ylabel('(radians)')
        axs[9].plot(uav._track_t, uav._track_p); axs[9].set_ylabel('(radians/sec)')
        axs[10].plot(uav._track_t, uav._track_q); axs[10].set_ylabel('(radians/sec)')
        axs[11].plot(uav._track_t, uav._track_r); axs[11].set_ylabel('(radians/sec)')

        # Format figure
        fig.set_tight_layout(True)

        # Display
        plt.show()
        return

class UAV():
    def __init__(self, meshFile: str, airspeed: float = 30.0) -> None:
        """Instantiates the object and sets values.
        """
        # Set mesh
        self._meshFile = meshFile
        self._Mesh = mesh.Mesh.from_file(self._meshFile)

        # Set state values
        self._track_north = np.array([0.0], float)     # Position North
        self._north = 0.0
        self._track_east = np.array([0.0], float)      # Position East
        self._east = 0.0
        self._track_down = np.array([0.0], float)      # Position South
        self._down = 0.0
        self._track_u = np.array([0.0], float)         # Velocity along i
        self._u = 0.0
        self._track_v = np.array([0.0], float)         # Velocity along j
        self._v = 0.0
        self._track_w = np.array([0.0], float)         # Velocity along k
        self._w = 0.0
        self._track_phi = np.array([0.0], float)       # Roll
        self._phi = 0.0
        self._track_theta = np.array([0.0], float)     # Pitch
        self._theta = 0.0
        self._track_psi = np.array([0.0], float)       # Yaw
        self._psi = 0.0
        self._track_p = np.array([0.0], float)         # Roll rate
        self._p = 0.0
        self._track_q = np.array([0.0], float)         # Pitch rate
        self._q = 0.0
        self._track_r = np.array([0.0], float)         # Yaw rate
        self._r = 0.0

        # List state names
        self._state_names = np.array(['North', 'East', 'Down', 'u', 'v', 'w', 'Phi', 'Theta', 'Psi', 'l', 'm', 'n'], str)

        # Set control surfaces
        self._thrust = 0.46237247565931516
        self._aileron = -0.02440474
        self._elevator = -0.026687397543602744
        self._rudder = -0.03046846

        # Set control surface forces
        self._fx = 0.0
        self._fy = 0.0
        self._fz = 0.0
        self._l = 0.0
        self._m = 0.0
        self._n = 0.0

        # Set aerodynamics parameters
        self._S_wing = 0.55
        self._b_wing = 2.8956
        self._c_wing = 0.18994
        self._S_prop = 0.2027
        self._c_prop = 1.0
        self._k_motor = 80.0
        self._rho = 1.2682
        self._e = 0.9
        self._ar = self._b_wing**2/self._S_wing

        # Set aerodynamic coefficients
        self._coeffs = {
            'CL0': 0.23,
            'CD0': 0.03,
            'Cm0': -0.02338,
            'CLalpha': 3.45,
            'CDalpha': 0.030,
            'Cmalpha': -0.38,
            'CLq': 0.0,
            'CDq': 0.0,
            'Cmq': -3.6,
            'CLdeltae': 0.36,
            'CDdeltae': 0.0,
            'Cmdeltae': -0.5,
            'Cdp': 0.0,
            'Cy0': 0.0,
            'Cl0': 0.0,
            'Cn0': 0.0,
            'Cybeta': -0.98,
            'Clbeta': -0.12,
            'Cnbeta': 0.25,
            'Cyp': 0.0,
            'Clp': -0.26,
            'Cnp': 0.022,
            'Cyr': 0.0,
            'Clr': 0.14,
            'Cnr': -0.35,
            'Cydeltaa': 0.0,
            'Cldeltaa': 0.08,
            'Cndeltaa': 0.06,
            'Cydeltar': -0.17,
            'Cldeltar': 0.105,
            'Cndeltar': -0.032
        }

        # Set wind state values
        self._airspeed = airspeed
        self._alpha = 0.0
        self._beta = 0.0

        # Set additional parameters
        self._M = 50.0
        self._alpha0 = 0.4712
        self._epsilon = 0.1592

        # Set inertial parameters
        self._mass = 13.5
        self._g = 9.806650
        self._Jx = 0.8244
        self._Jy = 1.135
        self._Jz = 1.759
        self._Jxz = 0.1204

        # Set gamma values
        self._G0 = (self._Jx * self._Jz) - self._Jxz ** 2
        self._G1 = (self._Jxz * (self._Jx - self._Jy + self._Jz))/self._G0
        self._G2 = (self._Jz * (self._Jz - self._Jy) + self._Jxz ** 2)/self._G0
        self._G3 = self._Jz/self._G0
        self._G4 = self._Jxz/self._G0
        self._G5 = (self._Jz - self._Jx) / self._Jy
        self._G6 = self._Jxz/self._Jy
        self._G7 = (self._Jx * (self._Jx - self._Jy) + self._Jxz ** 2) / self._G0
        self._G8 = self._Jx/self._G0

        # Set simulation parameters
        self._track_t = np.array([0.0], float)
        self._dt = 0.05
        self._duration = 60
        return

    # # Getters, setters, and deleters
    # @property
    # def north(self: Self):
    #     return self._track_north[-1]

    # @north.setter
    # def north(self: Self, val: float):
    #     if isinstance(val, float):
    #         self._track_north = np.append(self._track_north, val)
    #         self._north = val
    #     else:
    #         print('Value must be a float')

    # @north.deleter
    # def north(self: Self):
    #     del self._track_north
    #     del self._north

    def _print_coordinates(self: Self) -> None:
        """Print the coordinates of the self object.

        Args:
            self (Self): Self object.
        """
        print('\n---------Coordinates---------')
        print(f'Time: {self._track_t[-1]} seconds')
        print(f'North: {self._track_north[-1]} meters')
        print(f'East: {self._track_east[-1]} meters')
        print(f'Down: {self._track_down[-1]} meters')
        print(f'Roll: {self._track_phi[-1]:.4f} radians')
        print(f'Pitch: {self._track_theta[-1]:.4f} radians')
        print(f'Yaw: {self._track_psi[-1]:.4f} radians')
        print('-----------------------------')
        return

    def plot(self: Self, title: str, scaleFactor: float = 6) -> Tuple[Figure, Axes]:
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
        collection = mpl.art3d.Poly3DCollection(self._Mesh.vectors, edgecolor='black', linewidth=0.2)
        ax.add_collection3d(collection)

        # Auto scale to mesh size
        scale = self._Mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)

        # Format the plot
        ax.set_title(title)

        return fig, ax

    def compute_trim(self: Self) -> None:
        return

    def update_uav(self: Self, sliders: np.ndarray) -> None:
        """Updates the self paramters based on the slider changes.

        Args:
            self (Self): The UAV Object.
            sliders (np.ndarray): The numpy array of sliders.
        """
        def _update_sliders(val) -> None:
            """Private function used to satisfy "Slider.on_changed()" method.

            Args:
                val (private): Inherent to "Slider.on_changed()" method.
            """
            self._thrust = sliders[0].val
            self._aileron = sliders[1].val
            self._elevator = sliders[2].val
            self._rudder = sliders[3].val
            self._print_coordinates()
            return

        sliders[0].on_changed(_update_sliders)
        sliders[1].on_changed(_update_sliders)
        sliders[2].on_changed(_update_sliders)
        sliders[3].on_changed(_update_sliders)
        return

    def uav_dynamics(self: Self) -> None:
        def _wind(self: Self) -> None:
            """Generates wind and constantly updates according to the Dryden Gust Model from "_Environment"

            Args:
                self (Self): The UAV object.
            """
            Vw = _Environment._wind(self._phi, self._theta, self._psi, self._airspeed, self._dt)
            ur = self._u - Vw[0]
            vr = self._v - Vw[1]
            wr = self._w - Vw[2]

            self._airspeed = np.sqrt(ur**2 + vr**2 + wr**2)
            self._alpha = np.arctan(wr/ur)
            self._beta = np.arcsin(vr/self._airspeed)
            return

        def _dynamics(t, y, UAV: object) -> np.ndarray:
            """Calculates x_dot all in one function.

            Args:
                t (private): Inherent to solve_ivp().
                y (private): Inherent to solve_ivp().
                UAV (object): The UAV.

            Returns:
                np.ndarray: The x_dot states.
            """
            # l moment
            self._l = (0.5 * self._rho * self._airspeed**2 * self._S_wing) * (self._b_wing * (self._coeffs['Cl0'] + (self._coeffs['Clbeta'] * self._beta) + (self._coeffs['Clp'] * (self._b_wing/(2 * self._airspeed)) * self._p) + (self._coeffs['Clr'] * (self._b_wing/(2 * self._airspeed)) * self._r) + (self._coeffs['Cldeltaa'] * self._aileron) + (self._coeffs['Cldeltar'] * self._rudder))) + (0)

            # m moment
            self._m = (0.5 * self._rho * self._airspeed**2 * self._S_wing) * (self._c_wing * (self._coeffs['Cm0'] + (self._coeffs['Cmalpha'] * self._alpha) + (self._coeffs['Cmq'] * (self._c_wing/(2 * self._airspeed)) * self._q) + (self._coeffs['Cmdeltae'] * self._elevator))) + (0)

            # n moment
            self._n = (0.5 * self._rho * self._airspeed**2 * self._S_wing) * (self._b_wing * (self._coeffs['Cn0'] + (self._coeffs['Cnbeta'] * self._beta) + (self._coeffs['Cnp'] * (self._b_wing/(2 * self._airspeed)) * self._p) + (self._coeffs['Cnr'] * (self._b_wing/(2 * self._airspeed)) * self._r) + (self._coeffs['Cndeltaa'] * self._aileron) + (self._coeffs['Cndeltar'] * self._rudder))) + (0)

            # Calculate sigma, Cl, and Cd
            sigma = (1 + np.exp(-self._M * (self._alpha - self._alpha0)) + np.exp(self._M * (self._alpha + self._alpha0)))/((1 + np.exp(-self._M * (self._alpha - self._alpha0))) * (1 + np.exp(self._M * (self._alpha + self._alpha0))))
            Cl = (1 - sigma) * (self._coeffs['CL0'] + ((self._coeffs['CLalpha']) * self._alpha)) + (sigma * (2 * np.sign(self._alpha) * np.sin(self._alpha)**2 * np.cos(self._alpha)))
            Cd = self._coeffs['Cdp'] +((self._coeffs['CL0'] + (self._coeffs['CLalpha'] * self._alpha))**2/(np.pi * self._e * self._ar))

            # Calculate Cx, Cxq, Cxde, Cz, Czq, and Czqe
            Cx = -Cd * np.cos(self._alpha) + Cl * np.sin(self._alpha)
            Cxq = -self._coeffs['CDq'] * np.cos(self._alpha) + self._coeffs['CLq'] * np.sin(self._alpha)
            Cxde = -self._coeffs['CDdeltae'] * np.cos(self._alpha) + self._coeffs['CLdeltae'] * np.sin(self._alpha)
            Cz = -Cd * np.sin(self._alpha) - Cl * np.cos(self._alpha)
            Czq = -self._coeffs['CDq'] * np.sin(self._alpha) - self._coeffs['CLq'] * np.cos(self._alpha)
            Czde = -self._coeffs['CDdeltae'] * np.sin(self._alpha) - self._coeffs['CLdeltae'] * np.cos(self._alpha)

            # Calculate Fx
            self._fx = (-self._mass * self._g * np.sin(self._theta)) + ((0.5 * self._rho * self._airspeed**2 * self._S_wing) * (Cx + (Cxq * (self._c_wing/(2 * self._airspeed) * self._q)) + (Cxde * self._elevator))) + ((0.5 * self._rho * self._S_prop * self._c_prop) * ((self._k_motor * self._thrust)**2 - self._airspeed**2))

            # Calculate Fy
            self._fy = (self._mass * self._g * np.cos(self._theta) * np.sin(self._phi)) + ((0.5 * self._rho * self._airspeed**2 * self._S_wing) * (self._coeffs['Cy0'] + (self._coeffs['Cybeta'] * self._beta) + (self._coeffs['Cyp'] * (self._b_wing/(2 * self._airspeed)) * self._p) * (self._coeffs['Cyr'] * (self._b_wing/(2 * self._airspeed)) * self._r) + (self._coeffs['Cydeltaa'] * self._aileron) + (self._coeffs['Cydeltar'] * self._rudder))) + (0)

            # Calculate Fz
            self._fz = (self._mass * self._g * np.cos(self._theta) * np.cos(self._phi)) + ((0.5 * self._rho * self._airspeed**2 * self._S_wing) * (Cz + (Czq * (self._c_wing/(2 * self._airspeed)) * self._q) + (Czde * self._elevator))) + (0) + (self._mass * self._g)

            # Calculate derivatives
            p_prime = (self._G1 * self._p * self._q - self._G2 * self._q * self._r) + (self._G3 * self._l + self._G4 * self._n)
            q_prime = (self._G5 * self._p * self._r - self._G6 * (self._p**2 - self._r**2)) + ((1/self._Jy) * self._m)
            r_prime = (self._G7 * self._p * self._q - self._G1 * self._q * self._r) + (self._G4 * self._l + self._G8 * self._n)
            phi_prime, theta_prime, psi_prime = np.dot(_Framing._pqr2phithetapsi(self._phi, self._theta, self._psi), np.array([self._p, self._q, self._r]))
            u_prime = (self._r * self._v - self._q * self._w) + (self._fx/self._mass)
            v_prime = (self._p * self._w - self._r * self._u) + (self._fy/self._mass)
            w_prime = (self._q * self._u - self._p * self._v) + (self._fz/self._mass)
            n_prime, e_prime, d_prime = np.dot(np.transpose(_Framing._vehicle2body(self._phi, self._theta, self._psi)), np.array([self._u, self._v, self._w]))

            # Format
            x_dot = np.array([n_prime, e_prime, d_prime, u_prime, v_prime, w_prime, phi_prime, theta_prime, psi_prime, p_prime, q_prime, r_prime], float)

            return x_dot

        # Define states
        x = np.array([self._north, self._east, self._down, self._u, self._v, self._w, self._phi, self._theta, self._psi, self._p, self._q, self._r])

        # Run functions
        _wind(self)
        s = solve_ivp(lambda t, y: _dynamics(t, y, self), [0, self._dt], x)

        # Get results
        self._north, self._east, self._down, self._u, self._v, self._w, self._phi, self._theta, self._psi, self._p, self._q, self._r = s.y[:, -1].T

        return

if __name__ == '__main__':
    meshFile = 'F117.stl'
    title = 'F117 Nighthawk (1:1)'
    uav = UAV(meshFile)
    fig, ax = uav.plot(title)
    sliders = Plotting.generate_sliders(fig)
    uav.update_uav(sliders)
    anim = FuncAnimation(fig, Plotting.update_plot, frames=300, fargs=[uav, ax, title], repeat=False)
    plt.show()
    Plotting.plot_states(uav)
    anim.save