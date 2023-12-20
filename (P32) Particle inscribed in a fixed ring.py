'''
Trajectory finder via the Lagrangian method

This program resolves the trajectory traced by a particle moving alongside the inner circunference of a ring whose center is fixed at (0,0,0) and that is rotating around the z axis with frequency w.
We'll work on spherical coordinates, and will use the angle theta as the generalized coordinate (as r will always be a and phi=wt)

Notation:
    a = r = module of the position vector
    theta = angle between the position vector and the z axis
    phi = angle between the x axis and the projection of the position vector in the XY plane
    
    dotx = first time-derivative of any variable x
    dotnx = n time-derivative of any variable x (n is a number) 
    x0 = initial value for any variable x at time t=0
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.transforms import Affine2D
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.integrate import odeint

#The ring has its center fixed at (0,0,0)

nsteps = 1001   #number of steps in the time array
tp = 0.1      #pause time between steps of the animation

#Definition of initial parameters
global m
global R
global theta0
global dottheta0
global w
global g


m = float(input('Mass of the particle moving inside the ring (kg): '))             #mass 
R = float(input('Radius of the sphere (m): '))                                     #radius of the ring
theta0 = float(input('Initial angle theta (radians): \u03C0·'))*np.pi                    #initial angle theta
dottheta0 = float(input('Initial velocity in theta (radians/s): \u03C0·'))*np.pi       #initial angle speed dottheta
w = float(input('Frequency of rotation around the z axis (radians/s): \u03C0·'))*np.pi   #initial frequency around the z axis w
g = float(input('Gravitational Field Strength (kgm/s^2): '))

#Kinetic and Potential Energies, L function
def T(theta, dottheta):
    return 0.5*m*R**2*(dottheta**2 + w**2*(np.sin(theta))**2)

def V(theta):
    return m*g*R*np.cos(theta)

def L(theta,dottheta):
    return T(theta,dottheta) - V(theta)


#Euler-Lagrange Equations:
def EulerLagrange(y,t):
    theta, x = y
    dydt = [x, w**2*np.sin(theta)*np.cos(theta) + g/R*np.sin(theta)]
    return dydt

t = np.linspace(0,100,nsteps)

sol = odeint(EulerLagrange, [theta0, dottheta0], t)   #solution for theta,dottheta


#Cartesian Coordinates

x = np.ones((nsteps,))
x[:] = R*np.sin(sol[:,0])*np.cos(w*t[:])
y = np.ones((nsteps,))
y[:] = R*np.sin(sol[:,0])*np.sin(w*t[:])
z = np.ones((nsteps,))
z[:] = R*np.cos(sol[:,0])


#Function that creates the ring
def ring(time):
    n = 200
    thetas = np.linspace(0,2*np.pi,n)
    
    coords = np.ones((n,3))
    coords[:,0] = R*np.sin(thetas[:])*np.cos(w*time)    #x
    coords[:,1] = R*np.sin(thetas[:])*np.sin(w*time)    #y
    coords[:,2] = R*np.cos(thetas[:])    #z

    return coords




#Plotting in Crtesian Coordinates
fig = plt.figure(figsize=(8,8))
ejes = fig.add_subplot(projection='3d')

ejes.scatter(x[0],y[0],z[0],marker='o')


#Animating the trajectory in 3D for every time in the t array
for step in range(1,nsteps):
    ejes.cla()
    
    ejes.scatter(x[step],y[step],z[step],marker='o',c='r')  #plotting position for time t[step]
    
    ringcoords = ring(t[step])
    ejes.scatter(ringcoords[:,0], ringcoords[:,1], ringcoords[:,2],marker='2')
    
    ejes.set_xlim(-1.2*R,1.2*R)
    ejes.set_ylim(-1.2*R,1.2*R)
    ejes.set_zlim(-1.2*R,1.2*R)
    
    
    
    
    plt.pause(tp)
    
    
    
    









































    












































