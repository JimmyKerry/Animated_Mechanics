'''
Trajectory finder via the Lagrangian method

This program resolves the trajectory traced by a particle moving alongside the inner surface of the bottom half of a sphere of radius r whose center is fixed at (0,0,0) due to the gravitational potential 
We'll work on spherical coordinates, and will use the angles theta and phi as generalized coordinates (as r will always be a)

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
from scipy.integrate import odeint

#The ring has its center fixed at (0,0,0)

nsteps = 101   #number of steps in the time array
tp = 0.01      #pause time between steps of the animation

#Definition of initial parameters
global m
global R
global theta0
global dottheta0
global w
global g


m = float(input('Mass of the particle moving inside the ring (kg): '))             #mass 
R = float(input('Radius of the sphere (m): '))                                     #radius of the ring
theta0 = float(input('Initial angle theta (radians): \u03C0·'))                    #initial angle theta
dottheta0 = float(input('Initial velocity in theta (radians/s): \u03C0·'))         #initial angle speed dottheta
w = float(input('Frequency of rotation around the z axis (radians/s): \u03C0·'))   #initial frequency around the z axis w
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

t = np.linspace(0,10,nsteps)

sol = odeint(EulerLagrange, [theta0, dottheta0], t)   #solution for theta,dottheta


#Cartesian Coordinates

x = np.ones((nsteps,))
x[:] = R*np.sin(sol[:,0])*np.cos(w*t[:])
y = np.ones((nsteps,))
y[:] = R*np.sin(sol[:,0])*np.sin(w*t[:])
z = np.ones((nsteps,))
z[:] = R*np.cos(sol[:,0])


#Plotting in Crtesian Coordinates
fig = plt.figure(figsize=(8,8))
ejes = fig.add_subplot(111)

ejes.plot(t[0],x[0],'bo')
ejes.plot(t[0],y[0],'go')
ejes.plot(t[0],z[0],'ro')

ejes.set_xlim(0,t[-1])
ejes.set_ylim(-1.2*R,1.2*R)

#Animating the trajectory in 3D for every time in the t array
for step in range(1,nsteps):
    #ejes.cla()
    ejes.plot(t[step],x[step],'bo')
    ejes.plot(t[step],y[step],'go')
    ejes.plot(t[step],z[step],'ro')
    
    ejes.set_xlim(0,t[-1])
    ejes.set_ylim(-1.2*R,1.2*R)
    
    plt.pause(tp)
    
    
    
    









































    












































