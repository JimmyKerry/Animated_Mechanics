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

default_flag = bool(int(input('\n-- (1) -- Use default values for the initial conditions of the system\n\tor\n-- (0) -- Manually set initial conditions \n\t(1 or 0): ')))

if default_flag==True:
    m = 1
    R = 1.5
    theta0 = 0.65*np.pi
    dottheta0 = 0*np.pi
    w = 0.35*np.pi
    g = 9.81
    print('\n----- Initial Conditions of the System (Defaults) -----')
    print('\n\tMass of the particle moving inside the ring (kg):', m)
    print('\tRadius of the sphere (m):', R)
    print('\tInitial angle theta (radians): \u03C0·', theta0)
    print('\tInitial velocity in theta (radians/s): \u03C0·', dottheta0)
    print('\tFrequency of rotation around the z axis (radians/s): \u03C0·', w)
    print('\tGravitational Field Strength (kgm/s^2): ', g)
else:
    print('\n----- Initial Conditions of the System (Manual Settings) -----')
    m = float(input('\n\tMass of the particle moving inside the ring (kg): '))             #mass 
    R = float(input('\tRadius of the sphere (m): '))                                     #radius of the ring
    theta0 = float(input('\tInitial angle theta (radians): \u03C0·'))*np.pi                    #initial angle theta
    dottheta0 = float(input('\tInitial velocity in theta (radians/s): \u03C0·'))*np.pi       #initial angle speed dottheta
    w = float(input('\tFrequency of rotation around the z axis (radians/s): \u03C0·'))*np.pi   #initial frequency around the z axis w
    g = float(input('\tGravitational Field Strength (kgm/s^2): '))




#Euler-Lagrange Equations:
def EulerLagrange(y,t):
    theta, x = y
    dydt = [x, w**2*np.sin(theta)*np.cos(theta) + g/R*np.sin(theta)]
    return dydt

tmax = 100
t = np.linspace(0,tmax,nsteps)

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
    n = 150
    thetas = np.linspace(0,2*np.pi,n)
    
    coords = np.ones((n,3))
    coords[:,0] = R*np.sin(thetas[:])*np.cos(w*time)    #x
    coords[:,1] = R*np.sin(thetas[:])*np.sin(w*time)    #y
    coords[:,2] = R*np.cos(thetas[:])    #z

    return coords

def flatring(time):
    anglephi = w*time
    flatcoords = np.ones((2,2))
    flatcoords[0,0] = R*np.cos(anglephi)
    flatcoords[0,1] = R*np.sin(anglephi)
    flatcoords[1,0] = -R*np.cos(anglephi)
    flatcoords[1,1] = -R*np.sin(anglephi)
    return flatcoords


#Kinetic and Potential Energies, L function
def Tfun(theta, dottheta):
    return 0.5*m*R**2*(dottheta**2 + w**2*(np.sin(theta))**2)

def Vfun(theta):
    return m*g*R*np.cos(theta)

def Lfun(theta,dottheta):
    return Tfun(theta,dottheta) - Vfun(theta)

T = np.ones((len(t)))
V = np.ones((len(t)))
L = np.ones((len(t)))
T[:] = Tfun(sol[:,0], sol[:,1])
V[:] = Vfun(sol[:,0])
L[:] = Lfun(sol[:,0], sol[:,1])
maxL = max(L)                               #maximum value found in the T,V and L arrays. Since V can be negative, the maximum value of L (T-V) will be bigger than the maximum values of T and V individually, so we dont need to compute all 3 



#h function and Energy
def Efun(theta, dottheta):
    return Tfun(theta,dottheta) + Vfun(theta)

def hfun(theta, dottheta):
    return Efun(theta,dottheta) - m*R**2*w**2*(np.sin(theta))**2

E = np.ones((len(t)))
h = np.ones((len(t)))
E[:] = Efun(sol[:,0], sol[:,1])
h[:] = hfun(sol[:,0], sol[:,1])
maxE = max(E)
minh = min(h)


#Plotting in Crtesian Coordinates
fig = plt.figure(figsize=(14,10))
ejes = fig.add_subplot(222, projection='3d')      #3D movement
ejes2 = fig.add_subplot(224)                      #2D movement
ejes3 = fig.add_subplot(221)                      #T, V, L
ejes4 = fig.add_subplot(223)                      #E, h

ejes.scatter(x[0],y[0],z[0],marker='o')           #3D movement

ejes2.plot(x[0], y[0], 'ro')                      #2D movement

ejes3.plot(t[0], T[0])                            #evolutions of T, V and L
ejes3.plot(t[0], V[0])
ejes3.plot(t[0], L[0])
ejes3.set_xlim(0,t[-1])
ejes3.set_ylim(-1.1*maxL, 1.1*maxL)

ejes4.plot(t[0], E[0])                            #evolutions of E and h
ejes4.plot(t[0], h[0])
ejes4.set_xlim(0,t[-1])
ejes4.set_ylim(-1.1*maxE, 1.1*maxE)

#Animating the trajectory in 3D for every time in the t array
for step in range(1,nsteps):
    ejes.cla()
    ejes2.cla()
    ejes3.cla()
    ejes4.cla()
    
    #ejes1
    ejes.scatter(x[step],y[step],z[step],marker='o',c='r', linewidths = 4)  #plotting position for time t[step]
    ringcoords = ring(t[step])
    ejes.scatter(ringcoords[:,0], ringcoords[:,1], ringcoords[:,2],marker='_',c='b')
    
    #ejes2
    ejes2.plot(x[step], y[step], 'ro', markersize = 11)
    ejes2.plot(0,0,'kP')
    flatringcoords = flatring(t[step])
    ejes2.plot(flatringcoords[:,0],flatringcoords[:,1],'bo',linestyle =':',linewidth=2)
    
    #ejes3
    ejes3.plot(t[step], L[step], 'bo', label='L')
    ejes3.plot(t[step], T[step], 'mo', label='T')
    ejes3.plot(t[step], V[step], 'go', label='V')
    ejes3.plot(t[:],T[:], 'm')
    ejes3.plot(t[:],V[:], 'g')
    ejes3.plot(t[:],L[:], 'b')
    ejes3.legend()
    
    #ejes4
    ejes4.plot(t[step], E[step], 'yo', label='Energy')
    ejes4.plot(t[step], h[step], 'o', color='purple', label='h function')
    ejes4.plot(t[:], E[:], 'y')
    ejes4.plot(t[:], h[:], 'purple')
    ejes4.legend()
    
    #axis limits for every graph
    ejes.set_xlim(-1.2*R,1.2*R)
    ejes.set_ylim(-1.2*R,1.2*R)
    ejes.set_zlim(-1.2*R,1.2*R)
    ejes.set_xlabel('m')
    ejes.set_ylabel('m')
    ejes.set_zlabel('m')

    
    ejes2.set_xlim(-1.2*R,1.2*R)
    ejes2.set_ylim(-1.2*R,1.2*R)
    ejes2.set_xlabel('m')
    ejes2.set_ylabel('m')
    
    ejes3.set_xlim(t[step]-tmax/15,t[step]+tmax/10)
    ejes3.set_ylim(-1.0*maxL, 1.1*maxL)
    ejes3.set_xlabel('Time (s)')
    ejes3.set_ylabel('Energy (J)')
    
    ejes4.set_xlim(t[step]-tmax/15,t[step]+tmax/10)
    ejes4.set_ylim(1.1*minh, max(1.1*min(E),0.9*max(E)))
    ejes4.set_xlabel('Time (s)')
    ejes4.set_ylabel('Energy (J)')
    
    plt.pause(tp)
    
    
    
    









































    












































