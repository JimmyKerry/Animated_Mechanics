# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 18:12:30 2024

@author: 34648
"""

import numpy as np 
import matplotlib.pyplot as plt



#Simulation parameters
nsteps = 1001                     #number of steps in the time array
tmax = 50                         #amount of seconds that will be simulated
tp = tmax/nsteps                           #pause time between steps of the animation
t = np.linspace(0,tmax,nsteps)

#Definition of initial parameters
global m             #mass of one of the three mases (same for all)
global r             #radius of the ring
global k             #hooke constant for the springs (same for all)
global phi1          #initial offset angle for spring 1
global phi2          #initial offset angle for spring 2 
global phi3          #initial offset angle for spring 3
global c1            #intensity coefficient c1 for the first mode of oscillation (eta_i == c1*eta_i1 + c2*eta_i2 + c3*eta_i3)
global c2            #intensity coefficient c2 for the second mode of oscillation (eta_i == c1*eta_i1 + c2*eta_i2 + c3*eta_i3)
global c3            #intensity coefficient c3 for the third mode of oscillation (eta_i == c1*eta_i1 + c2*eta_i2 + c3*eta_i3)

global x01           #equilibrium position for the first mass  (ANGLE IN RADIANS)
global x02           #equilibrium position for the second mass (ANGLE IN RADIANS)
global x03           #equilibrium position for the third mass  (ANGLE IN RADIANS)

default_flag = bool(int(input('\n-- (1) -- Use default values for the initial conditions of the system\n\tor\n-- (0) -- Manually set initial conditions \n\t(1 or 0): ')))

if default_flag==True:
    m = 1.5
    r = 0.5
    k = 20
    phi1 = 0
    phi2 = 0
    phi3 = 0
    c1 = 1/np.sqrt(3)            #1/np.sqrt(3)
    c2 = 1/np.sqrt(3)
    c3 = 1/np.sqrt(3)
    print('\n----- Initial Conditions of the System (Defaults) -----')
    print('\n\tMass of the particles (kg):', m)
    print('\tRadius of the ring (m):', r)
    print('\tHooke constant for all springs (N/m):', k)
    print('\tInitial offset angle for spring 1 (radians): \u03C0·', phi1)
    print('\tInitial offset angle for spring 2 (radians): \u03C0·', phi2)
    print('\tInitial offset angle for spring 3 (radians): \u03C0·', phi3)
    print('\tIntensity coefficient c1 for the first oscillating mode (eta_i == c1*eta_i1 + c2*eta_i2 + c3*eta_i3):', "%.3f"%c1)
    print('\tIntensity coefficient c2 for the second oscillating mode (eta_i == c1*eta_i1 + c2*eta_i2 + c3*eta_i3):', "%.3f"%c2)
    print('\tIntensity coefficient c3 for the third oscillating mode (eta_i == c1*eta_i1 + c2*eta_i2 + c3*eta_i3):', "%.3f"%c3)
else:
    print('\n----- Initial Conditions of the System (Manual Settings) -----')
    m = float(input('\n\tMass of the particles (kg): '))             
    r = float(input('\tRadius of the ring (m): '))                        
    k = float(input('\tHooke constant for all springs (N/m): '))             
    phi1 = float(input('\tInitial offset angle for spring 1 (radians): \u03C0·'))*np.pi                 
    phi2 = float(input('\tInitial offset angle for spring 2 (radians): \u03C0·'))*np.pi    
    phi3 = float(input('\tInitial offset angle for spring 3 (radians): \u03C0·'))*np.pi     
    c1 = float(input('\tIntensity coefficient for the first oscillating mode c1 (eta_i == c1*eta_i1 + c2*eta_i2 + c3*eta_i3):'))
    c2 = float(input('\tIntensity coefficient for the second oscillating mode (eta_i == c1*eta_i1 + c2*eta_i2 + c3*eta_i3):'))
    c3 = float(input('\tIntensity coefficient for the third oscillating mode (eta_i == c1*eta_i1 + c2*eta_i2 + c3*eta_i3):'))

#checking if the coefficients c1 c2 and c3 are normalized
if (c1**2)+(c2**2)+(c3**2)>1.01:
    print('\n\t---------------------------------------\n \
\tWARNING! Coefficients c1 and c2 are not normalized! \n\tModelled amplitudes will be larger than they should    \
\n\t---------------------------------------\n')
elif (c1**2)+(c2**2)+(c3**2)<0.99 and (c1**2)+(c2**2)+(c3**2)>0:
    print('\n\t---------------------------------------\n \
\tWARNING! Coefficients c1 and c2 are not normalized! \n\tModelled amplitudes will be shorter than they should    \
\n\t---------------------------------------\n')
elif (c1**2)+(c2**2)+(c3**2)==0:
    print('\n\t---------------------------------------\n \
\tC1=C2=0 -> The result is the position of equilibrium     \
\n\t---------------------------------------\n')


#equilibrium positions    (the Reference Frame is on the center of the ring)
x01 = 0*np.pi         #for mass 1
x02 = 1/3*np.pi       #for mass 2
x03 = 2/3*np.pi       #for mass 3

def computex(eta, xeq):
    '''
    Parameters
    ----------
    eta : normal mode for the oscillator.
    xeq : value of x for which the oscillator is at its equilibrium position

    Returns
    The coordinate of the oscillator at any given time x(t) = eta(t) + xeq
    '''
    return eta + xeq


#Amplitudes, frequencies and eta coordinates
    # the amplitudes for both oscillators are related; defining only three amplitudes (a11 == a1 , a22 == a2, a23 == a3) is enough to solve the problem. They have to be normalized so they can't be an input like other values and instead have the fixed values given here
a1 = 1/np.sqrt(3*m)
a2 = 1/np.sqrt(2*m)
a3 = 1/np.sqrt(6*m)

    # three oscillators -> three normal frequencies, w1, w2 and w3. w2 and w3 end up being the same after solving the eigenvalue/vector problem
w1 = 0       #this frequency implies a symmetry under translation, which in turn implies the conservation of linear momentum
w2 = np.sqrt(3*k/m)
w3 = np.sqrt(3*k/m)


    # eta coordinates are by definition eta_i == x_i - x_0i , and are obtained by solving the eigenvalue/vector problem by hand, resulting in the following expressions:
    # etas are already arrays, and are linear combinations of eta11, eta12, eta21 and eta22 

eta1 = c1*a1*np.sin(phi1) - c3*2*a3*np.sin(w3*t+phi3)                                       #first mode
eta2 = c1*a1*np.sin(phi1) + c2*a2*np.sin(w2*t+phi2) + c3*a3*np.sin(w3*t+phi3)               #second mode
eta3 = c1*a1*np.sin(phi1) - c2*a2*np.sin(w2*t+phi2) + c3*a3*np.sin(w3*t+phi3)               #third mode

#coordinates of the oscillating masses and of the center of mass
    #Divide by r because eta_i == r*(x_i), x_i are angles, not distances
x1 = computex(eta1,x01) / r                     
x2 = computex(eta2,x02) / r
x3 = computex(eta3,x03) / r



#check if the masses will collide
collide_flag=False
for i in range(len(x1)):
    if abs(x1[i]-x2[i])<0.05:
        collide_flag=True
        print('\n\t---------------------------------------\n \
\tWARNING! With these parameters the masses will collide!\n\tThis program does not model collisions, only the behavior of the system as a coupled oscillator \
\n\t---------------------------------------\n')
        break
    if abs(x1[i]-x3[i])<0.05:
        collide_flag=True
        print('\n\t---------------------------------------\n \
\tWARNING! With these parameters the masses will collide!\n\tThis program does not model collisions, only the behavior of the system as a coupled oscillator \
\n\t---------------------------------------\n')
        break
    if abs(x2[i]-x3[i])<0.05:
        collide_flag=True
        print('\n\t---------------------------------------\n \
\tWARNING! With these parameters the masses will collide!\n\tThis program does not model collisions, only the behavior of the system as a coupled oscillator \
\n\t---------------------------------------\n')
        break


#Animation
    #The animation is donde via a loop where each iteration generates one frame of the animation

#setting up the plots
fig = plt.figure(figsize=(12,8))
ejes1 = fig.add_subplot(121, projection='polar')       #the animated movement
ejes2 = fig.add_subplot(122)

ejes1.set_rmax(1.5*r)
ejes2.set_ylim(-2*np.pi,2*np.pi)
ejes2.set_xlim(t[0]-tmax/15,t[0]+tmax/10)


fig.show()


step = 0
while step < nsteps:
    
    ejes1.cla()
    ejes2.cla()
    
    #ejes1 (animated ring)

    ejes1.plot(np.linspace(x1[step],x2[step],25), r*np.ones(25), 'r.')                   #first spring
    ejes1.plot(np.linspace(x2[step],x3[step],25), r*np.ones(25), 'g.')                   #second
    ejes1.plot(np.linspace(x3[step],2*np.pi+x1[step],25), r*np.ones(25), 'b.')           #third    
    ejes1.plot(x1[step], r, 'ro', markersize=18)                                         #first particle
    ejes1.plot(x2[step], r, 'go', markersize=18)                                         #second 
    ejes1.plot(x3[step], r, 'bo', markersize=18)                                         #third
    
    ejes1.set_title('Simulated movement', fontsize=20)
    ejes1.set_rmax(1.5*r)
    
    
    #ejes2 (theta_i over time)
    
    ejes2.plot(t, x1, 'r-', linewidth=2)                                       #curva x1(t)
    ejes2.plot(t, x2, 'g-', linewidth=2)                                       #curva x2(t)
    ejes2.plot(t, x3, 'b-', linewidth=2)                                       #curva x3(t)
    ejes2.plot(t[step], x1[step], 'ro', label='x1', markersize=10)             #marcador de x1
    ejes2.plot(t[step], x2[step], 'go', label='x2', markersize=10)             #marcador de x2
    ejes2.plot(t[step], x3[step], 'bo', markersize=10, label='x3')             #marcador de x3
              
    ejes2.set_title('Angles over time', fontsize=20)
    ejes2.set_ylim(-2*np.pi,2*np.pi)
    ejes2.set_xlim(t[step]-tmax/15,t[step]+tmax/10)
    ejes2.set_xlabel('Time (s)', fontsize=16)
    ejes2.set_ylabel('Angle (radians)', fontsize=16)
    ejes2.legend()


    plt.pause(tp)
    step += 1




























