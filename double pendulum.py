# -*- coding: utf-8 -*-
"""
Trajectory finder for a system of coupled oscillators

Let a spring hang from a ceiling on one end and be attached to a small particle of mass m on the other. Let another equal spring be attached to this mass, with another particle of mass m hanging from the lower end of it.
This program resolves the movement of the system by considering it a system of 2 coupled oscillators.
"""

import numpy as np 
import matplotlib.pyplot as plt



#Simulation parameters
nsteps = 1001                     #number of steps in the time array
tmax = 50                         #amount of seconds that will be simulated
tp = tmax/nsteps                           #pause time between steps of the animation
t = np.linspace(0,tmax,nsteps)

#Definition of initial parameters
global m             #mass of one of the two mases (same for both)
global l             #length of one spring (same for both)
global k             #hooke constant for both springs (ame for both)
global phi1          #initial offset angle for spring 1 (the top spring)
global phi2          #initial offset angle for spring 2 (the bottom spring)
global c1            #intensity coefficient for the first oscillator (eta_i == c1*eta_i1 + c2*eta_i2)
global c2            #intensity coefficient for the first oscillator (eta_i == c1*eta_i1 + c2*eta_i2)
global g             #force of gravity (m/s^2)

global x01           #equilibrium position for the top mass
global x02           #equilibrium position for the bottom mass

default_flag = bool(int(input('\n-- (1) -- Use default values for the initial conditions of the system\n\tor\n-- (0) -- Manually set initial conditions \n\t(1 or 0): ')))

if default_flag==True:
    m = 0.5
    l = 0.5
    k = 2.25
    phi1 = 0
    phi2 = 0
    c1 = 1/np.sqrt(2)
    c2 = 1/np.sqrt(2)
    g = 9.81
    print('\n----- Initial Conditions of the System (Defaults) -----')
    print('\n\tMass of the particles hanging from the springs (kg):', m)
    print('\tLength of the pendulums (m):', l)
    print('\tHooke constant for both springs (N/m):', k)
    print('\tInitial offset angle for spring 1 (radians): \u03C0·', phi1)
    print('\tInitial offset angle for spring 2 (radians): \u03C0·', phi2)
    print('\tIntensity coefficient for the first oscillating mode (eta_i == c1*eta_i1 + c2*eta_i2):', c1)
    print('\tIntensity coefficient for the second oscillating mode (eta_i == c1*eta_i1 + c2*eta_i2):', c2)
    print('\tGravitational Field Strength (kgm/s^2): ', g)
else:
    print('\n----- Initial Conditions of the System (Manual Settings) -----')
    m = float(input('\n\tMass of the particles hanging from the springs (kg): '))             
    l = float(input('\tLength of the pendulums (m): '))                        
    k = float(input('\tHooke constant for both springs (N/m): '))             
    phi1 = float(input('\tInitial offset angle for spring 1 (radians): \u03C0·'))*np.pi                 
    phi2 = float(input('\tInitial offset angle for spring 2 (radians): \u03C0·'))*np.pi     
    c1 = float(input('\tIntensity coefficient for the first oscillating mode c1 (eta_i == c1*eta_i1 + c2*eta_i2):'))
    c2 = float(input('\tIntensity coefficient for the second oscillating mode (eta_i == c1*eta_i1 + c2*eta_i2):'))
    g = float(input('\tGravitational Field Strength (kgm/s^2): '))

#checking if the coefficients c1 c2 are normalized
if (c1**2)+(c2**2)>1.01:
    print('\n\t---------------------------------------\n \
\tWARNING! Coefficients c1 and c2 are not normalized! \n\tModelled amplitudes will be larger than they should    \
\n\t---------------------------------------\n')
elif (c1**2)+(c2**2)<0.99 and (c1**2)+(c2**2)>0:
    print('\n\t---------------------------------------\n \
\tWARNING! Coefficients c1 and c2 are not normalized! \n\tModelled amplitudes will be shorter than they should    \
\n\t---------------------------------------\n')
elif (c1**2)+(c2**2)==0:
    print('\n\t---------------------------------------\n \
\tC1=C2=0 -> The result is the position of equilibrium     \
\n\t---------------------------------------\n')
    



#equilibrium positions    (the Reference Frame is on the surface where the top spring is attached, x is the vertical coordinate)
x01 = 2*m*g/k + l         #for the top mass
x02 = 3*m*g/k + 2*l       #for the bottom mass

def computex1(eta1, xeq1 = x01):
    '''
    Parameters
    ----------
    eta1 : normal mode for the top oscillator.
    xeq1 : value of x for which the first oscillator is at its equilibrium position

    Returns
    The coordinate of the top oscillator at any given time x(t) = eta1(t) + xeq1
    '''
    return eta1 + xeq1 

def computex2(eta2, xeq2 = x02):
    '''
    Same as the function computex1 but for the bottom oscillator
    '''
    return eta2 + xeq2 


#Amplitudes, frequencies and eta coordinates
    # the amplitudes for both oscillators are related; a21 ~ a11 , a22 ~ a12 . Defining only two amplitudes (a11 == a1 and a12 == a2) is enough to solve the problem. They have to be normalized so they can't be an input like other values and instead have the fixed values given here
a1 = np.sqrt(2 / ((5 - np.sqrt(5))*m) )
a2 = np.sqrt(2 / ((5 + np.sqrt(5))*m) )

    # two oscillators -> two normal frequencies, w1 and w2
w1 = np.sqrt(k * ((3 + np.sqrt(5)) / (2*m) ))
w2 = np.sqrt(k * ((3 - np.sqrt(5)) / (2*m) ))


    # eta coordinates are by definition eta_i == x_i - x_0i , and are obtained by solving the eigenvalue/vector problem by hand, resulting in the following expressions:
    # etas are already arrays, and are linear combinations of eta11, eta12, eta21 and eta22 

eta1 = c1*a1*np.sin(w1*t+phi1) + c2*a2*np.sin(w2*t+phi2)                                       #top oscillator
eta2 = c1*(1-np.sqrt(5))/2*a1*np.sin(w1*t+phi1) + c2*(1+np.sqrt(5))/2*a2*np.sin(w2*t+phi2)     #bottom oscillator



#coordinates of the oscillating masses and of the center of mass
x1 = computex1(eta1)                     
x2 = computex2(eta2) 
xCOM = (x1+x2) / 2

#check if the masses will collide
collide_flag=False
for i in range(len(x1)):
    if x1[i]>x2[i]:
        collide_flag=True
        print('\n\t---------------------------------------\n \
\tWARNING! With these parameters the masses will collide!\n\tThis program does not model collisions, only the behavior of the system as a coupled oscillator \
\n\t---------------------------------------\n')
        break


#Animation
    #The animation is donde via a loop where each iteration generates one frame of the animation

#setting up the plots
fig = plt.figure(figsize=(12,8))
ejes1 = fig.add_subplot(121)           #The animated movement
ejes2 = fig.add_subplot(122)           #x1 and x2 over time

ejes1.set_xlim(-2,2)
ejes1.set_ylim(-1.1*max(x2),0)

ejes2.set_xlim(t[0]-tmax/15,t[0]+tmax/10)
fig.show()



step = 0
while step < nsteps:
    
    ejes1.cla()
    ejes2.cla()
    
    #ejes1 (animated pendulum)
    
    ejes1.plot(0, -x1[step], 'ro', markersize=10, label='$m_1$')                #top mass
    ejes1.plot(0, -x2[step], 'go', markersize=10, label='$m_2$')                #bottom mass
    ejes1.plot(np.zeros(shape=20), np.linspace(0,-x1[step],20), 'r.')           #top spring
    ejes1.plot(np.zeros(shape=20), np.linspace(-x1[step],-x2[step],20), 'g.')   #bottom spring
    ejes1.plot([-1e4,1e4],[0,0], 'k-', linewidth = 20)                          #ceiling
    ejes1.text(-1.8,-max(x2),'t = '+str("%.2f"%t[step])+' s', fontsize=18)
    if collide_flag==True:
        ejes1.text(0.5,-max(x2),'Masses will collide! \n(see terminal)', fontsize=12)
    
    ejes1.set_xlim(-2,2)
    ejes1.set_ylim(-1.1*max(x2),0)    
    ejes1.set_ylabel('Height (m)', fontsize=16)
    frame1 = fig.gca()
    for xlabel_i in frame1.axes.get_xticklabels():
        xlabel_i.set_visible(False)
    ejes1.legend()
    
    
    #ejes2 (x(t) and y(t))
    
    ejes2.plot(t, -x1, 'r-', linewidth=2)                                                                               #curva x1(t)
    ejes2.plot(t, -x2, 'g-', linewidth=2)                                                                               #curva x2(t)
    ejes2.plot(t, -xCOM, 'purple', linestyle='-', linewidth=2)                                                          #curva xCOM(t)  (XCOM == posicion del centro de masas)
    ejes2.plot(t[step], -x1[step], 'ro', label='x1', markersize=10)                                                     #marcador de x1
    ejes2.plot(t[step], -x2[step], 'go', label='x2', markersize=10)                                                     #marcador de x2
    ejes2.plot(t[step], -xCOM[step], 'o', mec='purple', mfc='purple', markersize=10, label='x of the Center of Mass')   #marcador de xCOM
    
    ejes2.set_xlim(t[step]-tmax/15,t[step]+tmax/10)
    ejes2.set_ylim(-1.5*max(max(x1),max(x2)) , 0 )
    ejes2.legend()
    
    plt.pause(tp)
    step += 1





















