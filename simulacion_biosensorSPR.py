# -*- coding: utf-8 -*-
"""
Created on Tue Apr 9 10:51:07 2024

@author: 34648
"""

import numpy as np
import matplotlib.pyplot as plt
import keyboard

global r      #el radio de los rayos de luz en la animacion
r=10
#constantes dielectricas
er_dielmin = 5
er_dielmax = 4

er_diel = np.linspace(er_dielmax, er_dielmin, 12)
er_gold = 6
er_prism = 3.25

#angulo de resonancia (formula del paper)
theta_res =  360/(2*np.pi) * np.arcsin(np.sqrt( (er_gold*er_diel[:]) / (er_gold+er_diel[:]) )  /  np.sqrt(er_prism))

#funciones para generar las curvas de intensidad en funcion del angulo de resonancia
#en las graficas hay que poner xdiracdelta(angulo_resonancia) para los putnos del eje x, y luego diracdelta() para el eje y
def ydiracdelta(a=1/1.25, n=100):
    '''
    Returns
    -------
    Devuelve los puntos de una casi delta de dirac en el eje y
    '''
    xpuntosdd = np.linspace(-3,3,100)
    return -np.exp(-(xpuntosdd[:]/a)**2) / (abs(a)*np.sqrt(np.pi))

def xdiracdelta(x):
    '''
    Returns
    -------
    Devuelve las posiciones en el eje x alrededor de la cual se centrara la delta de dirac de la funcion anterior
    '''
    return np.linspace(x-3,x+3,100)


#funciones para dibujar los rayos de luz entrantes y salientes
def luzentrantex(theta,grados_deltatheta=2,r=r):
    theta = (2*np.pi)/(360) * theta        #paso a radianes
    x = r*np.cos(theta)
    return [-x,0]

def luzentrantey(theta,grados_deltatheta=2,r=r):
    theta = (2*np.pi)/(360) * theta        #paso a radianes
    y = r*np.sin(theta)
    return [y,0]

def luzsalientex(theta,grados_deltatheta=2,r=r):
    theta = (2*np.pi)/(360) * theta        #paso a radianes
    deltatheta = (2*np.pi)/(360) * grados_deltatheta
    return [0,r*np.cos(theta-deltatheta),r*np.cos(theta+deltatheta),0]

def luzsalientey(theta,grados_deltatheta=2,r=r):
    theta = (2*np.pi)/(360) * theta        #paso a radianes
    deltatheta = (2*np.pi)/(360) * grados_deltatheta
    return [0,r*np.sin(theta-deltatheta),r*np.sin(theta+deltatheta),0]



#representacion
fig = plt.figure(figsize=(18,8))
ejes1 = fig.add_subplot(121)                                                       #la animacion del SPR
ejes2 = fig.add_subplot(122)                                                       #el grafico de la intensidad en funcion del angulo

print('Presionar Espacio para parar la animación')

#particulas a analizar
nparticulas = 50                                                                                                #numero de particulas de analito
xparticulas = np.random.randint(-8,8,size=nparticulas) + 0.1*np.random.randint(-10,10,size=nparticulas)          #posiciones iniciales en x
yparticulas = np.random.randint(-10,-0.42,size=nparticulas) + 0.1*np.random.randint(-10,-0.5,size=nparticulas)        #posiciones iniciales en y
t=0                                                 #un parametro que luego sirve para dibujar el movimiento de estas particulas como sinusoidales
w=0.1*np.random.randint(-10,10,size=nparticulas)    #la frecuencia de dicho movimiento
v=0.03                                              #la velocidad (vertical) del movimiento
xparticulas0 = xparticulas[:]                       #en este array guardamos las posicones x iniciales

for step in range(len(theta_res)):
    
    intensidadx = xdiracdelta(theta_res[step])     #coords x de la gráfica de intensidad vs angulo para el angulo de esta iteracion
    intensidady = ydiracdelta()     #coords y de la gráfica de intensidad vs angulo para el angulo de esta iteracion
    
    framesoscilacion=20      #el numero de frames de animacion que consume cada oscilacion del haz de luz alrededor del angulo theta_res[step]
    #angulos = np.linspace(theta_res[step]+3, theta_res[step]-3, framesoscilacion)
    angulos = np.linspace(np.average(theta_res)+15, np.average(theta_res)-15, framesoscilacion)
    
    for step2 in range(2*len(angulos)):
        ejes1.cla()
        ejes2.cla()
        
        if step2<len(angulos):
            #actualizamos las posiciones de las particulas a analizar
            t += 0.1
            xparticulas = xparticulas0[:] + 1*np.sin(w[:]*t)      
            yparticulas = yparticulas[:] + v
            #comprobamos si algun analito colisiona con la lamina detectora
            for i in range(len(yparticulas)):
                if yparticulas[i]>-0.5:
                    yparticulas[i]=-10
            
            #ejes1
            ejes1.plot([-10,10,10,-10,-10],[0,0,-0.5,-0.5,0],'k-',linewidth=2)                                         #la superficie de detección
            ejes1.plot([-2,0,2,-2],[0,2,0,0],'b-', linewidth=1.5)                                                      #el prisma
            ejes1.plot(luzentrantex(angulos[step2]), luzentrantey(angulos[step2]), 'r-', linewidth=1.5)                #luz incidente
            ejes1.plot(luzsalientex(angulos[step2]), luzsalientey(angulos[step2]), 'r-', linewidth=1.5)                #cono de luz saliente
            ejes1.plot(luzentrantex(np.average(theta_res)) , luzentrantey(np.average(theta_res)), 'k--', linewidth=1)              #línea guía que marca el angulo de theta_res en el que estamos en esta iteracion
            ejes1.text(x=5, y=2, s='\u0394\u03b5 = '+'%.2f'%(abs(min(er_diel)-er_diel[step])),fontsize=20)             #lo que los analitos han cambiado la permitividad del dielectrico en esta iteracion
            ejes1.plot(xparticulas,yparticulas, 'go', markersize=5)                                                    #particulas a analizar
            
            ejes1.set_xlim(-10,10)
            ejes1.set_ylim(-5,10)
            ejes1.set_axis_off()
            
            #ejes2
            ejes2.plot(intensidadx[:round(step2/len(angulos)*len(intensidadx))], intensidady[:round(step2/len(angulos)*len(intensidadx))], 'r-', linewidth=2.5)
            
            ejes2.set_xlim(min(theta_res)-5,max(theta_res)+5)
            ejes2.set_ylim(-1,0.25)
            ejes2.set_xlabel('Ángulo (º)', fontsize=16)
            ejes2.set_ylabel('Intensidad (u.a)',fontsize=16)
            ejes2.set_yticks([])
            
            fig.show()
            plt.pause(0.1)
            if keyboard.is_pressed('space'):
                print('Animación detenida')
                break
            
        
        if step2>len(angulos):
            #actualizamos las posiciones de las particulas a analizar
            t += 0.1
            xparticulas = xparticulas0[:] + 1*np.sin(w[:]*t)      
            yparticulas = yparticulas[:] + v
            #comprobamos si algun analito colisiona con la lamina detectora
            for i in range(len(yparticulas)):
                if yparticulas[i]>-0.5:
                    yparticulas[i]=-10
            
            #ejes1
            ejes1.plot([-10,10,10,-10,-10],[0,0,-0.5,-0.5,0],'k-',linewidth=2)                                                                     #la superficie de detección
            ejes1.plot([-2,0,2,-2],[0,2,0,0],'b-', linewidth=1.5)                                                                                  #el prisma
            ejes1.plot(luzentrantex(angulos[-step2+len(angulos)]), luzentrantey(angulos[-step2+len(angulos)]), 'r-', linewidth=1.5)                #luz incidente
            ejes1.plot(luzsalientex(angulos[-step2+len(angulos)]), luzsalientey(angulos[-step2+len(angulos)]), 'r-', linewidth=1.5)                #cono de luz saliente
            ejes1.plot(luzentrantex(np.average(theta_res)) , luzentrantey(np.average(theta_res)), 'k--', linewidth=1)                                          #línea guía que marca el angulo de theta_res en el que estamos en esta iteracion
            ejes1.text(x=5, y=2, s='\u0394\u03b5 = '+'%.2f'%(abs(min(er_diel)-er_diel[step])),fontsize=20)                                         #lo que los analitos han cambiado la permitividad del dielectrico en esta iteracion
            ejes1.plot(xparticulas,yparticulas, 'go', markersize=5)                                                                                #particulas a analizar
            
            ejes1.set_xlim(-10,10)
            ejes1.set_ylim(-5,10)
            ejes1.set_axis_off()
            
            #ejes2
            ejes2.plot(intensidadx, intensidady, 'r-', linewidth=2.5)
            ejes2.plot([theta_res[step], theta_res[step]], [-10,10], 'k--', linewidth=1.5)
            
            ejes2.set_xlim(min(theta_res)-5,max(theta_res)+5)
            ejes2.set_ylim(-1,0.25)
            ejes2.set_xlabel('Ángulo (º)', fontsize=16)
            ejes2.set_ylabel('Intensidad (u.a)', fontsize=16)
            ejes2.set_yticks([])
            
            fig.show()
            plt.pause(0.1)
            if keyboard.is_pressed('space'):
                print('Animación detenida')
                break
            
            
    if keyboard.is_pressed('space'):
        break
    
    plt.pause(0.5)
    
    if keyboard.is_pressed('space'):
        break
    



















































