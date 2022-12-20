import numpy as np
import mpmath as mp
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
from pylab import *
from sympy import *

import sys                                         
sys.path.insert(0,"/home/user/matrices/conic/code/CoordGeo")   
##local imports
#from line.funcs import *
#from triangle.funcs import *
from conics.funcs import circ_gen
from conics.funcs import *


#if using termux
#import subprocess
#import shlex
#end if


#for parabola
a = 1
V = np.array([[0,0],[0,1]])
u = np.array(([-2,0]))
f = 0
V1 = V
u1 = u/2
f1 = f+a**2
print(f1)

#Vertex= np.array(([1,1]))
Focus = np.array(([1,0])) #h

def affine_transform(P,c,x):
    return P@x + c
   # return P1@x + c1

#Transformation 
lamda,P = LA.eigh(V)
if(lamda[1] == 0):  # If eigen value negative, present at start of lamda 
    lamda = np.flip(lamda)
    P = np.flip(P,axis=1)

lamda1,P1 = LA.eigh(V1)
if(lamda1[1] == 0):
    lamda1 = np,flip(lamda1)
    P1 = np.flip(P1,axis=1)

eta = u@P[:,0]
a = np.vstack((u.T + eta*P[:,0].T, V))
b = np.hstack((-f, eta*P[:,0]-u)) 
center = LA.lstsq(a,b,rcond=None)[0]
O = center 
n = np.sqrt(lamda[1])*P[:,0]
c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)
F = np.array(([1,0]))
fl = LA.norm(F)


eta1 = u1@P1[:,0]
a1 = np.vstack((u1.T + eta1*P1[:,0].T, V1))
b1 = np.hstack((-f1, eta1*P1[:,0]-u1)) 
center1 = LA.lstsq(a1,b1,rcond=None)[0]
O1 = center1 
n1 = np.sqrt(lamda1[1])*P1[:,0]
c1 = 0.5*(LA.norm(u1)**2 - lamda1[1]*f1)/(u1.T@n1)
F = np.array(([1,0]))
fl = LA.norm(F)

#pmeters to generate parabola
num_points = 8000
delta = 100*np.abs(fl)/10
p_y = np.linspace(-4*np.abs(fl)-delta,4*np.abs(fl)+delta,num_points)
a = -2*eta/lamda[1]   # y^2 = ax => y'Dy = (-2eta)e1'y


num_points1 = 8000
delta1 = 100*np.abs(fl)/10
p_y1 = np.linspace(-4*np.abs(fl)-delta1,4*np.abs(fl)+delta1,num_points1)
a1 = -2*eta1/lamda1[1]   # y^2 = ax => y'Dy = (-2eta)e1'y

#Generating all shapes
p_x = parab_gen(p_y,a)

p_std = np.vstack((p_x,p_y)).T

p_q = parab_gen(p_y1,a1)
p_std1 = np.vstack((p_q,p_y1)).T

##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T
plt.plot(p[0,:], p[1,:],label='parabola')


p1 = np.array([affine_transform(P1,center1,p_std1[i,:]) for i in range(0,num_points1)]).T
plt.plot(p1[0,:], p1[1,:],label='parabola1')

#Labeling the coordinates
tri_coords = np.vstack((Focus.T,)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['F']
for i, txt in enumerate(vert_labels): 
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i],tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    
#if using termux
#plt.savefig('/sdcard/fwc/conic/fig.pdf')
#subprocess.run(shlex.split("termux-open "+os.path.join(script_dir, fig_relative)))
#else
#plt.legend()
plt.legend(loc='upper left')
plt.xlabel('$ X $')
plt.ylabel('$ Y $')
plt.legend()
plt.grid(True) # minor
plt.axis('equal')
plt.axhline(y=0,color='black')
plt.axvline(x=0,color='red',label='directrix')

plt.show()
