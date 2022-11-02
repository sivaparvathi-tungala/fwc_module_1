import numpy as np
from pylab import *
from sympy import *

import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sys                           #for path to external scripts
#sys.path.insert(0,'/sdcard/Download/line/CoordGeo')
omat=np.array([[0,1],[-1,0]])
def dir_vec(A,B):
   return B-A
 

def norm_vec(A,B):
   return np.matmul(omat, dir_vec(A,B))

#Generate line points
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

def line_dir_pt(m,A,k1,k2):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB

#local imports
#from line.funcs import *
#from triangle.funcs import *
#from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#input parameters
P = np.array(([1,2]))
n1 = np.array(([3,4]))
n2 = np.array(([4,-3]))
D = np.array(([5,15]))
v = np.block([[n1.T],[n2.T]])
o = np.linalg.inv(v)
A = o@D                             #intersection point
B = omat@n1  #direction vector of AB
C = omat@n2     # direction vector of AC
s = np.array([[1,0],[0,-1]]) 
B=s@B
C=s@C
M = C - B    # direction vector of BC
N = B + C    #direction vector of BC
O = omat@M    #normal vector of m
E = omat@N    #normal vector of n


k1 = 0.5
k2 = -0.5
x_BA = line_dir_pt(M,P,k1,k2)
k1 = 0.5
k2 = -0.5
x_CA = line_dir_pt(N,P,k1,k2)

#Generating all lines
x_AB = line_gen(A,B)
x_AC = line_gen(A,C)
#x_BA = line_gen(B,A)
#x_CA = line_gen(C,A)
#x_PM = line_gen(P,M)
#x_PN = line_gen(P,N)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])#,label='$Diameter$')
plt.plot(x_AC[0,:],x_AC[1,:])#,label='$Diameter$')
plt.plot(x_BA[0,:],x_BA[1,:])#,label='$Diameter$')
plt.plot(x_CA[0,:],x_CA[1,:])#,label='$Diameter$')
#plt.plot(x_PM[0,:],x_PM[1,:])#,label='$Diameter$')
#plt.plot(x_PN[0,:],x_PN[1,:])#,label='$Diameter$')


#Labeling the coordinates
tri_coords = np.vstack((A,B,C,P)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','P']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/Download/line/fig.pdf')
subprocess.run(shlex.split("termux-open /sdcard/Download/line/fig.pdf"))
#else
#plt.show()
