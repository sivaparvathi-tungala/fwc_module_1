import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
#sys.path.insert(0,'/storage/emulated/0/github/cbse-papers/CoordGeo')         #path to my scripts
sys.path.insert(0,'/home/user/circle/CoordGeo')


#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if


#Standard basis vectors
e1 = np.array((1,0)).reshape(2,1)
e2 = np.array((0,1)).reshape(2,1)

#Input parameters
r  = 1
h = np.array((-2,0)).reshape(2,1)
V = np.eye(2)
r1 = 1/3
h1 = np.array(([-4/3,0]))#.reshape(2,1)
r2 = 3
h2 = np.array(([4,0]))#.reshape(2,1)
u = np.zeros((2,1))
f =-r**2
#f1 = -r1**2
#f2 = -r2**2
S = (V@h+u)@(V@h+u).T-(h.T@V@h+2*u.T@h+f)*V
print("s matrix:",S)
##Centre and point 
O = u.T
#Intermediate parameters
f0 = np.abs(f+u.T@LA.inv(V)@u)
#f_1 = np.abs(f+h1.T@LA.inv(V)@h1)
#f_2 = np.abs(f+h2.T@LA.inv(V)@h2)
#Eigenvalues and eigenvectors
D_vec,P = LA.eig(S)
lam1 = D_vec[0]
lam2 = D_vec[1]
p1 = P[:,1].reshape(2,1)
p2 = P[:,0].reshape(2,1)
D = np.diag(D_vec)
print("eigen values:",D_vec)
print("eigen vectors:",P)
t1= np.sqrt(np.abs(D_vec))
negmat = np.block([e1,-e2])
t2 = negmat@t1

#Normal vectors to the conic
n1 = P@t1
n2 = P@t2
print("normal of tangent n1 :",n1,n2)

#kappa
den1 = n1.T@LA.inv(V)@n1
den2 = n2.T@LA.inv(V)@n2

k1 = np.sqrt(f0/(den1))
k2 = np.sqrt(f0/(den2))
print(k1,k2)

q1 = LA.inv(V)@((-k1*n1-u.T).T)
q2 = LA.inv(V)@((-k2*n2-u.T).T)
print("poc : ",q1,q2)
#
##Generating all lines
xhq1 = line_gen(h,q1)
xhq2 = line_gen(h,q2)


m1=h-q1
m2=h-q2

def line_dir_pt(m,A,k1,k2):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB

x_KD=line_dir_pt(m1,q1,1,-4)
x_KS=line_dir_pt(m2,q2,1,-4)

##Generating the circle
x_circ= circ_gen(O,r)
x1_circ = circ_gen(h1,r1)
x2_circ = circ_gen(h2,r2)
#
##Plotting all lines
plt.plot(x_KD[0,:],x_KD[1,:],label='$T-1$')
plt.plot(x_KS[0,:],x_KS[1,:],label='$T-2$')
#
#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$C$')
plt.plot(x1_circ[0,:],x1_circ[1,:],label='$C-1$')
plt.plot(x2_circ[0,:],x2_circ[1,:],label='$C-2$')
#
#
#Labeling the coordinates
tri_coords = np.vstack((h.T,q1.T,q2.T,O,h1,h2)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['h','q1','q2','O','h1','h2']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axhline(y=0,color='black')
plt.axvline(x=0,color='black')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
#
#if using termux
#plt.savefig('/sdcard/Download/circle/cfig.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/Download/circle/cfig.pdf"))
#else
plt.show()
#
#
#
#
#
#
