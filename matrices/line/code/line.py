import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

n1 = np.array([3,4]).reshape(2,1)
n2 = np.array([4,-3]).reshape(2,1)
c1 = 5
c2 = 15
P = np.array([1,2]).reshape(2,1)

c = np.array([c1,c2]).reshape(2,1)
A = LA.inv(np.vstack((n1.T,n2.T)))@c
#print(A)

omat = np.array([[0,-1],[1,0]])
m1 = omat@n1/LA.norm(n1)
m2 = omat@n2/LA.norm(n2)

k1 = (LA.norm(n2)*n1-LA.norm(n1)*n2).T@(A-P)/n2.T@omat@n1
k2 = -(LA.norm(n2)*n1+LA.norm(n1)*n2).T@(A-P)/n2.T@omat@n1

#print(k1, k2)

B1 = A + k1*m1
C1 = A + k1*m2
v1 = omat@(B1-C1)
w1 = v1.T@P
print(B1, C1)
print(v1,w1)

B2 = A + k2*m1
C2 = A - k2*m2
v2 = omat@(B2-C2)
w2 = v2.T@P
#print(B2, C2)
#print(v2,w2)


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

#line_gen
x_AB1=line_gen(A,B1)
x_AB2=line_gen(A,B2)
x_AC1=line_gen(A,C1)
x_AC2=line_gen(A,C2)
x_B1C1=line_gen(B1,C1)
x_B2C2=line_gen(B2,C2)

#Plotting line
plt.plot(x_AB1[0,:],x_AB1[1,:],color='red')
plt.plot(x_AB2[0,:],x_AB2[1,:],color='black')
plt.plot(x_AC1[0,:],x_AC1[1,:],color='blue')
plt.plot(x_AC2[0,:],x_AC2[1,:],color='orange') 
plt.plot(x_B1C1[0,:],x_B1C1[1,:],color='green')
plt.plot(x_B2C2[0,:],x_B2C2[1,:],color='brown')

#Labeling the coordinates                        
tri_coords = np.vstack((A.T,B1.T,B2.T,C1.T,C2.T)).T                
plt.scatter(tri_coords[0,:], tri_coords[1,:])      
vert_labels = ['B1','B2','C1','C2']             
for i, txt in enumerate(vert_labels):                   
    plt.annotate(txt, # this is the text                           
            (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
            textcoords="offset points", # how to position the text                                     
            xytext=(0,5), # distance from text to points (x,y)                                           
            ha='center') # horizontal alignment can be left, right or center
 
plt.xlabel('$ X $')
plt.ylabel('$ Y $')
 #plt.legend(loc='best')                           
plt.grid() # minor                                
plt.axis('equal')
 #plt.title('Orthocenter of Triangle')
 #if using termux
 #plt.savefig('/sdcard/fwc/line/fig.pdf')
plt.show()
