import matplotlib.pyplot as plt
import numpy as np


import subprocess
import shlex

p = 100     #Perimeter of window is constant, can have any value

def f(x):
    return (3*x*p)-(12+(5*np.pi/2))*(x**2)  #Function for light emitted from window
def df(x):
    return (3)*p - (24+(5*np.pi))*x

#For maxima using gradient ascent
cur_x = 0.5
alpha = 0.0001 
precision = 0.000001 
previous_step_size = 1
max_iters = 1000000
iters = 1000

#Gradient ascent calculation
while (previous_step_size > precision) & (iters < max_iters) :
    prev_x = cur_x             
    cur_x += alpha * df(prev_x)   
    previous_step_size = abs(cur_x - prev_x)   
    iters+=1  
    
print('val:',cur_x)
max_val = f(cur_x)
print("<Maximum value of f(x) is", max_val, "at","x =",cur_x)

y1 = (p-(4+np.pi)*cur_x)/4    #y (Breadth in terms of length and perimeter)
print(y1)

r = round((cur_x/y1),3)
print(r)     # (Length/Breadth) 

print(round(6/(6+np.pi),3))  #geometric solution

if(r == round(6/(6+np.pi),3)):
    print("The ratio for the sides of the rectangle so that the window transmits the maximum light is", r)

#Plotting f(x)
x=np.linspace(0,100,100)
y=f(x)
label_str = "$3xP-(12+5(pi)/2)x^2$"
plt.plot(x,y,label=label_str)
#Labelling points
plt.plot(cur_x,max_val,'o')
plt.text(cur_x, max_val,f'P({cur_x:.4f},{max_val:.4f})')

plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid()
plt.legend()
#plt.show()
plt.savefig('/sdcard/Download/optm/opfig.pdf')
subprocess.run(shlex.split("termux-open /sdcard/Download/optm/opfig.pdf"))
