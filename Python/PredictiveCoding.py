

"""This code compares the Euler method vs. analytical
solution for:
    1) The supervised rule alone.
    2) The predictive rule alone.
    3) The combined supervised+predictive rule"""




#Libraries
import numpy as np
from matplotlib import pyplot as plt





#%% SuperVised Rule 

#Parameters
y = 1
num_iter = 1000
g = 0.5
w0 = 0
tauw = 0.01


#Euler method
w_var = []
w = w0
for i in range(num_iter):
    dw = y * (y-w)
    w = w + tauw*dw
    w_var.append(w)
    

#Analytical Solution
y_var = []

for i in range(num_iter):
    
   
    yp = np.exp(-(y*i*tauw)) * (y*(np.exp(y*i*tauw) - 1 + w0))
    y_var.append(yp)


    
plt.plot(y_var, color = 'red',  linewidth=5, label='Wolfram Solution')
plt.plot(w_var, color = 'blue', label='Custom Iteration')
plt.title('Supervised Rule')
plt.legend()
plt.show()
#%%Predictive Rule 

#Parameters
y = 1
s = 0.11
num_iter = 1000
tauw = 0.01
g = 0.5
w = 0.2
b = 0.1
bhat = 0.8
c1 = bhat
c2 = w

#Euler's method
w_var = []

for i in range(num_iter):
    dw = -s * (w - bhat)
    d_bhat = -bhat + w
    w += tauw* dw   
    bhat = bhat + tauw*d_bhat
    
    w_var.append(w)
    



#Analytical Solution
wt_var = []


for i in range(num_iter):
    wt = c2*(s*np.exp(-1*(s+1)*i*tauw)+1)/(s+1) - c1*(s*(np.exp(-1*(s+1)*i*tauw)-1))/(s+1)
    
   
    wt_var.append(wt)
    
   
    

plt.plot(wt_var, color = 'red', label = 'Wolfram Solution', linewidth = 6)
plt.plot(w_var, color = 'blue', label = 'Custom Iteration')
plt.title('Predictive Rule')
plt.legend()
plt.show()
#%% Supervised + Predictive Rule 

#Parameters
y = 15
g = 15
num_iter = 1000
tauw = 0.001
taub = 0.01
s = 0.01
w = 0.01
b = 0.1
bhat = 0.8
c1 = bhat
c2 = w

#Euler's method
w2_var = []

for i in range(num_iter):
    dw = g * (y-w) - s * (w - bhat)           
    d_bhat = -bhat + w
    bhat += tauw*d_bhat
    w += tauw*dw
       
    w2_var.append(w)
    
    
#Analytical solution
wtps = []

for i in range(num_iter):
    wt = (np.exp(-g*i*tauw) * (g*(np.exp(g*i*tauw) - 1))) + (c2*(s*np.exp(-1*(s+1)*i*tauw)+1)/(s+1) - c1*(s*(np.exp(-1*(s+1)*i*tauw)-1))/(s+1))
    wtps.append(wt)
    
    ###Supervised + predictive (wt) for wolfram solution
plt.plot(wtps, color = 'red', label = 'Wolfram Solution', linewidth = 6)
plt.plot(w2_var, color = 'blue', label = 'Custom Iteration')
plt.title('Supervised + Predictive')
plt.legend()
plt.show()















