# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:36:53 2019
@author: kjohri
"""
import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
# %% Fluid selection and properties
fluids = ['air','N2','CO2','MM','D6','toluene']
Tcrit = np.array([])
Pcrit = np.array([])
Rgas  = np.array([])
gamma = np.array([])
# Gas properties
for fluid in fluids:
    Tcrit = np.append(Tcrit,CP.PropsSI('TCRIT',fluid))
    Pcrit = np.append(Pcrit,CP.PropsSI('PCRIT',fluid))
    Rgas  = np.append(Rgas,CP.PropsSI('gas_constant',fluid)/CP.PropsSI('M',fluid))
#    gamma = np.append(gamma,CP.PropsSI('Cpmolar','P',Pcrit[fluid],fluid)/CP.PropsSI('Cvmolar','P',Pcrit[fluid],fluid))
    
# %% Plotting contour plots for gases
def Plot(fluid):
    Tcrit = CP.PropsSI('TCRIT',fluid)
    Pcrit = CP.PropsSI('PCRIT',fluid)
    Pr = np.linspace(0.6,3,100)
    Tr = np.linspace(0.6,3,100)
    Prange = Pr*Pcrit
    Trange = Tr*Tcrit
    T,P = np.meshgrid(Trange,Prange)
    Z     = np.zeros((100,100))
    Gamma = np.zeros((100,100))
    for i in range(0,len(T)):
        Z[:,i]     = CP.PropsSI('Z','P',P[:,i],'T',Trange[i],fluid)
        Gamma[:,i] = CP.PropsSI('FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS','P',P[:,i],'T',Trange[i],fluid)
    plt.figure()
    plt.contourf(T/Tcrit,P/Pcrit,Z,50)
    plt.xlabel('Tr')
    plt.ylabel('Pr')
    plt.title('Compressibility factor (Z) for %s'%(fluid))    
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.contourf(T/Tcrit,P/Pcrit,Gamma,50)
    plt.xlabel('Tr')
    plt.ylabel('Pr')
    plt.title('Fundamental derivative for %s'%(fluid) )    
    plt.colorbar()
    plt.show()
    return Z,Gamma


#
#for fluid in fluids:
#    Plot(fluid)
    
# %% Final selection of fluids


# %% Optimise factor to find values for Z

from scipy.optimize import minimize_scalar
import numpy as np 
import CoolProp.CoolProp as CP

def Zeval(factor, FLUID,Z):
    Z_desired = Z
    fluid = FLUID   
    Tcrit = CP.PropsSI('TCRIT',fluid)
    Pcrit = CP.PropsSI('PCRIT',fluid)
    Tinput = factor*Tcrit
    Z = CP.PropsSI('Z','P',Pcrit,'T',Tinput,fluid)
    error = abs((1- Z/Z_desired))
    return error

FLUID = 'air'
Z = 1
#factor = 0.8
bnds = ((0.5, 5))
res = minimize_scalar(Zeval, bounds=bnds,args=(FLUID,Z,), tol = 1e-3, method='bounded')
Tr = res.x

# %%

fluid = 'N2'   
Tcrit = CP.PropsSI('TCRIT',fluid)
Pcrit = CP.PropsSI('PCRIT',fluid)
factor = Tr
Gamma  = CP.PropsSI('FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS','P',Pcrit,'T',Tcrit*factor,fluid)
Z = CP.PropsSI('Z','P',Pcrit,'T',Tr*Tcrit,fluid)
gamma = CP.PropsSI('Cpmolar','P',Pcrit,'T',Tr*Tcrit,fluid)/CP.PropsSI('Cvmolar','P',Pcrit,'T',Tr*Tcrit,fluid)
# %%Generate grid of data points
# Absolute angle at stator inlet
alpha = np.pi/6.0
# Flow and load coefficient
phi      = np.linspace(0.5,1.3,30)
load     = np.linspace(3,8,30)
phi,load = np.meshgrid(phi,load)
# Degree of reaction
r = (load/4)-(phi*np.tan(alpha))+1
#GAMMA = CP.PropsSI('FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS', 'P',Pcrit[1],'T',T_t*Tcrit[1],fluids[1])