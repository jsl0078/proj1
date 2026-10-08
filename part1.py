# -*- coding: utf-8 -*-
"""
proj 1 part 1 

it is what it is, i like popcorn and fried chicken & fried rice

@author: jonniboi
"""

import numpy as np
import scipy.sparse as sp 
import scipy.sparse.linalg as spla 
import matplotlib.pyplot as plt 
import pandas as pd


# parameters 
#L & W will be same size (12cm) 

L= 12e-02 
W = 12e-02
g= 0.0005 #0.5mm spacing of the sense plates
d= 1e-3 #shutter-to-sense-plate spacing

area = L*W

E_o= 100 #V/m, stated
eps_o= 8.854187817e-12 
v = 5  # shutter vel

tia_gain = 100000  # TIA 

time_sweep = (L + g) / v  # time it takes for one stroke

q_s = -eps_o * E_o   #q surface                     
q_plate = eps_o * E_o * area #q plate            

def charges(t):
    A1 = W * np.minimum(v * t, L)   #from hand analysis                                 
    A2 = W * (L - np.minimum(np.maximum(v * t - g, 0), L))          
    return -eps_o * E_o * A1, -eps_o * E_o * A2

t1 = np.linspace(0, time_sweep, 4821)  # transient, 0 to time_sweep in 5us lin step 
Q1, Q2 = charges(t1) #plate 1 plate 2 

i1 = np.gradient(Q1, t1)   #for plate 1                
i2 = np.gradient(Q2, t1) #for plate 2       
V_out = tia_gain * (i2 - i1) #TIA 

I_o = eps_o*E_o*W*v   # max current 
V_peak = 2* I_o* tia_gain 


print(f"q_surface  = {q_s:.4e} C/(m^2)")
print(f"q_plate = {q_plate*1e12:.4f} pC")
print(f"time_sweep = {time_sweep:.4f} S")

print(f"I_o = {I_o:.16f} A")

print(f"V_out= {V_peak:.9f} V")



#------------------------------------------------------------------------------

#figuresssssssssssssssssssssssssssssssssssssssssssssssssssssssss

plt.rcParams.update({"font.size": 14})

fig, ax = plt.subplots(figsize=(7, 3.5)) 
ax.plot(t1 * 1e3, Q1 * 1e12, label="Q1") #convert to millisec 
ax.plot(t1 * 1e3, Q2 * 1e12, "--", label="Q2")
ax.set_xlabel("t (ms)"); ax.set_ylabel("Q (pC)")
ax.set_title("plate charges for 1 stroke")
ax.legend(); fig.tight_layout(); fig.savefig("Tequila_part1_charge.png", dpi=150)

tP = np.concatenate([t1, time_sweep + t1])            
VP = np.concatenate([V_out, -V_out[::-1]])

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(tP * 1e3, VP * 1e3, "k")
ax.set_xlabel("t (ms)"); ax.set_ylabel(r"$V_{out}$ (mV)")
ax.set_title(f"TIA output")
fig.tight_layout(); fig.savefig("BudLight_output.png", dpi=150)
print ("\n")
print("images are saved after compiled output")