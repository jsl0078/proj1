# -*- coding: utf-8 -*-
"""
proj 1 part 2 

I used spyder w/ mods 

it is what it is, i like popcorn and fried chicken & fried rice

@author: jonniboi
"""

import numpy as np
import scipy.sparse as sp 
import scipy.sparse.linalg as spla 
import matplotlib.pyplot as plt 
import pandas as pd

#paramters ------------------------------------------

#L & W will be same size (12cm) 

L= 12e-02 
W = 12e-02
g= 0.0005 #0.5mm spacing of the sense plates
d= 1e-3 #shutter-to-sense-plate spacing

area = L*W

E_o= 100 #V/m, static electric field 
eps_o= 8.854187817e-12 #vacuum permittivity F/m 
z_core= 0.006 # z region 
core_pad= 0.005 # distance the grid extends past 
                #the outer ends of the two sense plates

X_ps= 0.4 #distance form plate to side wall 
stretch= 1.2 #ratio of cell size outside fine grid region 

tia_gain= 1e5 #TIA gain 

v= 5 #linear shutter velocity m/s 

dx= 0.0001 #grid spacing 

pulse_segments= 2000 

time_sweep= (L+g)/v #time it takes for one stroke

plate1= (0,L) #plate 1 
plate2= (L+g, L*2+g)

X_far    = 0.4    # meters, distance from the plates to side-walls
H        = 0.4     # meters, ~height box region

#---------------------------------------------------
def stretched(h0, r, length): 
    
    pts, i, h = [] , 0, h0 
    while i < length: 
        h *= r 
        i += h 
        pts.append(i)
    return np.array(pts)
    
def grid_space(dx, X_ps, H, stretch):
    n_L = int(round(L/dx)) #plate divided by shutter length L
    n_g = int(round(g/dx)) #converts to integer for the grid and array positions 
    n_d = int(round(d/dx))
    n_c0 = int(round (core_pad/dx))
    n_zc= int(round(z_core/dx))
    x_1= np.arange(-n_c0,2*n_L+n_g+n_c0+1)*dx 
    x_2= stretched(dx, stretch, X_ps)
    
    x= np.concatenate([x_1[0]-x_2[::-1], x_1, x_1[-1] + x_2])
    z= np.concatenate ([np.arange(n_zc+1)*dx, 
                        n_zc*dx+stretched(dx, stretch, 
                                          H-n_zc*dx)])
    return dict(x=x, z=z, dx=dx, 
                n_L=n_L, n_d=n_d, n_zc=n_zc,
                n_m=len(x_2)+n_c0,
                ic=slice(len(x_2), len(x_2)+len(x_1)))
    
def laplace_eq_solve(G,x_s): #solve laplace 
        x, z = G["x"], G["z"]
        nx, nz = len(x), len(z)
        N= nx*nz 
        idx = np.arange(N).reshape(nz,nx)
        
        #boundary stuffs
        fixed = np.zeros((nz, nx), bool)
        val = np.zeros((nz, nx))
        fixed[0, :] = True                
        fixed[-1, :] = True    
        val[-1, :] = E_o * z[-1]
        if x_s is not None:    # shutter @ z=d, V = 0
            i0 = G["n_m"] + int(round(x_s / G["dx"]))
            fixed[G["n_d"], i0:i0 + G["n_L"] + 1] = True
        
        # nx × nz nodes
        K, I = np.nonzero(~fixed)
        P = idx[K, I]
        Iw = np.where(I == 0, 1, I - 1)            
        Ie = np.where(I == nx - 1, nx - 2, I + 1)  
        
        ## x[I] x-position node
        hw = np.abs(x[I] - x[Iw]); he = np.abs(x[Ie] - x[I]) 
        hs = z[K] - z[K - 1];      hn = z[K + 1] - z[K]
        
        #finite difference 
        cW = 2 / (hw * (hw + he)); cE = 2 / (he * (hw + he))
        cS = 2 / (hs * (hs + hn)); cN = 2 / (hn * (hs + hn))
        cP = -(cW + cE + cS + cN)
        
        #store the equations in matrix form
        rows = np.concatenate([P, P, P, P, P])
        cols = np.concatenate([P, idx[K, Iw], idx[K, Ie], idx[K - 1, I], idx[K + 1, I]])
        data = np.concatenate([cP, cW, cE, cS, cN])

        Kf, If = np.nonzero(fixed)                  
        Pf = idx[Kf, If]
        
        #equations for fixed nodes
        rows = np.concatenate([rows, Pf])
        cols = np.concatenate([cols, Pf])
        data = np.concatenate([data, np.ones(len(Pf))])
        
        #2-D boundary-value array
        Amat = sp.csc_matrix((data, (rows, cols)), shape=(N, N))
        return spla.spsolve(Amat, val.ravel()).reshape(nz, nx)
        
def q_s(G,V): #surface charge 
    
    dV_dz= (-3*V[0]+4*V[1]-V[2])/(2*G["dx"])
    return -1*eps_o * dV_dz 
    
def q_c(G, q_es, range): #plate charge

    x = G["x"]
    m = (x >= range[0] - 1e-12) & (x <= range[1] + 1e-12)
    return W * np.trapezoid(q_es[m], x[m])
    
def q_c_shutter(G, x_s): #chqrge when shuttering 
     
     V = laplace_eq_solve(G, x_s)
     q = q_s(G, V)
     return q_c(G, q, plate1), q_c(G, q, plate2), V, q   
 
#-----------------------------------------------------------------------------------

G = grid_space(dx, X_far, H, stretch)
x, z = G["x"], G["z"]
print(f"finite difference grid: {len(x)} x {len(z)} nodes (dx = {dx*1e3:.4f} mm near the plates, "
      f"box is {(x[-1]-x[0])*1e3:.0f} mm x {z[-1]*1e3:.0f} mm)")

# Checking purposes
Q_free = q_c_shutter(G, None)[0]
print(f"Checking (a), no shutter: Q1 = {Q_free:.6e} Coulombs, expected {-eps_o*E_o*area:.6e} Coulombs") 

stroke = L + g
x_pos = np.unique(np.round(np.concatenate([
    np.arange(0, 6.0001e-3, 0.2e-3),
    np.arange(6e-3, stroke - 6e-3, 4e-3),
    np.arange(stroke - 6e-3, stroke + 1e-9, 0.2e-3), [stroke]])/dx)*dx)

Q1, Q2 = [], []  #plate1 & plate 2 
for k, xs in enumerate(x_pos): #position of shutter for the current pass of loop 
    q1, q2, V, q = q_c_shutter(G, xs)
    Q1.append(q1); Q2.append(q2)
    if k == len(x_pos) // 2: 
        V_mid, q_mid, x_sh_mid = V, q, xs
Q1, Q2 = np.array(Q1), np.array(Q2)
t1 = x_pos/v  # time for each position

#--------------------------------------------------------------------------------------
#output currents and voltages 

i1 = np.gradient(Q1, t1)          
i2 = np.gradient(Q2, t1)
V_out = tia_gain * (i2 - i1) #TIA         

# avg abs  V_out / one stroke = TIA gain * (charges that aremoved) / time_sweep
V_avg = tia_gain * (abs(Q1[-1] - Q1[0]) + abs(Q2[-1] - Q2[0])) / time_sweep
    
#----------------------------------------------------------------------------------------


#----------------------------------------------------------------------
#output in COMPILER


print(f"\n========== Sanity Checks Results ==========")

print(f"Plates: L = {L* 1e3:.3f} mm, W = {W *1e3:.3f} mm, d = {d* 1e3:.3f} mm, "
      f"g = {g*1e3:.1f} mm, v = {v:.4f} m/s, tia_gain = {tia_gain:.9f}Ohm")

print(f"Stroke time T = (L+g)/v = {time_sweep:.6f} seconds")



print(f"Q1, plate 1 fully covered  (t=0)  = {Q1[0]*1e12:.6f} picoCoulombs")
print(f"Q1, plate 1 fully exposed  (t=T)  = {Q1[-1]*1e12:.6f} picoCoulombs")
    
# accuracy check
print("\nCheck (b), Q1 at t = T with different grids:")
cases = [("base grid",                 dx,     X_far,     H,     stretch),
         ("coarser grid, dx = 0.25 mm", 2.5 * dx, X_far,    H,     stretch),
         ("larger box, 0.8 m",         dx,     2 * X_far, 2 * H, stretch),
         ("stretch = 1.08",            dx,     X_far,     H,     1.08)]


Q_ref = None #set to 0 
for name, dx_, Xf_, H_, r_ in cases:
    Gc = grid_space(dx_, Xf_, H_, r_)
    q1 = q_c_shutter(Gc, stroke)[0]
    Q_ref = q1 if Q_ref is None else Q_ref
    print(f"{name:28s} Q1 = {q1*1e12:.6f} picoCoulomb   change = " f"{100*(q1-Q_ref)/Q_ref:+.6f} %")


#---------------------------------------------------------------------------------

# FIGURESsssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
#the following is to just generate the plots and txt outputs in plot window 

plt.rcParams.update({"font.size": 16})

# Fig 1: potential & field lines near the shutter's left edge (mid-stroke)
fig, ax = plt.subplots(figsize=(9, 4))
ic, n_zc = G["ic"], G["n_zc"]
xc_, zc_ = x[ic], z[:n_zc + 1]                 
Vc = V_mid[:n_zc + 1, ic]
X, Zg = np.meshgrid(xc_ * 1e3, zc_ * 1e3)
cs = ax.contourf(X, Zg, Vc, levels=np.linspace(0, E_o * 6e-3, 31),
                 cmap="viridis", extend="both")

Ez, Ex = np.gradient(-Vc, dx)                  
under = (Zg < d * 1e3) & (X > x_sh_mid * 1e3) & (X < (x_sh_mid + L) * 1e3)
Ex = np.ma.masked_where(under, Ex); Ez = np.ma.masked_where(under, Ez)   
seeds = np.column_stack([np.arange(x_sh_mid * 1e3 - 7.5, x_sh_mid * 1e3 + 8, 0.75),
                         np.full(21, 5.9)])
ax.streamplot(xc_ * 1e3, zc_ * 1e3, Ex, Ez, color="w", linewidth=0.6,
              arrowsize=0.6, start_points=seeds, density=3)

ax.plot([x_sh_mid * 1e3, (x_sh_mid + L) * 1e3], [d * 1e3] * 2, "r", lw=3, label="shutter")
ax.plot([plate1[0] * 1e3, plate1[1] * 1e3], [0, 0], "orange", lw=5, label="plate 1")
ax.set_xlim(x_sh_mid * 1e3 - 8, x_sh_mid * 1e3 + 8); ax.set_ylim(0, 6)
ax.set_xlabel("x (millimetres)"); ax.set_ylabel("z (millimetres)")
ax.set_title("Charge Potential and Electric field lines near the shutter's left edge, mid-stroke")
fig.colorbar(cs, label="V (volts)"); ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig("I_am_the_storm_that_is_approaching.png", dpi=150)
    
# Fig 2: q_s on z = 0 (middle of the stroke)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
for ax, xl in zip(axs, [(-10, 2 * L * 1e3 + g * 1e3 + 10),
                        (x_sh_mid * 1e3 - 6, x_sh_mid * 1e3 + 6)]):
    ax.plot(x * 1e3, q_mid * 1e9, "C0", lw=2, label="finite Difference")
    ax.axvspan(plate1[0] * 1e3, plate1[1] * 1e3, color="orange", alpha=0.15, label="plate 1")
    ax.axvspan(plate2[0] * 1e3, plate2[1] * 1e3, color="m", alpha=0.15, label="plate 2")
    ax.axvline(x_sh_mid * 1e3, color="r", lw=0.8, ls="--")
    ax.axvline((x_sh_mid + L) * 1e3, color="r", lw=0.8, ls="--")
    ax.set_xlim(*xl); ax.set_xlabel("x (mm)")
    
axs[0].set_ylabel(r"$q_{s}$ (nC/m$^2$)")
axs[0].set_title("Full plates (dashed: shutter edge)")
axs[1].set_title("Zoomed In: shutter left edge")
axs[0].legend(fontsize=7, loc="center left")
fig.suptitle(r"Induced surface charge $q_{s}$ at 1/2 stroke")
fig.tight_layout(); fig.savefig("Yippy.png", dpi=150)

# Fig 3: q_c vs time plot
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(t1 * 1e3, Q1 * 1e12, "C0o-", ms=3, lw=1, label="Q1")
ax.plot(t1 * 1e3, Q2 * 1e12, "C1o-", ms=3, lw=1, label="Q2")
ax.set_xlabel("t (ms)"); ax.set_ylabel("Q (pC)")
ax.set_title("Charge Induced on sense plates in one stroke")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("Chilling.png", dpi=150)

# TIA output over one cycle
tP = np.concatenate([t1, time_sweep + t1])
VP = np.concatenate([V_out, -V_out[::-1]])
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(tP * 1e3, VP * 1e3, "C0o-", ms=2.5, lw=1)
ax.set_xlabel("t (ms)"); ax.set_ylabel(r"$V_{out}$ (mV)")
ax.set_title(f"TIA output")
fig.tight_layout(); fig.savefig("Stuffs.png", dpi=150)

print("images are saved after compiled output")   
    