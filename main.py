# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 12:13:51 2022

@author: vbhen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize

e_ref = 1.602176634*1e-19

def lin(x, a): # used for linear fits through (0,0)
    return a*x

# ------ C_tot measurement ------
C_1 = 5365*1e-12
dC_1 = 0.5*1e-12
w_1 = 289.8*1e3
dw1 = 1*1e3
w_2 = 127.4*1e3
dw2 = 0.3*1e3
C_tot = C_1*((w_2)**2)/(((w_1)**2) - ((w_2)**2))
dC_tot = np.sqrt( dC_1**2 * ((w_2**2)/(w_1**2-w_2**2)**2)**2 + dw1**2 * ((2*C_1*w_1*w_2**2)/(w_1**2-w_2**2)**2)**2 + dw2**2 * ((2*C_1*w_1**2*w_2)/(w_1**2-w_2**2)**2)**2 )

# ------ rho measurement ------
rho = 32.79*1e3
drho = 0.01*1e3

# ------ damp measurement (attenuator) ------
U_pre = np.array([7.05, 10.7, 17.8])*1e-3
U_post = np.array([1.44, 2.16, 3.61])*1e-3
damp = np.average(U_pre / U_post)

# ------ A measurement ------
#Meauring A with the Atennuator for output voltages up 0.5V
UL_0 = np.array([0.25, 0.32, 0.388, 0.458, 0.525, 0.595, 0.66, 0.73, 0.8, 0.87, 1.25, 1.75])*1e-3/damp
UL_0_pp = np.array([1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 7, 10])*1e-3/(np.sqrt(2)*2*damp)
UL_out = np.array([91.27, 113.04, 134.71, 157.88, 181.26, 204.71, 228.07, 251.73, 275.53, 299.3, 418.01, 597.41])*1e-3
AL = UL_out / UL_0

#Measuring A in the middle range
UM_0 = np.array([0.45, 0.555, 0.658, 0.76, 0.861])*1e-3
UM_0_pp = np.array([1.3, 1.6, 1.9, 2.2, 2.5])*1e-3/(np.sqrt(2)*2)
UM_out = np.array([736.61, 907.10, 1074.36, 1223.8, 1394.7])*1e-3
AM = UM_out/UM_0

# A Measurement for Output voltages 1.5V to 3.5V
UH_0 = np.array([0.93, 1.06, 1.17, 1.26, 1.37, 1.46, 1.57, 1.68, 1.78, 1.88, 1.99, 2.09,  2.19, 2.3])*1e-3
UH_0_pp = np.array([2.7, 3, 3.3 , 3.6, 3.9, 4.2, 4.5, 4.8, 5.1, 5.4, 5.7, 6.0, 6.3, 6.6])*1e-3/(np.sqrt(2)*2)
UH_out = np.array([1504.1, 1674.6, 1841.6, 2008, 2176.9, 2342.1, 2506.3, 2672.1, 2832.6, 2989.4, 3142.5, 3282.5, 3409.7, 3527.6])*1e-3
AH = UH_out/UH_0

# All A measurements combined and fitted 
U_0 = np.append(UL_0, UM_0)
U_0 = np.append(U_0, UH_0)

U_out = np.append(UL_out, UM_out)
U_out = np.append(U_out, UH_out)

popt, pcov = curve_fit(lin, U_0, U_out)
A_R = popt[0]
dA_R_fit = np.sqrt(np.diag(pcov))[0]

dU_0 = 0.01*1e-3
dU_out = 0.01*1e-3

dA_R = np.sqrt( dU_out**2 * (1/U_0)**2 + dU_0**2 * (U_out/U_0**2)**2 )
dA_R_avg = np.average(dA_R)

plt.figure(figsize = (12, 10))
plt.xlabel('U in [mV]', fontsize=25)
plt.ylabel('U out [V]', fontsize=25)
plt.grid(True)
plt.scatter(U_0*1e3, U_out, label = "Amplification measurement")
plt.errorbar(U_0*1e3, U_out, xerr=dU_0*1e3, yerr=dU_out, markersize=4, capsize=5, fmt='o',barsabove=True)
plt.plot(U_0*1e3, A_R * U_0, label = "fit", color = 'r')
plt.legend(loc = 'best', fontsize=25)
plt.savefig('A')
plt.show()


# ------ R measurement ------
I_R = np.array([0, 5, 10, 16, 20, 25, 30])*1e-3

U11 = 0.21*1e-3
U11_pp = 0.6*1e-3/(np.sqrt(2)*2)
U21_ampl = np.array([161, 195, 218, 239, 252, 269, 279])*1e-3
U21 = U21_ampl / A_R
R1 = rho * U21 / (U11 - U21)

print("AAAAAAAAAAAAAAAAAAAA")
print(A_R)

U12 = 0.7*1e-3
U12_pp = 2.0*1e-3/(np.sqrt(2)*2)
U22_ampl = np.array([531.5, 515, 506, 498, 489, 481, 470])*1e-3
U22 = U22_ampl / A_R
R2 = rho * U22 / (U12 - U22)

U13 = 0.418*1e-3
U13_pp = 1.2*1e-3/(np.sqrt(2)*2)
U23_ampl = np.array([317.4, 324, 330, 337, 341, 345, 348])*1e-3
U23 = U23_ampl / A_R
R3 = rho * U23 / (U13 - U23)

U14_pp = 8*1e-3
U14 = 2.78*1e-3
U24_ampl = np.array([2406, 2273, 2142, 2034, 1945, 1851, 1754] )*1e-3
U24 = U24_ampl / A_R
R4 = rho * U24 / (U14 - U24)

U15_pp = 10*1e-3
U15 = 3.55*1e-3
U25_ampl = np.array([2956, 2801, 2642, 2512, 2413, 2292, 2173] )*1e-3
U25 = U25_ampl / A_R
R5 = rho * U25 / (U15 - U25)


U16_pp = 12*1e-3
U16 = 4.22*1e-3
U26_ampl = np.array([3487, 3336, 3170, 3019, 2902, 2758, 2616] )*1e-3
U26 = U26_ampl / A_R
R6 = rho * U26 / (U16 - U26)


dU1_R = 0.01*1e-3
dU2_ampl = 1*1e-3
dU24 = np.sqrt( (dU2_ampl/A_R)**2 + dA_R_avg**2 * (U24_ampl/A_R**2)**2 )
dU25 = np.sqrt( (dU2_ampl/A_R)**2 + dA_R_avg**2 * (U25_ampl/A_R**2)**2 )
dU26 = np.sqrt( (dU2_ampl/A_R)**2 + dA_R_avg**2 * (U26_ampl/A_R**2)**2 )
dI_R = 0.5 * 1e-3
dR4 = np.sqrt( dU24**2 * ((rho*U14) / (U14-U24)**2)**2 + drho**2 * ((U24)/(U14-U24))**2 + dU1_R**2 * ((rho*U24)/((U14-U24)**2))**2)
dR5 = np.sqrt( dU25**2 * ((rho*U15) / (U15-U25)**2)**2 + drho**2 * ((U25)/(U15-U25))**2 + dU1_R**2 * ((rho*U25)/((U15-U25)**2))**2)
dR6 = np.sqrt( dU26**2 * ((rho*U16) / (U16-U26)**2)**2 + drho**2 * ((U26)/(U16-U26))**2 + dU1_R**2 * ((rho*U26)/((U16-U26)**2))**2)

dR4_avg = np.average(dR4)

popt, pcov = np.polyfit(I_R, U24, 1, cov=True)
a_4 = popt[0]
b_4 = popt[1]
da_4 = np.sqrt(np.diag(pcov)[0])
db_4 = np.sqrt(np.diag(pcov)[1])
dU24_fit = np.sqrt(np.diag(pcov))[0]

popt, pcov = np.polyfit(I_R, U25, 1, cov=True)
a_5 = popt[0]
b_5 = popt[1]
da_5 = np.sqrt(np.diag(pcov)[0])
db_5 = np.sqrt(np.diag(pcov)[1])
dU25_fit = np.sqrt(np.diag(pcov))[0]

popt, pcov = np.polyfit(I_R, U26, 1, cov=True)
a_6 = popt[0]
b_6 = popt[1]
da_6 = np.sqrt(np.diag(pcov)[0])
db_6 = np.sqrt(np.diag(pcov)[1])
dU26_fit = np.sqrt(np.diag(pcov))[0]

def R(I, a=a_5, b=b_5, U1=U15):
    return (rho*(a*I + b)) / (U1 - a*I - b)


plt.figure(figsize = (12, 10))
plt.xlabel(r'|$\langle$I$\rangle$| [mA]', fontsize=25)
plt.ylabel('U2 [mV]', fontsize=25)
plt.grid(True)
plt.scatter(I_R*1e3, U24*1e3, label = "U1 = 2.78 mV", color='r')
plt.errorbar(I_R*1e3, U24*1e3, xerr=dU24*1e3, yerr=dI_R, markersize=4, capsize=5, fmt='o', color='r')
plt.scatter(I_R*1e3, U25*1e3, label = "U1 = 3.55 mV", color='g')
plt.errorbar(I_R*1e3, U25*1e3, xerr=dU25*1e3, yerr=dI_R, markersize=4, capsize=5, fmt='o', color='g')
plt.scatter(I_R*1e3, U26*1e3, label = "U1 = 4.22 mV", color='b')
plt.errorbar(I_R*1e3, U26*1e3, xerr=dU26*1e3, yerr=dI_R, markersize=4, capsize=5, fmt='o', color='b')
plt.plot(I_R*1e3, (a_4*I_R + b_4)*1e3, color='r')
plt.plot(I_R*1e3, (a_5*I_R + b_5)*1e3, color='g')
plt.plot(I_R*1e3, (a_6*I_R + b_6)*1e3, color='b')
plt.legend(loc = 'best', fontsize=25)
plt.savefig('R')
plt.show()


plt.figure(figsize = (12, 10))
plt.xlabel(r'|$\langle$I$\rangle$| [mA]', fontsize=25)
plt.ylabel('R [$\Omega$]', fontsize=25)
plt.grid(True)
plt.scatter(I_R*1e3, R4, label = "U1 = 2.78 mV", color='r')
plt.scatter(I_R*1e3, R5, label = "U1 = 3.55 mV", color='g')
plt.scatter(I_R*1e3, R6, label = "U1 = 4.22 mV", color='b')
plt.plot(I_R*1e3, R(I_R, a_4, b_4, U14), color='r')
plt.errorbar(I_R*1e3, R4, xerr=dI_R, yerr=dR4, markersize=4, capsize=5, fmt='o', color='r')
plt.plot(I_R*1e3, R(I_R, a_5, b_5, U15), color='g')
plt.errorbar(I_R*1e3, R5, xerr=dI_R, yerr=dR5, markersize=4, capsize=5, fmt='o', color='g')
plt.plot(I_R*1e3, R(I_R, a_6, b_6, U16), color='b')
plt.errorbar(I_R*1e3, R6, xerr=dI_R, yerr=dR6, markersize=4, capsize=5, fmt='o', color='b')
plt.legend(loc = 'best', fontsize=25)
plt.savefig('R1')
plt.show()


#----------2nd G_I*A^2 MEASUREMENT for I_z3 and I_e3
Upp_GIA = np.array([5.4, 4.9, 4.4, 3.9, 3.4, 2.9, 2.4, 1.9, 1.4, 0.9])*1e-3/(2*np.sqrt(2)) #eher nicht brauchen
U_GIA = np.array([0.945, 0.86, 0.776, 0.688, 0.6, 0.513, 0.428, 0.342, 0.26, 0.178])*1e-3/damp
Iz_GIA = np.array([50.2, 42.2, 34.6, 27.8, 21.9, 16.8, 12.4, 9, 6.8, 4.9])*1e-6

dIz = 0.3*1e-6
dU_GIA = 0.001*1e-3

popt, pcov = curve_fit(lin, U_GIA**2, Iz_GIA)
GIA_lin = popt[0]
dGIA_lin = np.sqrt(np.diag(pcov))[0]

popt, pcov = np.polyfit(U_GIA**2, Iz_GIA, 1, cov=True)
GIA_a = popt[0]
GIA_b = popt[1]
dGIA_a = np.sqrt(np.diag(pcov))[0]

GIA = GIA_a
dGIA_gauss = np.sqrt( dIz**2 * (1/U_GIA**2)**2 + dU_GIA**2 * (Iz_GIA/U_GIA**3)**2 )
dGIA = dGIA_a

plt.figure(figsize = (12, 10))
plt.xlabel(r'$U_{in}^2$ $[mV]^2$', fontsize=25)
plt.ylabel('$I_Z$ [$\mu A$]', fontsize=25)
plt.grid(True)
plt.scatter(U_GIA**2*1e6, Iz_GIA*1e6, label = r'measured $I_{Z}$', color='b')
plt.plot(U_GIA**2*1e6, GIA_lin * U_GIA**2 *1e6, color='g', label='linear fit')
plt.plot(U_GIA**2*1e6, (GIA_a * U_GIA**2 + GIA_b) *1e6, color='r', label='affine linear fit')
plt.errorbar(U_GIA**2*1e6, Iz_GIA*1e6, xerr=dU_GIA**2, yerr=dIz, markersize=4, capsize=5, fmt='o', color='b')
plt.legend(loc = 'best', fontsize=25)
plt.savefig('GIA')
plt.show()


#----------- e MEasurement----
Ie_3 = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])*1e-3
Iz_3 = np.array([5.4, 10.5, 16.7, 19.9, 24.2, 27.4, 30.8, 32.6, 36.1, 38.8, 39.9, 41.8, 43.3, 45.0, 46.3])*1e-6


plt.figure(figsize = (12, 10))
plt.xlabel(r'|$\langle$I$\rangle$| [mA]', fontsize=25)
plt.ylabel('$I_Z$ [$\mu A$]', fontsize=25)
plt.grid(True)
plt.scatter(Ie_3*1e3, Iz_3*1e6, label = r'measured $I_{Z}$', color='b')
plt.legend(loc = 'best', fontsize=25)
plt.savefig('e')
plt.show()


e = (2 * Iz_3 * C_tot) / (GIA * R(Ie_3, a_5, b_5, U15) * Ie_3)
e_avg = np.average(e)
e_avg_arr = np.zeros(len(Ie_3)) + e_avg

dIe = 0.1*1e-3
de = np.sqrt( dC_tot**2 * ((2*Iz_3)/(GIA*R(Ie_3)*Ie_3))**2 + dIz**2 * ((2*C_tot)/(GIA*R(Ie_3)*Ie_3))**2 + dGIA**2 * ((2*C_tot*Iz_3)/(GIA**2*R(Ie_3)*Ie_3))**2 + dR4_avg**2 * ((2*C_tot*Iz_3)/(GIA*R(Ie_3)**2*Ie_3))**2  + dIe**2 * ((2*C_tot*Iz_3)/(GIA))**2 * (((U14-a_4*Ie_3-b_4)*(2*a_4*Ie_3+b_4) + a_4*Ie_3*(a_4*Ie_3-b_4))/(rho*(a_4*Ie_3**2+b_4*Ie_3)**2))**2)

de_avg = np.average(de) / np.sqrt(len(de))

popt, pcov = curve_fit(lin, Ie_3*R(Ie_3, a_5, b_5, U15), Iz_3)
k = popt[0]
dk = np.sqrt(np.diag(pcov))[0]

e_fit = k * 2 * C_tot / GIA
de_fit = dk * 2 * C_tot / GIA
e_fit_arr = np.zeros(len(Ie_3)) + e_fit
e_ref_arr = np.zeros(len(Ie_3)) + e_ref

plt.figure(figsize = (12, 10))
plt.xlabel(r'|$\langle$I$\rangle$| [mA]', fontsize=25)
plt.ylabel('e [C]', fontsize=25)
plt.grid(True)
plt.scatter(Ie_3*1e3, e, label = 'seperatly calculated |e|', color='b')
plt.errorbar(Ie_3*1e3, e, xerr=dIe, yerr=de, markersize=4, capsize=5, fmt='o', color='b')
plt.plot(Ie_3*1e3, e_fit_arr, color='r', label='fitted |e|')
plt.fill_between(Ie_3*1e3, e_fit_arr - de_fit, e_fit_arr + de_fit, color='r', alpha=0.2)
plt.plot(Ie_3*1e3, e_avg_arr, color='g', label='average |e|')
plt.fill_between(Ie_3*1e3, e_avg_arr - de_avg, e_avg_arr + de_avg, color='g', alpha=0.2)
plt.plot(Ie_3*1e3, e_ref_arr, color='k', label='reference value for |e|')
plt.legend(loc = 'best', fontsize=25)
plt.savefig('e1')
plt.show()



plt.figure(figsize = (12, 10))
plt.xlabel(r'|$\langle$I$\rangle$| * R(|$\langle$I$\rangle$|)', fontsize=25)
plt.ylabel(r'$I_Z$ [$\mu A$]', fontsize=25)
plt.grid(True)
plt.scatter(Ie_3*R(Ie_3, a_5, b_5, U15), Iz_3*1e6, label = 'measured $I_Z$', color='b')
plt.plot(Ie_3*R(Ie_3, a_5, b_5, U15), k * Ie_3*R(Ie_3, a_5, b_5, U15)*1e6, color='b')
plt.legend(loc = 'best', fontsize=25)
plt.savefig('e2')
plt.show()

print("C measurement")
print("C_1 = " + str(C_1) + " +/- " + str(dC_1))
print("w_1 = " + str(w_1) + " +/- " + str(dw1))
print("w_2 = " + str(w_2) + " +/- " + str(dw2))
print("C_tot = " + str(C_tot) + " +/- " + str(dC_tot) + '\n')

print("R measurement")
print("rho = " + str(rho) + " +/- " + str(drho))
print("U1_R = (" + str(U14) + ", " + str(U15) + ", " + str(U16) + ") +/- " + str(dU1_R))
print("A_R = " + str(A_R) + " +/- " + str(dA_R_avg))
print("a_4 = " + str(a_4) + " +/- " + str(da_4))
print("b_4 = " + str(b_4) + " +/- " + str(db_4))
print("a_5 = " + str(a_5) + " +/- " + str(da_5))
print("b_5 = " + str(b_5) + " +/- " + str(db_5))
print("a_6 = " + str(a_6) + " +/- " + str(da_6))
print("b_6 = " + str(b_6) + " +/- " + str(db_6) + '\n')

print("GI*A^2 measurement")
print("GIA_lin = " + str(GIA_lin) + " +/- " + str(dGIA_lin))
print("GIA = " + str(GIA) + " +/- " + str(dGIA) + '\n')

print("|e| measurement")
print("k = " + str(k) + " +/- " + str(dk))
print("e_fit = " + str(e_fit) + " +/- " + str(de_fit))
print("e_avg = " + str(e_avg) + " +/- " + str(de_avg))
print("bias e_fit = " + str(e_ref-e_fit))
print("bias e_avg = " + str(e_ref-e_avg))
print("deviation e_fit = " + str((e_ref-e_fit)/de_fit) + "sigma")
print("deviation e_avg = " + str((e_ref-e_avg)/de_avg) + "sigma")





## further plots

A_dis = U_out / U_0

plt.figure(figsize = (12, 10))
plt.xlabel(r'U in [mV]', fontsize=25)
plt.ylabel("Amplification", fontsize=25)
plt.grid(True)
plt.scatter(U_0*1e3, A_dis, label = 'calculated Amplification', color='b')
plt.legend(loc = 'best', fontsize=25)
plt.savefig('A dis')
plt.show()


GIA_dis = Iz_GIA / U_GIA**2

plt.figure(figsize = (12, 10))
plt.xlabel(r'U in [mV]', fontsize=25)
plt.ylabel(r'$G_I * A^2$ [$\frac{A}{V^2}$]', fontsize=25)
plt.grid(True)
plt.scatter(U_GIA*1e3, GIA_dis, label = r'calculated $G_I * A^2$', color='b')
plt.legend(loc = 'best', fontsize=25)
plt.savefig('GIA dis')
plt.show()


R_11 = rho * U21 / (U11 - U21)
R_12 = rho * U22 / (U12 - U22)
R_13 = rho * U23 / (U13 - U23)
R_14 = rho * U24 / (U14 - U24)
R_15 = rho * U25 / (U15 - U25)
R_16 = rho * U26 / (U16 - U26)

R2_12 = rho * (U22 - U21) / (U21 - U22 + U12 - U11)
R2_13 = rho * (U23 - U21) / (U21 - U23 + U13 - U11)
R2_14 = rho * (U24 - U21) / (U21 - U24 + U14 - U11)
R2_15 = rho * (U25 - U21) / (U21 - U25 + U15 - U11)
R2_16 = rho * (U26 - U21) / (U21 - U26 + U16 - U11)

R2_23 = rho * (U22 - U23) / (U23 - U22 + U12 - U13)
R2_24 = rho * (U22 - U24) / (U24 - U22 + U12 - U14)
R2_25 = rho * (U22 - U25) / (U25 - U22 + U12 - U15)
R2_26 = rho * (U22 - U26) / (U26 - U22 + U12 - U16)

R2_34 = rho * (U23 - U24) / (U24 - U23 + U13 - U14)
R2_35 = rho * (U23 - U25) / (U25 - U23 + U13 - U15)
R2_36 = rho * (U23 - U26) / (U26 - U23 + U13 - U16)

R2_45 = rho * (U24 - U25) / (U25 - U24 + U14 - U15)
R2_46 = rho * (U24 - U26) / (U26 - U24 + U14 - U16)

R2_56 = rho * (U25 - U26) / (U26 - U25 + U15 - U16)

plt.figure(figsize = (12, 10))
plt.xlabel(r'$I_e$ [$\mu$ A]', fontsize=25)
plt.ylabel(r'R', fontsize=25)
plt.grid(True)
plt.scatter(I_R*1e6, R2_45, label = "45")
plt.scatter(I_R*1e6, R2_46, label = "46")
plt.scatter(I_R*1e6, R2_56, label = "56")
plt.legend(loc = 'best', fontsize=25)
plt.savefig('R dis')
plt.show()

Iz_4 = np.array([13.6, 24.2, 32.6, 38.8, 42.55, 46.3])*1e-6
Ie_4 = I_R[1:]

e_4 = (2 * Iz_4 * C_tot) / (GIA * R2_46[1:] * Ie_4)
plt.plot(Iz_4,e_4)


plt.figure(figsize = (12, 10))
plt.xlabel(r'|$\langle$I$\rangle$| [mA]', fontsize=25)
plt.ylabel('e [C]', fontsize=25)
plt.grid(True)
plt.scatter(Ie_4*1e3, e_4, label = '|e| with the new R value', color='b')
plt.plot(Ie_3*1e3, e_ref_arr, color='k', label='reference value for |e|')
plt.legend(loc = 'best', fontsize=25)
plt.savefig('R e dis')
plt.show()



plt.figure(figsize = (12, 10))
plt.xlabel(r'|$\langle$I$\rangle$| [mA]', fontsize=25)
plt.ylabel('R [$\Omega$]', fontsize=25)
plt.grid(True)
plt.plot(I_R*1e3, R(I_R, a_5, b_5, U15), color='g', label = "U1 = 3.55mV (report)")
plt.scatter(I_R*1e3, R1, label = "U1 = 0.21mV", color='b')
plt.scatter(I_R*1e3, R3, label = "U1 = 0.418mV", color='y')
plt.legend(loc = 'best', fontsize=25)
plt.savefig('R1 dis')
plt.show()

plt.figure(figsize = (12, 10))
plt.xlabel(r'|$\langle$I$\rangle$| [mA]', fontsize=25)
plt.ylabel('R [$\Omega$]', fontsize=25)
plt.grid(True)
plt.plot(I_R*1e3, R(I_R, a_5, b_5, U15), color='g', label = "U1 = 3.55mV (report)")
plt.scatter(I_R*1e3, R2_12, label = "U1 = 0.21mV & 0.7mV", color='r')
plt.scatter(I_R*1e3, R1, label = "U1 = 0.21mV", color='b')
plt.scatter(I_R*1e3, R2, label = "U1 = 0.7mV", color='k')
plt.legend(loc = 'best', fontsize=25)
plt.savefig('R1 dis1')
plt.show()

plt.figure(figsize = (12, 10))
plt.xlabel(r'|$\langle$I$\rangle$| [mA]', fontsize=25)
plt.ylabel('R [$\Omega$]', fontsize=25)
plt.grid(True)
plt.plot(I_R*1e3, R(I_R, a_5, b_5, U15), color='g', label = "U1 = 3.55mV (report)")
plt.scatter(I_R*1e3, R2_23, label = "U1 = 0.418mV & 0.7mV", color='r')
plt.scatter(I_R*1e3, R2_13, label = "U1 = 0.21mV & 0.418mV", color='b')
plt.scatter(I_R*1e3, R2_12, label = "U1 = 0.21mV & 0.7mV", color='y')
plt.legend(loc = 'best', fontsize=25)
plt.savefig('R1 dis2')
plt.show()





U1_o = U15
U2_o = U25[1:]
Iz_o = Iz_4
Ie_o = Ie_4



def e_opt(i, k_o):
    return (2*C_tot * Iz_o[i] * (U1_o - U2_o[i] + rho * Ie_o[i] / k_o)) / (GIA * Ie_o[i] * rho * U2_o[i])

def R_opt(k_o):
    avg = 0
    ret = 0
    for i in range(6):
        avg += e_opt(i,k_o)
    avg = avg/6
    for i in range(6):
        ret += np.abs(e_opt(i,k_o) - avg)
    return ret

def R_opt1(k_o):
    ret = 0
    for i in range(6):
        ret += np.abs(e_opt(i,k_o) - e_ref)
    return ret
    
print("K:")
print(minimize(R_opt1, 1))
print("K1:")
print(minimize(R_opt1, 1).fun)
for i in range(6):
    print(e_opt(i,1.3*1e-9))


    
    
"""
Iz_4 = np.array([13.6, 24.2, 32.6, 38.8, 42.55, 46.3])*1e-6
Ie_4 = I_R[1:]




Ie_3 = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])*1e-3
Iz_3 = np.array([5.4, 10.5, 16.7, 19.9, 24.2, 27.4, 30.8, 32.6, 36.1, 38.8, 39.9, 41.8, 43.3, 45.0, 46.3])*1e-6



I_R = np.array([0, 5, 10, 16, 20, 25, 30])*1e-3


U15_pp = 10*1e-3
U15 = 3.55*1e-3
U25_ampl = np.array([2956, 2801, 2642, 2512, 2413, 2292, 2173] )*1e-3
U25 = U25_ampl / A_R
R5 = rho * U25 / (U15 - U25)
"""














