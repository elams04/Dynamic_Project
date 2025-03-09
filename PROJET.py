# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 13:52:53 2025

@author: Utilisateur
"""

import sympy as sp

import numpy as np

j2,j3,j4=sp.symbols('j2,j3,j4')
xa,xb,yb,xd,yd,xf,yf,xh,yh,phic,phie,phig,w,h,l2,l3,l4,m1,m2,m3,m4,t=sp.symbols('x_a x_b y_b x_d y_d x_f y_f x_h y_h phi_c phi_e phi_g w h l2 l3 l4 m1 m2 m3 m4 t')
xad,xbd,ybd,xdd,ydd,xfd,yfd,xhd,yhd,phicd,phied,phiegd=sp.symbols('')
x_t=0.46+0.1*sp.sqrt(2)*sp.sin(sp.pi*(t-1))/(1+sp.cos(sp.pi*(t-1))**2)
y_t=0.47+0.1*sp.sqrt(2)*sp.sin(sp.pi*(t-1))*sp.cos(sp.pi*(t-1))/(1+sp.cos(sp.pi*(t-1))**2)
q=sp.Matrix(12,1,[xa,xb,yb,xd,yd,xf,yf,xh,yh,phic,phie,phig])
g=sp.Matrix([xa+w/2 -xb,
             xb+l2*sp.cos(phie)-xd,
            yb-h,yb+l2*sp.sin(phic)-yd,
            xd+l3*sp.cos(phie)-xf,
            yd+l3*sp.sin(phie)-yf,
            xf+l4*sp.cos(phig)-xh,
            yf-l4*sp.sin(phig)-yh,
            x_t-xh,
            y_t-yh])
G=g.jacobian(q)
M=sp.diag(m1,m2,0,m3,m3,m4,m4,0,0,j2+m2*l2**2/4,j3+m3*l3**2/4,j4+m4*l4**2/4)
M[1,9]=-m2*l2*sp.sin(phic)
M[9,1]=-m2*l2*sp.sin(phic)
M[10,3]=-m3*l3*sp.sin(phie)
M[3,10]=-m3*l3*sp.sin(phie)
M[4,10]=m3*l3*sp.cos(phie)
M[10,4]=m3*l3*sp.cos(phie)
M[5,11]=-m4*l4*sp.sin(phig)
M[11,5]=-m4*l4*sp.sin(phig)
M[6,11]=m4*l4*sp.cos(phig)
M[11,6]=m4*l4*sp.cos(phig)
k=0.5*q.T@M@q

sp.pprint(k)

with open("resulta.txt","w", encoding="utf-8") as f:
    f.write("Matrice M en LaTeX :\n")
    f.write(sp.latex(M))
    f.write(":\n Kinetic energy :\n")
    f.write(sp.latex(k))




