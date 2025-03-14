# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 13:52:53 2025

@author: Utilisateur
"""

import sympy as sp
import numpy as np

t=sp.Symbol('t') #time 

#Parameters of probleme
j2,j3,j4,w,h,l2,l3,l4,m1,m2,m3,m4,h,l,gravity=sp.symbols('j2 j3 j4 w h l2 l3 l4 m1 m2 m3 m4 h l gamma')

#Coordinates 
xa=sp.Function('x_a')(t)
xb=sp.Function('x_b')(t)
yb=sp.Symbol('y_b')  # yb is constant =h
xd=sp.Function('x_d')(t)
yd=sp.Function('y_d')(t)
xf=sp.Function('x_f')(t)
yf=sp.Function('y_f')(t)
xh=sp.Function('x_h')(t)
yh=sp.Function('y_h')(t)
phic=sp.Function('phi_c')(t)
phie=sp.Function('phi_e')(t)
phig=sp.Function('phi_g')(t)

#Coordinate's symbols
Xa,Xb,Xd,Yd,Xf,Yf,Xh,Yh,PHIc,PHIe,PHIg=sp.symbols('x_a x_b x_d y_d x_f y_f x_h y_h phi_c phi_e phi_g')
xad, xbd, xdd, ydd, xfd, yfd, xhd, yhd, phicd, phied, phigd = sp.symbols(
    r'\dot{x_a}, \dot{x_b}, \dot{x_d}, \dot{y_d}, \dot{x_f}, \dot{y_f}, \dot{x_h}, \dot{y_h}, \dot{\phi_c}, \dot{\phi_e}, \dot{\phi_g}'
)
#Dictionary of substitution between function and symbols
substitution={
    xa: Xa,
    xb: Xb,
    xd: Xd,
    yd: Yd,
    xf: Xf,
    yf: Yf,
    xh: Xh,
    yh: Yh,
    phic: PHIc,
    phie: PHIe,
    phig: PHIg,
    sp.diff(xa,t):xad,
    sp.diff(xb,t):xbd,
    sp.diff(xd,t):xdd,
    sp.diff(yd,t):ydd,
    sp.diff(xf,t):xfd,
    sp.diff(yf,t):yfd,
    sp.diff(xh,t):xhd,
    sp.diff(yh,t):yhd,
    sp.diff(phic,t):phicd,
    sp.diff(phie,t):phied,
    sp.diff(phig,t):phigd,
    m1: 10,
    m2: 10,
    m3: 10,
    m4:10,
    j2:10,
    j3:10,
    j4:2,
    w:2,
    }
#Imposed displacement 
x_t=0.46+0.1*sp.sqrt(2)*sp.sin(sp.pi*(t-1))/(1+sp.cos(sp.pi*(t-1))**2)
y_t=0.47+0.1*sp.sqrt(2)*sp.sin(sp.pi*(t-1))*sp.cos(sp.pi*(t-1))/(1+sp.cos(sp.pi*(t-1))**2)
#vectors
q=sp.Matrix(12,1,[xa,xb,yb,xd,yd,xf,yf,xh,yh,phic,phie,phig])
q_dot=q.diff(t)
g=sp.Matrix([xa+w/2 -xb,
             xb+l2*sp.cos(phic)-xd,
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
M[10,3]=-m3*l3*sp.sin(phie)
M[4,10]=m3*l3*sp.cos(phie)
M[5,11]=-m4*l4*sp.sin(phig)
M[6,11]=m4*l4*sp.cos(phig)
f_ext=sp.Matrix(12,1,[0,0,-(m1+m2*0.5)*gravity,0,-(m2+m3)*gravity*0.5,0,-(m3+m4)*gravity*0.5,0,-m4*gravity*0.5,0,0,0])
U=-f_ext.T@q.subs(substitution)

k=0.5*q_dot.T@M@q_dot #kenecti energy


q_dot_display=q_dot.subs(substitution)
k_display=k.subs(substitution)
M_display=M.subs(substitution)
g_display=g.subs(substitution)
G_display=G.subs(substitution)

#file who store latex expression of M g G k q

with open("result.txt","w", encoding="utf-8") as f:
    f.write("Matrice M en LaTeX :\n")
    f.write(sp.latex(M_display))
    f.write(":\n Kinetic energy :\n")
    f.write(sp.latex(k_display))
    f.write(":\n g matrix :\n")
    f.write(sp.latex(g_display))
    f.write(":\n G matrix :\n")
    f.write(sp.latex(G_display))
    f.write(":\n q_dot matrix :\n")
    f.write(sp.latex(q_dot_display))
    f.write(":\n U matrix :\n")
    f.write(sp.latex(U))




