import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import time

# =============================================================================
## Evaluation of the symbolic matrices expressions
# =============================================================================

#Time symbol definition
t=sp.Symbol('t') #time 

#Parameters of problem
j2,j3,j4,w,h,l2,l3,l4,m1,m2,m3,m4,hl,l,f_n,gravity,mu=sp.symbols('j2 j3 j4 w h l2 l3 l4 m1 m2 m3 m4 hl l f_n ga mu')

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

#Imposed displacement 
x_t=0.46+0.1*sp.sqrt(2)*sp.sin(sp.pi*(t-1))/(1+sp.cos(sp.pi*(t-1))**2)
y_t=0.47+0.1*sp.sqrt(2)*sp.sin(sp.pi*(t-1))*sp.cos(sp.pi*(t-1))/(1+sp.cos(sp.pi*(t-1))**2)

#Coordinates vector definition
sq=sp.Matrix(12,1,[xa,xb,yb,xd,yd,xf,yf,xh,yh,phic,phie,phig])
sq_d=sq.diff(t)

# Define named symbols for the derivatives
sq_dot_symbols = sp.Matrix(12, 1, sp.symbols('x_a_dot x_b_dot y_b_dot x_d_dot y_d_dot x_f_dot y_f_dot x_h_dot y_h_dot phi_c_dot phi_e_dot phi_g_dot'))

xa_dot=sp.Function('x_a_dot')(t)
xb_dot=sp.Function('x_b_dot')(t)
xd_dot=sp.Function('x_d_dot')(t)
yd_dot=sp.Function('y_d_dot')(t)
xf_dot=sp.Function('x_f_dot')(t)
yf_dot=sp.Function('y_f_dot')(t)
xh_dot=sp.Function('x_h_dot')(t)
yh_dot=sp.Function('y_h_dot')(t)
phic_dot=sp.Function('phi_c_dot')(t)
phie_dot=sp.Function('phi_e_dot')(t)
phig_dot=sp.Function('phi_g_dot')(t)

# Replace the time derivatives of q with the new symbols
sq_d = sq_d.subs({sq[i].diff(t): sq_dot_symbols[i] for i in range(sq.shape[0])})

# Constriants definition
sg=sp.Matrix([xa+w/2 -xb,
             xb+l2*sp.cos(phic)-xd,
            yb-hl,
            yb+l2*sp.sin(phic)-yd,
            xd+l3*sp.cos(phie)-xf,
            yd+l3*sp.sin(phie)-yf,
            xf+l4*sp.cos(phig)-xh,
            yf+l4*sp.sin(phig)-yh,
            x_t-xh,
            y_t-yh])
sG=sg.jacobian(sq)

# Mass matrix definition
sM=sp.diag(m1,m2/4,m2/4,(m2+m3)/4,(m2+m3)/4,(m3+m4)/4,(m3+m4)/4,m4/4,m4/4,j2,j3,j4)
sM[1,3] = (m2)/4
sM[2,4] = (m2)/4
sM[3,5] = (m3)/4
sM[4,6] = (m3)/4
sM[5,7] = m4/4
sM[6,8] = m4/4
sM[3,1] = (m2)/4
sM[4,2] = (m2)/4
sM[5,3] = (m3)/4
sM[6,4] = (m3)/4
sM[7,5] = m4/4
sM[8,6] = m4/4

# F_ext vector definition
sf_ext=sp.Matrix(12,1,[-sp.tanh(10*xa_dot)*mu*np.abs(f_n),
                      0,
                      -(m1+m2/2)*gravity,
                      0,
                      -(m2+m3)*gravity/2,
                      0,
                      -(m3+m4)*gravity/2,
                      0,
                      -m4*gravity/2,
                      0,
                      0,               
                      0])


# f_ext derivative with respect to q_d
sf_ext_dqdot = sf_ext.jacobian(sq_d)

# =============================================================================
## Initial conditions and parameters definition 
# =============================================================================

n = 12 # Number of generalized coordinates
m = 10 # Number of constraints

# Define problem parameters
par = { # Original parameters
    'm1': 5.6,
    'm2': 5.2,
    'm3': 4.4,
    'm4': 0.8,
    'l2': 0.629,
    'l3': 0.444,
    'l4': 0.22,
    'j2': 0.1714,
    'j3': 0.07228,
    'j4': 0.00322667,
    'hl': 0.283,
    'w': 0.129,
    'k1': 100,
    'k2': 10,
    'k3': 10,
    'mu': 0.01,
    'gravity': 9.81
}

par1 = { # Modified parameters
    'm1': 10.6,
    'm2': 5.2,
    'm3': 4.4,
    'm4': 0.8,
    'l2': 0.629,
    'l3': 0.444,
    'l4': 0.22,
    'j2': 0.1714,
    'j3': 0.07228,
    'j4': 0.00322667,
    'hl': 0.283,
    'w': 0.129,
    'k1': 50,
    'k2': 10,
    'k3': 10,
    'mu': 0.01,
    'gravity': 9.81
}

# Matrix M definition
# Substitute the symbols in sM with the values from the par dictionary
sM_numeric = sM.subs(par)

# Convert the resulting symbolic matrix to a NumPy array for numerical computations
M = np.array(sM_numeric).astype(np.float64)

# Stiffness matrix definition
K=np.zeros((12,12))
K[9,9] = par['k1']
K[10,10] = par['k2']
K[11,11] = par['k3']

# Define integrator parameters
h = 0.005
alpha = 0.3
gamma = 0.5 + alpha
beta = 0.25 * (gamma + 1/2)**2
tolQ = 1e-9
maxiter = 502

t_0 = 0
t_f = 2
tt = np.linspace(t_0, t_f, int((t_f-t_0)/h))

# Initializing variables
q = np.zeros((len(tt),n))
q_d = np.zeros((len(tt),n))
q_dd = np.zeros((len(tt),n))
l = np.zeros((len(tt),m))

# Define intial conditions
phic_0 = 89 * (np.pi / 180)
phie_0 = -30 * (np.pi / 180)
phig_0 = -90 * (np.pi / 180)
q_0 = np.array([
    0,
    par['w']/2,
    par['hl'],
    par['w']/2+par['l2']*math.cos(phic_0),
    par['hl'] + par['l2']*math.sin(phic_0),
    par['w']/2+par['l2']*math.cos(phic_0)+par['l3']*math.cos(phie_0),
    par['hl'] + par['l2']*math.sin(phic_0)+par['l3']*math.sin(phie_0),
    par['w']/2+par['l2']*math.cos(phic_0)+par['l3']*math.cos(phie_0)+par['l4']*math.cos(phig_0),
    par['hl'] + par['l2']*math.sin(phic_0)+par['l3']*math.sin(phie_0)+par['l4']*math.sin(phig_0),
    phic_0,
    phie_0,
    phig_0,
])
q[0,:] = q_0
q_d[0] = np.zeros((n))
q_dd[0] = np.zeros((n))
l[0] = np.zeros((m))

# =============================================================================
## Numerical matrices and residuals evaluation 
# =============================================================================

def G(q):
   
    # Evaluate the G matrix numerically
    G = np.array([
        [1,-1,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,-1,0,0,0,0,0,-par['l2']*np.sin(q[9]),0,0],
        [0,0,1,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,-1,0,0,0,0,par['l2']*np.cos(q[9]),0,0],
        [0,0,0,1,0,-1,0,0,0,0,-par['l3']*np.sin(q[10]),0],
        [0,0,0,0,1,0,-1,0,0,0,par['l3']*np.cos(q[10]),0],
        [0,0,0,0,0,1,0,-1,0,0,0,-par['l4']*np.sin(q[11])],
        [0,0,0,0,0,0,1,0,-1,0,0,par['l4']*np.cos(q[11])],
        [0,0,0,0,0,0,0,-1,0,0,0,0],
        [0,0,0,0,0,0,0,0,-1,0,0,0]
    ])

    return G

def Kt(q,l):
    # Evaluate the d(Gl)/dq matrix
    dGl = np.zeros((12,12))
    dGl[9,9] = -par['l2']*np.cos(q[9])*l[1]-par['l2']*np.sin(q[9])*l[3]
    dGl[10,10] = -par['l3']*np.cos(q[10])*l[4]-par['l3']*np.sin(q[10])*l[5]
    dGl[11,11] = -par['l4']*np.cos(q[11])*l[6]-par['l4']*np.sin(q[11])*l[7]

    # Evaluate the matrix numerically
    Kt =  dGl + K 
    return np.array(Kt).astype(np.float64) 

def Ct(q_d):
    # Define the numerical values for q
    q_values = {
        xa_dot: q_d[0],
    }

    # Define the parameter values
    param_values = {
        m1: par['m1'],
        m2: par['m2'],
        m3: par['m3'],
        m4: par['m4'],
        mu: par['mu'],
        gravity: par['gravity'],
    }

    # Combine all substitutions
    sub = {**q_values, **param_values}

    # Evaluate the G matrix numerically
    Ct = sf_ext_dqdot.subs(sub)

    return np.array(Ct).astype(np.float64)

def f_ext(q,q_d,l):
    
    # Evaluate the f_ext vector numerically
    f_ext = np.array([
        -np.tanh(10*q_d[0])*par['mu']*np.abs(l[2]),
        0,
        -(par['m1']+par['m2']/2)*par['gravity'],
        0,
        -(par['m2']+par['m3'])*par['gravity']/2,
        0,
        -(par['m3']+par['m4'])*par['gravity']/2,
        0,
        -par['m4']*par['gravity']/2,
        0,
        0,
        0])
    
    return f_ext 

# Evaluate residuals
def res(q,q_d, q_dd, l,tt):

    # Evaluate the equation of motion residuals
    r = M @ q_dd + G(q).T @ l + K @ (q-q_0) - f_ext(q,q_d,l)
    
    x_t=0.46+0.1*np.sqrt(2)*np.sin(np.pi*(tt-1))/(1+np.cos(np.pi*(tt-1))**2)
    y_t=0.47+0.1*np.sqrt(2)*np.sin(np.pi*(tt-1))*np.cos(np.pi*(tt-1))/(1+np.cos(np.pi*(tt-1))**2)

    # Evaluate the constraint residuals
    g = np.array([
        q[0] + par['w']/2 - q[1],
        q[1] + par['l2']*np.cos(q[9]) - q[3],
        q[2] - par['hl'],
        q[2] + par['l2']*np.sin(q[9]) - q[4],
        q[3] + par['l3']*np.cos(q[10]) - q[5],
        q[4] + par['l3']*np.sin(q[10]) - q[6],
        q[5] + par['l4']*np.cos(q[11]) - q[7],
        q[6] + par['l4']*np.sin(q[11]) - q[8],
        x_t - q[7] ,
        y_t - q[8]
    ])

    resc = np.concatenate([r,g])
    return resc 

# =============================================================================
## Motion Solver
# =============================================================================

# Initialize lists to store the trace of marker H
trace_x = []
trace_y = []

# Time loop
for i in range(len(tt)-1):

    # Next step prediction
    q[i+1] = q[i] + h*q_d[i] + h**2*(0.5-beta)*q_dd[i]
    q_d[i+1] = q_d[i] + h*(1-gamma)*q_dd[i]
    q_dd[i+1] = 0
    l[i+1] = l[i]

    # Differential corrector loop
    j = 0
    # Initial guess
    qt = np.zeros((maxiter,n))
    qt_d = np.zeros((maxiter,n))
    qt_dd = np.zeros((maxiter,n))
    lt = np.zeros((maxiter,m))    
    qt[0] = q[i+1]
    qt_d[0] = q_d[i+1]
    qt_dd[0] = q_dd[i+1]
    lt[0] = l[i+1]

    while np.linalg.norm(res(qt[j],qt_d[j],qt_dd[j],lt[j],tt[i])) > tolQ:

        # Defining matrix S_t 
        first_block = 1/(beta*h**2)*M - gamma/(beta*h)*Ct(qt_d[j]) + Kt(qt[j],lt[j])
        second_block =  G(qt[j]).T
        top_row = np.hstack((first_block, second_block))  
        bottom_row = np.hstack((second_block.T, np.zeros((m,m))))   

        # Final matrix: concatenate top and bottom rows
        S_t = np.vstack((top_row, bottom_row))  

        # Solve the linear system for Dql
        #Dql = np.linalg.solve(S_t, -res(qt[j],qt_d[j],qt_dd[j],lt[j])) 
        Dql= np.linalg.inv(S_t) @ -res(qt[j],qt_d[j],qt_dd[j],lt[j],tt[i])

        # Update variables
        qt[j+1] = qt[j] + Dql[:n].T
        qt_d[j+1] = qt_d[j] + gamma/(beta*h)*Dql[:n].T
        qt_dd[j+1] = qt_dd[j] + 1/(beta*h**2)*Dql[:n].T
        lt[j+1] = lt[j] + Dql[n:].T

        #print('Iteration:',j," Residual:",np.linalg.norm(res(qt[j],qt_d[j],qt_dd[j],lt[j],tt[i])))
        
        if j == maxiter-2:
            print('Differential corrector did not converged')
            exit()
        
        j += 1

    # Update the trace of marker H
    trace_x.append(q[i, 7])  # x-coordinate of marker H
    trace_y.append(q[i, 8])  # y-coordinate of marker H

    q[i+1] = qt[j]
    q_d[i+1] = qt_d[j]
    q_dd[i+1] = qt_dd[j]
    l[i+1] = lt[j]

    print('Completion: ',int(i/(len(tt)-1)*100),'%')

print('Total required space: ',np.max(q[:,7])-np.min(q[:,0]),'[m]')

# =============================================================================
## Plotting
# =============================================================================

xa,xb,yb,xd,yd,xh,yh,xf,yf=q[:,0],q[:,1],q[:,2],q[:,3],q[:,4],q[:,5],q[:,6],q[:,7],q[:,8]
Max_xa=np.max(xa)
Max_xf=np.max(xf)
Min_xa=np.min(xa)
Min_xf=np.min(xf)

fig=plt.figure()
line1,=plt.plot([],[],'r')
line2,=plt.plot([],[],'r')
line3,=plt.plot([],[],'r')
line4,=plt.plot([],[],'r')
line5,=plt.plot([],[],'b')
line6,=plt.plot([],[],'g')
line7,=plt.plot([],[],'r')
plt.xlabel('x axis in metere')
plt.ylabel('y axis in meter')
plt.title('movment of robot')
plt.xlim(-1,1)
plt.ylim(0,1)
plt.grid(True)
def anime(i):
    line1.set_data([xa[i],xa[i]],[0,par['hl']])
    line2.set_data([xa[i],xa[i]+par['w']],[0,0])
    line3.set_data([xa[i]+par['w'],xa[i]+par['w']],[0,par['hl']])
    line4.set_data([xa[i],xa[i]+par['w']],[par['hl'],par['hl']])
    line5.set_data([xb[i],xd[i]],[yb[i],yd[i]])
    line6.set_data([xd[i],xh[i]],[yd[i],yh[i]])
    line7.set_data([xh[i],xf[i]],[yh[i],yf[i]])
    return line1,line2,line3,line4,line5,line5,line6,line7

ani=animation.FuncAnimation(fig,anime,frames=range(0,int((t_f-t_0)/h)),interval=1, blit=False, repeat=False)

# Enregistrement de l'animation
ani.save("arms robot video.mp4", writer="ffmpeg", fps=60)
plt.show()


# Robot at t_end --------------------------------------------------------------
# Plot the markers and connect them with lines
plt.plot([q[-1, 1], q[-1, 3]], [q[-1, 2], q[-1, 4]], 'g-', label='First Beam')  
plt.plot([q[-1, 3], q[-1, 5]], [q[-1, 4], q[-1, 6]], 'b-', label='Second Beam')   
plt.plot([q[-1, 5], q[-1, 7]], [q[-1, 6], q[-1, 8]], 'k-', label='Third Beam')  

# Plot the markers
plt.plot(q[-1, 0], 0, 'ro', label='A')  
plt.plot(q[-1, 1], q[-1, 2], 'go', label='B')  
plt.plot(q[-1, 3], q[-1, 4], 'bo', label='D')   
plt.plot(q[-1, 5], q[-1, 6], 'ko', label='F')  
plt.plot(q[-1, 7], q[-1, 8], 'mo', label='H')   

# Plot the trace of marker H
plt.plot(trace_x[:], trace_y[:], 'm--', label='Trace of H')  

# Draw the base of the robot as a rectangle
width = 2*(q[-1, 1] - q[-1, 0])  
height = par['hl']      
rect = plt.Rectangle((q[-1, 0], 0), width, height, color='gray', alpha=0.5, label='Robot Base')
plt.gca().add_patch(rect)  

# Add labels and legend
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid()
plt.axis('equal')


# Base position, velocity and acceleration ------------------------------------
# Create a figure with 3 subplots 
fig, axs = plt.subplots(3, 1, figsize=(8, 12))  

# Plot base position
axs[0].plot(tt, q[:, 0], label=r'Base position: $x_A$')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Position on x [m]')
axs[0].legend()
axs[0].grid()

# Plot base velocity
axs[1].plot(tt, q_d[:, 0], label=r'Base velocity: $\dot{x}_A$')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Velocity on x [m/s]')
axs[1].legend()
axs[1].grid()

# Plot base acceleration
axs[2].plot(tt, q_dd[:, 0], label=r'Base acceleration: $\ddot{x}_A$')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Acceleration on x [m/$s^2$]')
axs[2].legend()
axs[2].grid()

fig.subplots_adjust(hspace=0.1)  

plt.show(block=False)

# Base displacement -----------------------------------------------------------
plt.figure()
plt.plot(tt, q[:, 0], label=r'Base position: $x_A$')
plt.plot(tt, q[:, 7], label=r'EF position: $x_H$')
plt.axhline(y=np.min(q[:,0]), color='r', linestyle='--', label=f'Min $x_A$ = {np.min(q[:,0]):.2f}m')
plt.axhline(y=np.max(q[:,7]), color='r', linestyle='--', label=f'Max $x_H$ = {np.max(q[:,7]):.2f}m')
plt.xlabel('Time [s]')
plt.ylabel('Position on x [m]')
plt.legend()
plt.grid()
plt.show(block=False)


plt.figure()
plt.plot(tt,xa,label='$ x_a(t)$')
plt.axhline(y=Max_xa, color='r', linestyle='--', linewidth=2, label=f'max of $x_a$ ={Max_xa:.2f}')
plt.axhline(y=Min_xa, color='r', linestyle='--', linewidth=2, label=f'min of $x_a$ ={Min_xa:.2f}')
plt.grid(True)
plt.xlabel('time [s] ')
plt.ylabel('diplasment along x axis [m] ')
plt.title('Displacment over time')
plt.legend()
plt.show(block=False)

plt.figure()
plt.plot(tt,xf,label='$ x_f(t)$')
plt.axhline(y=Max_xf, color='b', linestyle='--', linewidth=2, label=f'max of $ x_f $ ={Max_xf:.2f}')
plt.axhline(y=Min_xf, color='b', linestyle='--', linewidth=2, label=f'min $ x_f $ ={Min_xf:.2f}')
plt.grid(True)
plt.xlabel('time [s] ')
plt.ylabel('diplasment along x axis [m] ')
plt.title('horizontal diplacment of robot arms (F point) over time')
plt.legend()
plt.show()
