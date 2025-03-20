import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import math

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

# #Coordinate's symbols
# Xa,Xb,Xd,Yd,Xf,Yf,Xh,Yh,PHIc,PHIe,PHIg=sp.symbols('x_a x_b x_d y_d x_f y_f x_h y_h phi_c phi_e phi_g')
# xad, xbd, xdd, ydd, xfd, yfd, xhd, yhd, phicd, phied, phigd = sp.symbols(
#    r'\dot{x_a}, \dot{x_b}, \dot{x_d}, \dot{y_d}, \dot{x_f}, \dot{y_f}, \dot{x_h}, \dot{y_h}, \dot{\phi_c}, \dot{\phi_e}, \dot{\phi_g}'
# )

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
par = {
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
alpha = 0.02
gamma = 0.5 + alpha
beta = 0.25 * (gamma + 1/2)**2
tolQ = 1e-9
kscal = 10000
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
    # Define the numerical values for q
    q_values = {
        phic: q[9],   # phic
        phie: q[10],  # phie
        phig: q[11],  # phig 
    }

    # Define the parameter values
    param_values = {
        l2: par['l2'],
        l3: par['l3'],
        l4: par['l4'],
    }

    # Combine all substitutions
    sub = {**q_values, **param_values}

    # Evaluate the G matrix numerically
    G = sG.subs(sub)

    return np.array(G).astype(np.float64)

def Kt(q,q_dd,l):
    # Define the numerical values for q
    q_values = {
        phic: q[9],   
        phie: q[10],  
        phig: q[11],  
    }

    # Define the parameter values
    param_values = {
        m1: par['m1'],
        m2: par['m2'],
        m3: par['m3'],
        m4: par['m4'],
        l2: par['l2'],
        l3: par['l3'],
        l4: par['l4'],
        j2: par['j2'],
        j3: par['j3'],
        j4: par['j4'],
        mu: par['mu'],
        gravity: par['gravity'],
    }

    # Derive the Gl matrix
    Gl = sp.Matrix([sG.T @ l])
    if Gl ==sp.zeros(Gl.rows, Gl.cols):
        dGl = sp.zeros(n, n)
    else:
        dGl = Gl.jacobian(sq)


    # Combine all substitutions
    sub = {**q_values, **param_values}

    # Evaluate the matrix numerically
    Kt =  dGl.subs(sub) + K 

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
    # Define the numerical values for q
    q_values = {
        xa_dot: q_d[0],
        f_n: l[1],
        phic: q[9],   # phic
        phie: q[10],  # phie
        phig: q[11],  # phig
    }

    # Define the parameter values
    param_values = {
        m1: par['m1'],
        m2: par['m2'],
        m3: par['m3'],
        m4: par['m4'],
        l2: par['l2'],
        l3: par['l3'],
        l4: par['l4'],
        mu: par['mu'],
        gravity: par['gravity'],
    }

    # Combine all substitutions
    sub = {**q_values, **param_values}

    # Evaluate the mass matrix numerically
    f_ext = sf_ext.subs(sub)

    return np.array(f_ext).astype(np.float64) 


# Evaluate residuals
def res(q,q_d, q_dd, l,tt):

    # Evaluate the equation of motion residuals
    r = M @ q_dd + G(q).T @ l + K @ (q-q_0) - f_ext(q,q_d,l).T

    
    # Evaluate the constraint residuals
    q_values = {
        t: tt,
        xa: q[0],
        xb: q[1],
        yb: q[2],
        xd: q[3],
        yd: q[4],
        xf: q[5],
        yf: q[6],
        xh: q[7],
        yh: q[8],
        phic: q[9],   
        phie: q[10],  
        phig: q[11],  
    }

    param_values = {
        m1: par['m1'],
        m2: par['m2'],
        m3: par['m3'],
        m4: par['m4'],
        l2: par['l2'],
        l3: par['l3'],
        l4: par['l4'],
        w: par['w'],
        hl: par['hl'],
        mu: par['mu'],
        gravity: par['gravity'],
    }
    
    sub = {**q_values, **param_values}
    g = sg.subs(sub)
    g = np.array(g.T).astype(np.float64)

    resc = np.concatenate([r.T,g.T])
    return resc 

# =============================================================================
## Motion Solver
# =============================================================================
plt.figure()

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
        first_block = 1/(beta*h**2)*M - gamma/(beta*h)*Ct(qt_d[j]) + Kt(qt[j],qt_dd[j],lt[j])
        second_block =  G(qt[j]).T
        top_row = np.hstack((first_block, second_block))  
        bottom_row = np.hstack((second_block.T, np.zeros((m,m))))   

        # Final matrix: concatenate top and bottom rows
        S_t = np.vstack((top_row, bottom_row))  
        print('s_t cond',np.linalg.cond(S_t))

        # Solve the linear system for Dql
        #Dql = np.linalg.solve(S_t, -res(qt[j],qt_d[j],qt_dd[j],lt[j])) 
        Dql= np.linalg.inv(S_t) @ -res(qt[j],qt_d[j],qt_dd[j],lt[j],tt[i])

        # Update variables
        qt[j+1] = qt[j] + Dql[:n].T
        qt_d[j+1] = qt_d[j] + gamma/(beta*h)*Dql[:n].T
        qt_dd[j+1] = qt_dd[j] + 1/(beta*h**2)*Dql[:n].T
        lt[j+1] = lt[j] + Dql[n:].T

        print('Iteration:',j," Residual:",np.linalg.norm(res(qt[j],qt_d[j],qt_dd[j],lt[j],tt[i])))
        
        if j == maxiter-2:
            print('Differential corrector did not converged')
            exit()
        
        j += 1

    print(i)

    # Update the trace of marker H
    trace_x.append(q[i, 7])  # x-coordinate of marker H
    trace_y.append(q[i, 8])  # y-coordinate of marker H


    # Plot the markers and connect them with lines
    plt.plot([q[i, 0], q[i, 1]], [0, q[i, 2]], 'r-', label='First Beam')  # Line from first to second marker
    plt.plot([q[i, 1], q[i, 3]], [q[i, 2], q[i, 4]], 'g-', label='Second Beam')  # Line from second to third marker
    plt.plot([q[i, 3], q[i, 5]], [q[i, 4], q[i, 6]], 'b-', label='Third Beam')   # Line from third to fourth marker
    plt.plot([q[i, 5], q[i, 7]], [q[i, 6], q[i, 8]], 'k-', label='Fourth Beam')  # Line from fourth to fifth marker

    # Plot the markers
    plt.plot(q[i, 0], 0, 'ro', label='A')  # First marker
    plt.plot(q[i, 1], q[i, 2], 'go', label='B')  # Second marker
    plt.plot(q[i, 3], q[i, 4], 'bo', label='D')   # Third marker
    plt.plot(q[i, 5], q[i, 6], 'ko', label='F')  # Fourth marker
    plt.plot(q[i, 7], q[i, 8], 'mo', label='H')   # Fifth marker

    # Plot the trace of marker H
    plt.plot(trace_x, trace_y, 'm--', label='Trace of H')  # Dashed magenta line for the trace


    # Add labels and legend
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Double Pendulum Motion')
    #plt.legend()
    plt.grid()
    plt.axis('equal')
    # Pause and clear for the next iteration
    plt.pause(h)
    plt.clf()
    
 

    q[i+1] = qt[j]
    q_d[i+1] = qt_d[j]
    q_dd[i+1] = qt_dd[j]
    l[i+1] = lt[j]


# =============================================================================
## Plotting
# =============================================================================


