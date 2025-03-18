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
j2,j3,j4,w,h,l2,l3,l4,m1,m2,m3,m4,hl,l,gravity,mu=sp.symbols('j2 j3 j4 w h l2 l3 l4 m1 m2 m3 m4 hl l ga mu')

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
            yb-hl,yb+l2*sp.sin(phic)-yd,
            xd+l3*sp.cos(phie)-xf,
            yd+l3*sp.sin(phie)-yf,
            xf+l4*sp.cos(phig)-xh,
            yf-l4*sp.sin(phig)-yh,
            x_t-xh,
            y_t-yh])
sG=sg.jacobian(sq)

# Mass matrix definition
sM=sp.diag(m1,m2,0,m3,m3,m4,m4,0,0,j2+m2*l2**2/4,j3+m3*l3**2/4,j4+m4*l4**2/4)
sM[1,9]=-m2*l2*sp.sin(phic)
sM[10,3]=-m3*l3*sp.sin(phie)
sM[4,10]=m3*l3*sp.cos(phie)
sM[5,11]=-m4*l4*sp.sin(phig)
sM[6,11]=m4*l4*sp.cos(phig)


# F_ext vector definition
sf_ext=sp.Matrix(12,1,[-sp.tanh(10*xa_dot)*mu*(m1+m2+m3+m4)*gravity,
                      0,
                      -(m1+m2)*gravity,
                      0,
                      -m3*gravity,
                      0,
                      -m4*gravity,
                      0,
                      0,
                      -m2*gravity*l2*0.5*sp.cos(phic),
                      -m3*gravity*l3*0.5*sp.cos(phie),
                      -m4*gravity*l4*0.5*sp.cos(phig)])

# f_ext derivative with respect to q
sf_ext_dq = sf_ext.jacobian(sq)

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

# Stiffness matrix definition
K=np.zeros((12,12))
K[9,9] = par['k1']
K[10,10] = par['k2']
K[11,11] = par['k3']


# Define integrator parameters
h = 0.01
alpha = 0.015
gamma = 0.5 + alpha
beta = 0.25 * (gamma + 1/2)**2
tolQ = 1e-9
k = 1# Constraint scaling factor
maxiter = 102

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

def M(q):
    # Define the numerical values for q
    q_values = {
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
        j2: par['j2'],
        j3: par['j3'],
        j4: par['j4'],
    }

    # Combine all substitutions
    sub = {**q_values, **param_values}

    # Evaluate the mass matrix numerically
    M_numeric = sM.subs(sub)

    return np.array(M_numeric).astype(np.float64) 

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

    # Mass matrix derivative with respect to q
    Mq = sp.Matrix([sM @ q_dd])
    if Mq == sp.zeros(Mq.rows, Mq.cols):
        Mq_d = sp.zeros(n, n)
    else:
        Mq_d = Mq.jacobian(sq)
    
    # Combine all substitutions
    sub = {**q_values, **param_values}

    # Evaluate the matrix numerically
    Kt = Mq_d.subs(sub) @ q_dd + dGl.subs(sub) - K - sf_ext_dq.subs(sub)

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

def f_ext(q,q_d):
    # Define the numerical values for q
    q_values = {
        xa_dot: q_d[0],
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
    r = M(q) @ q_dd + G(q).T @ l - K @ (q-q_0) - f_ext(q,q_d).T
    
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
        first_block = 1/(beta*h**2)*M(qt[j]) - gamma/(beta*h)*Ct(qt_d[j]) + Kt(qt[j],qt_dd[j],lt[j])
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

        print('Iteration:',j," Residual:",np.linalg.norm(res(qt[j],qt_d[j],qt_dd[j],lt[j],tt[i])))
        
        if j == maxiter-2:
            print('Differential corrector did not converged')
            #exit()
            break
        
        j += 1

    print(i)

    q[i+1] = qt[j]
    q_d[i+1] = qt_d[j]
    q_dd[i+1] = qt_dd[j]
    l[i+1] = lt[j]


# =============================================================================
## Plotting
# =============================================================================

# Plotting the end effectormotion
x1 = par['length1'] * np.sin(q[:, 2])
y1 = -par['length1'] * np.cos(q[:, 2])
x2 = x1 + par['length2'] * np.sin(q[:, 5])
y2 = y1 - par['length2'] * np.cos(q[:, 5])

plt.figure()
plt.plot(x1, y1, label='First Beam')
plt.plot(x2, y2, label='Second Beam')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Double Pendulum Motion')
plt.legend()
plt.grid()
plt.show()
