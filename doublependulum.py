import numpy as np
import math
import matplotlib.pyplot as plt

##########################

# Define problem parameters
par = {
    'mass1': 1,
    'mass2': 1,
    'length1': 1,
    'length2': 1,
    'gravity': 10
}

# Defining mass matrix
epsilon = 1e-4
M = np.diag([par['mass1'],par['mass1'],0,par['mass2'],par['mass2'],0]) + np.eye(6) * epsilon

# Defining stiffness matrix
#K = np.diag([0,0,0,0,0,0])
K = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
# Defining force vector
f_ext = np.array([0, -par['mass1']*par['gravity'], 0, 0, -par['mass2']*par['gravity'],0])

# Define integrator parameters
h = 0.01
alpha = 0.015
gamma = 0.5 + alpha
beta = 0.25 * (gamma + 1/2)**2
tolQ = 1e-9
m = 4 # Number of constraints
k = 100000 # Constraint scaling factor
maxiter = 1002

t_0 = 0
t_f = 5
t = np.linspace(t_0, t_f, int((t_f-t_0)/h))

# Initializing variables
q = np.zeros((len(t),6))
q_d = np.zeros((len(t),6))
q_dd = np.zeros((len(t),6))
l = np.zeros((len(t),4))

# Define intial conditions
theta1 = 89 * (np.pi / 180)
theta2 = 0
q[0] = np.array([
    par['length1'] * np.sin(theta1),
    -par['length1'] * np.cos(theta1),
    theta1,
    par['length1'] * np.sin(theta1) + par['length2'] * np.sin(theta2),
    -par['length1'] * np.cos(theta1) - par['length2'] * np.cos(theta2),
    theta2
])
q_d[0] = ([0,0,0,0,0,0])
q_dd[0] = ([0,0,0,0,0,0])
l[0] = ([0,0,0,0])
###########################

# Evaluate residuals
def res(qt,qt_d, qt_dd, lt):
    r = M @ qt_dd + K @ qt_d + G(qt).T @ lt - f_ext
    
    gv = np.zeros(4)
    gv[0] = qt[0] - par['length1']*math.sin(qt[2])
    gv[1] = qt[1] - par['length1']*math.cos(qt[2])
    gv[2] = qt[3] - qt[0] - par['length2']*math.sin(qt[5])
    gv[3] = qt[4] - qt[1] - par['length2']*math.cos(qt[5])

    resc = np.concatenate([r,gv])
    return resc 

# Defining G matrix
def G(q):
    return np.array([
        [1,0,-par['length1']*math.cos(q[2]),0,0,0],
        [0,1,-par['length1']*math.sin(q[2]),0,0,0],
        [-1,0,0,1,0,-par['length2']*math.cos(q[5])],
        [0,-1,0,0,1,-par['length2']*math.sin(q[5])]
    ]) 

# Time loop
for i in range(len(t)-1):

    # Next step prediction
    q[i+1] = q[i] + h*q_d[i] + h**2*(0.5-beta)*q_dd[i]
    q_d[i+1] = q_d[i] + h*(1-gamma)*q_dd[i]
    q_dd[i+1] = 0
    l[i+1] = l[i]

    # Differential corrector loop
    j = 0
    # Initial guess
    qt = np.zeros((maxiter,6))
    qt_d = np.zeros((maxiter,6))
    qt_dd = np.zeros((maxiter,6))
    lt = np.zeros((maxiter,4))    
    qt[0] = q[i+1]
    qt_d[0] = q_d[i+1]
    qt_dd[0] = q_dd[i+1]
    lt[0] = l[i+1] 
    while np.linalg.norm(res(qt[j],qt_d[j],qt_dd[j],lt[j])) > tolQ:
        
        # Defining matrix S_t 
        first_block = 1/(beta*h**2)*M + K.T
        second_block = k*G(qt[j]).T
        top_row = np.hstack((first_block, second_block))  
        bottom_row = np.hstack((second_block.T, np.zeros((4,4))))   

        # Final matrix: concatenate top and bottom rows
        S_t = np.vstack((top_row, bottom_row))  
        print(np.linalg.cond(S_t))

        # Solve the linear system for Dql
        Dql = np.linalg.solve(S_t, -res(qt[j],qt_d[j],qt_dd[j],lt[j])) 

        # Update variables
        qt[j+1] = qt[j] + Dql[:6]
        qt_d[j+1] = qt_d[j] + gamma/(beta*h)*Dql[:6]
        qt_dd[j+1] = qt_dd[j] + 1/(beta*h**2)*Dql[:6]
        lt[j+1] = lt[j] + Dql[6:]

        print(Dql)

        if j == maxiter-2:
            print('Differential corrector did not converged')
            #exit()
            break
        
        j += 1

    print("Condition number of M:", np.linalg.cond(M))
    print(res(qt[j],qt_d[j],qt_dd[j],lt[j]))
    plt.figure()
    plt.plot(qt_dd[:,0])
    plt.plot(qt_dd[:,1])
    plt.plot(qt_dd[:,2])
    plt.plot(qt_dd[:,3])
    plt.plot(qt_dd[:,4])
    plt.plot(qt_dd[:,5])

    


    plt.figure()
    plt.plot(lt[:,0])
    plt.plot(lt[:,1])
    plt.plot(lt[:,2])
    plt.plot(lt[:,3])
    plt.xlabel('Iteration')
    plt.show()

    q[i+1] = qt[j]
    q_d[i+1] = qt_d[j]
    q_dd[i+1] = qt_dd[j]
    l[i+1] = lt[j]

# Plotting the double pendulum motion
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





