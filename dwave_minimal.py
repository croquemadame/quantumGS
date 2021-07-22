##################################################
# Import libraries
##################################################
#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#from threadpoolctl import threadpool_limits
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import neal
#import random
from dwave.cloud import Client
from dwave.system import DWaveSampler, EmbeddingComposite
#import dwave.inspector
import dimod
import pickle
#from memory_profiler import profile 
#import requests 

##################################################
# Inputs 
##################################################

# Size of rectangular plate
Lx = 1.0 # in meters
Ly = 1.0 # in meters

# Number of points in the rectangular grid
mx = 11
my = 11

# Spacing between adjacent points in grid
dx = Lx / (mx-1)
dy = Ly / (my-1)

# Dirichlet Boundary conditions
T_0_y = lambda y : 0.0
T_x_0 = lambda x : 0.0
T_x_Ly = lambda x : 100.0*(x/Lx)
T_Lx_y = lambda y : 100.0*(y/Ly)

# Parameters for solving the resulting linear system

# Size of the linear system
N = (mx-2)*(my-2)

# Number of iterations for the iterative QUBO LS solver
n_iter = 10

# Number of repetitions for the simulated annealing
n_rep = 1000

#Size of subsystems (L=NLxNL)
L = 9

# Minimum number of bits for representing
# solutions for the linear system
R_min = 7

# The algorithm will increase R by 'inc' in each iteration  (inc=0, disable)
inc = 0

# Initial interval for the unknowns (for Laplace equation these values are known from I.C. and B.C.)
# Upper bound
ub_init = 100.0
# Lower bound
lb_init = 0.0

# Initial guess for the temperature (mean of upper and lower bounds, could be something else)
T_guess = [(ub_init - lb_init)/2.0]*N


# The algorithm will shrink the interval of length 'le'  
# around the estimated solution at each step by a 
# factor given by 'sh'. Choosing a value of sh=1 results in a constant search interval
#le = (ub_init - lb_init)/2.0
le = (ub_init - lb_init)
#sh = 0.6
sh = 0.8
#sh = 1.0


#Heat Addition
#random.seed(9001) # we don't want any bias here.
#Q=np.random.randint(-100,100,size=N). If no heat sources are wanted, make Q=np.zeros(N).
#Qsave=Q
#Q=np.zeros(N)
Q=[-18,  16, -85,  16,  59,  84, -95, -99, -54,   9,  15, -26,  97,
        88,  64, -41, -20, -18, -88,  78,  -7,  42,  70,  47,  43, -86,
        -63, -12,  55,  -6,  22, -42,  12, -35,  58,  42,  42,  93,   5,
        -35,  24,  -3,  98,  70,  60,  57, -38, -18,  85,  23, -16, -75,
        47,  75,  -5,  29,  17, -91, -60, -34, -89,  71,  83,  24,  32,
        -33,  62,  61,  56,  51, -74, -30,  37,  -5, -98,  11, -73,  34,
          7, -40,  74]


##################################################
# Generate linear equations
##################################################


# Generate matrix A and vector b
A = np.zeros((N,N))
b = np.zeros(N)
for i in range(1,mx-1):
   for j in range(1,my-1):
      # new index: variables are numbered
      # from left to right and from bottom
      # to top
      l = i-1 + (j-1)*(mx-2)
      A[l][l] = -4.0
      if i == 1 :
         if j == 1 :
            b[l]           = - T_0_y(j*dy) - T_x_0(i*dx)-Q[l]
            A[l][l+1]      = 1.0
            A[l][l+(mx-2)] = 1.0            
         elif j == my - 2 :
            b[l]           = - T_0_y(j*dy) - T_x_Ly(i*dx)-Q[l]
            A[l][l-(mx-2)] = 1.0
            A[l][l+1]      = 1.0 
         else :
            b[l]           = - T_0_y(j*dy)-Q[l]
            A[l][l-(mx-2)] = 1.0
            A[l][l+1]      = 1.0
            A[l][l+(mx-2)] = 1.0  
      elif i == mx-2 :
         if j == 1 :
            b[l]           = - T_Lx_y(j*dy) - T_x_0(i*dx)-Q[l]
            A[l][l-1]      = 1.0
            A[l][l+(mx-2)] = 1.0
         elif j == my - 2 :
            b[l]           = - T_Lx_y(j*dy) - T_x_Ly(i*dx)-Q[l]
            A[l][l-(mx-2)] = 1.0
            A[l][l-1]      = 1.0
         else :
            b[l]           = - T_Lx_y(j*dy)-Q[l]
            A[l][l-(mx-2)] = 1.0
            A[l][l-1]      = 1.0
            A[l][l+(mx-2)] = 1.0
      else :
         if j == 1 :
            b[l]           = - T_x_0(i*dx)-Q[l]
            A[l][l-1]      = 1.0
            A[l][l+1]      = 1.0
            A[l][l+(mx-2)] = 1.0
         elif j == my - 2 :
            b[l]           = - T_x_Ly(i*dx)-Q[l]
            A[l][l-(mx-2)] = 1.0
            A[l][l-1]      = 1.0
            A[l][l+1]      = 1.0
         else :
            b[l]           = -Q[l]
            A[l][l-(mx-2)] = 1.0
            A[l][l-1]      = 1.0
            A[l][l+1]      = 1.0
            A[l][l+(mx-2)] = 1.0



##################################################
# Solving the Linear System - Iterative Method
##################################################
#@profile
def qubo_ls(M,y,lb,ub,N,R,n_rep):
   '''
   Solves a linear system Mx=y by mapping it into
   a QUBO problem.
   Linear system: Mx = y
   M : matrix for the linear system (np.matrix)
   b : column vector for the linear system (np.matrix)
   lb : list containing the lower bounds for the solution's interval
   ub : list containing the upper bounds for the solution's interval
   N : linear system size
   R : number of bits for representing each component 
       of the solution
   n_rep : number of repetitions for the annealing
   '''
   

# Define Solver (use simulated annealing if not using D-Wave systems). If using Dwave solver we can choose topology.
#   solver = neal.SimulatedAnnealingSampler()
#   solver = EmbeddingComposite(DWaveSampler(solver={'qpu': True}#))
#   solver = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'})) 
#   solver = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'},profile='denise')) 
#   solver = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'},profile='claudia')) 
#   solver = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'},profile='alexandra')) 
#   solver = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'},profile='maria')) 
#   solver = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'},profile='sonia')) 
#   solver = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'},profile='sonia')) 
#   solver = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'})) 
#   solver = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'},profile='duzzioni')) 
   solver = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'},profile='caio')) 

#   solver = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'},profile='duzzioni')) 


 #  solver = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus', 'postprocess': 'optimization'})) 


   # Interval: [-d,2*c - d)
   c = np.array([0.0]*N)
   d = np.array([0.0]*N)
   for i in range(0,N):
      d[i] = -lb[i]
      c[i] = (ub[i] + d[i])/2.0


   # Input QUBO coefficients
   a = np.zeros(N*R)
   b = np.zeros([N*R,N*R])

#   with threadpool_limits(limits=1,user_api='blas'):
   MtM = np.array(M.T.dot(M))
   Mty = np.array(M.T.dot(y))
   MtMd = MtM.dot(d.T)

   for l in range(0,N*R):
      r = l % R
      i = l // R
      a[l] = (-2) * (MtMd[i] + Mty[i]) * c[i] * (2**(-r))
      for h in range(0,N*R):
         s = h % R
         j = h // R
         b[l][h] = MtM[i][j] * c[i] * c[j] * (2**(-(r+s)))


   # QUBO coefficients
   Q = {}
   for l in range(0,N*R):
      for h in range(0,N*R):
         Q.update({(l,h): b[l][h] })
      Q[(l,l)] +=  a[l]


   # Get response from QUBO Solver   
   response = solver.sample_qubo(Q,num_reads=n_rep)


   # List result sample and energy
   # - the minimum energy corresponds to the answer to the problem
   sample = []
   energy = []
#print('Energy and samples for the Hamiltonian:')
#print('sample', ' | ' , 'energy')
   for s, e in response.data(['sample', 'energy']): 
#   print(s, ' | ',e)
      sample += [s]
      energy += [e]
#print('\n')


   ''''
   Find solutions (minimum energy). 
   This can be obtained directly using the dimod class, via response.first. The algorithm 
   below is more general.
   '''
   n_sol = 0
   index_sol = []
   for index in range(len(energy)):
      if energy[index] == min(energy):
         n_sol += 1
         index_sol += [index]


   # Write solutions - QUBO formulation
   x_sol = np.zeros([n_sol,N])
   for index in index_sol:
      for key, value in sample[index].items():
         x_sol[index][key // R] += 2**(- (key % R)) * value
      for i in range(N):
         x_sol[index][i] = c[i] * x_sol[index][i] - d[i]

   
   return np.matrix(x_sol[0]).T, response


#@profile
def qubo_iterative_solver_ls(A,b,N,x_guess,n_iter,L,R_min,le,sh):
   '''
   Solves a linear system using Gauss Seidel by blocks
   
   Ax=b
   A : matrix for the linear system
   b : vector for the linear system
   N : size of the linear system
   x_guess : initial guess for the solution
   n_iter : number of iterations for the iterative method
   L : size of subsystems
       (the last block is typically bigger than L if the 
        size of the system N is not divisible by L)
   le : initial length for the interval containing the solution
   sh : shrink factor for the interval containing the solution
        lengh = le * sh**it  (it = number of the iteration)
   
   '''
   master_response=[[[] for i in range(0,L)] for it in range(n_iter)]
   
   # Compute size of subsystems
   n = []
   for i in range(0,L-1):
      n += [N//L]
   n += [N - (L-1)*(N//L)]

   # Compute the indexes that separate each subsystem
   div = [0]
   for i in range(1,L+1):
      div += [div[i-1] + n[i-1]]

   # Partition the matrix A and the vector b 
   A_sub = [[] for i in range(0,L)]
   b_sub = [] 
   x_sub = []
   for i in range(0,L):
      b_sub += [b[div[i]:div[i+1]]]
      x_sub += [np.matrix(x_guess[div[i]:div[i+1]]).T]
      for j in range(0,L):
         A_sub[i] += [A[div[i]:div[i+1],div[j]:div[j+1]]]

   for it in range(n_iter):
      for i in range(0,L):
         b_new = np.matrix(b_sub[i]).T 
         for j in range(0,i):
            b_new += -A_sub[i][j].dot(x_sub[j])
         for j in range(i+1,L):
            b_new += -A_sub[i][j].dot(x_sub[j])
         #x_sub[i] = A_sub[i][i].I.dot(b_new) #This is cheating to me
         # Solve the 'i'-th linear system for iteration 'it'
         #R_sub = R_min + it #Increments precision at each iteration
         R_sub=R_min
         N_sub = len(A_sub[i][i])
         print('Iteration it={}, subsystem i={}, N_sub = {}, R_sub = {}'.format(it,i,N_sub,R_sub))
         print('Length of interval = {}'.format(le * (sh)**it))
         lb = []
         ub = []
         for k in range(0,n[i]):
            lb += [x_sub[i][k] - le/2.0 * (sh)**it]
            ub += [x_sub[i][k] + le/2.0 * (sh)**it]
         #print(lb,ub)
         # master_response is a matrix with all the solution for each subsystem. We keep it for posterior analysis.
         x_sub[i], master_response[it][i] = qubo_ls(A_sub[i][i],b_new,lb,ub,N_sub,R_sub,n_rep)
         #x_sub[i]=np.linalg.solve(A_sub[i][i],b_new) # In order to compare with classical solution
         #master_response[it][i]=x_sub[i]
         print('Solution for subsystem: {}'.format(x_sub))    


   x = np.vstack([ x_sub[i] for i in range(0,L) ])

   return x, master_response

# This is the actual call to the solver

T_ls, master_response = qubo_iterative_solver_ls(np.matrix(A),b.T,N,T_guess,n_iter,L,R_min,le,sh)

# Solution via matrix inversion

T_ls_mi = np.matrix(A).I.dot(np.matrix(b).T)

# Non-normalized error vector

error = T_ls - T_ls_mi

# This piece of code is so that we can pickle and save the data. 
# Otherwise we loose all the data from D-Wave upon code completion

master_s=[[[] for i in range(0,L)] for it in range(n_iter)]

for i in range(n_iter):
    for j in range(0,L):
        response=master_response[i][j]
        s=pickle.dumps(response.to_serializable())
        s_new=dimod.SampleSet.from_serializable(pickle.loads(s))
        master_s[i][j]=s_new
        
del(response,s_new,master_response)

# Now the data can be saved to disk via pickle or saved using "Save data" as spydata file if using spyder.

# Lets plot some results

##################################################
# Grid positions
##################################################

x = dx * np.arange(0,mx)
y = dy * np.arange(0,my)
x, y = np.meshgrid(x, y)

# Rewrite T for 3D plot (solution from iterative method)
T_grid = np.zeros((mx,my))
for i in range(1,mx-1):
  for j in range(1,my-1):
      l = i-1 + (j-1)*(mx-2)
      T_grid[i,j] = T_ls[l]

for i in range(0,mx):
  T_grid[i,0]  = T_x_0(i*dx)
  T_grid[i,my-1] = T_x_Ly(i*dx)

for j in range(0,my):
  T_grid[0,j]  = T_0_y(j*dy)
  T_grid[mx-1,j] = T_Lx_y(j*dy)


# Rewrite T for 3D plot (solution from matrix inversion)
T_grid_mi = np.zeros((mx,my))
for i in range(1,mx-1):
  for j in range(1,my-1):
      l = i-1 + (j-1)*(mx-2)
      T_grid_mi[i,j] = T_ls_mi[l]

for i in range(0,mx):
  T_grid_mi[i,0]  = T_x_0(i*dx)
  T_grid_mi[i,my-1] = T_x_Ly(i*dx)

for j in range(0,my):
  T_grid_mi[0,j]  = T_0_y(j*dy)
  T_grid_mi[mx-1,j] = T_Lx_y(j*dy)


##################################################
# Plot solution
##################################################

# Check Matplotlib docs
# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#getting-started


# Grid plot
fig_grid = plt.figure()
# Grid plot for Iterative Solution
ax_grid = fig_grid.add_subplot(1,1,1, projection='3d')
grid = ax_grid.plot_wireframe(x, y, T_grid, rstride=1, cstride=1)
ax_grid.set_xlabel('x (m)')
ax_grid.set_ylabel('y (m)')
ax_grid.set_zlabel('T (°C)')
plt.title('Solution for Heat Equation\n- Iterative QUBO Solver -')
'''
# Grid plot for Matrix Inversion Solution
ax_grid = fig_grid.add_subplot(1,2,2, projection='3d')
grid = ax_grid.plot_wireframe(x, y, T_grid_mi, rstride=1, cstride=1)
ax_grid.set_xlabel('x (m)')
ax_grid.set_ylabel('y (m)')
ax_grid.set_zlabel('T (°C)')
plt.title('Solution for Heat Equation\n- Matrix Inversion -')
'''
# Show grid plot
plt.show(grid)


# Color plot
fig_surf = plt.figure()
# Color plot for Iterative Solution
ax_surf = fig_surf.add_subplot(1,1,1, projection='3d')
surf = ax_surf.plot_surface(x, y, T_grid, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)
fig_surf.colorbar(surf, shrink=0.8, aspect=5, label='T (°C)')
ax_surf.set_xlabel('x (m)')
ax_surf.set_ylabel('y (m)')
ax_surf.set_zlabel('T (°C)')
plt.title('Solution for Heat Equation\n- Iterative QUBO Solver -')
'''
# Color plot for Matrix Inversion Solution
ax_surf = fig_surf.add_subplot(1,2,2, projection='3d')
surf = ax_surf.plot_surface(x, y, T_grid_mi, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)
fig_surf.colorbar(surf, shrink=0.8, aspect=5, label='T (°C)')
ax_surf.set_xlabel('x (m)')
ax_surf.set_ylabel('y (m)')
ax_surf.set_zlabel('T (°C)')
plt.title('Solution for Heat Equation\n- Matrix Inversion -')
'''
# Show color plot
plt.show(surf)


# Heatmap
fig_heatmap = plt.figure()
extent = [0, Lx, 0, Ly]
# Heatmap for Iterative Solution
ax_heatmap = fig_heatmap.add_subplot(1,1,1)
plt.title('Heatmap\n- Iterative QUBO Solver -')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
heatmap = plt.imshow(T_grid, extent=extent, cmap=cm.coolwarm, origin='lower')
plt.colorbar(heatmap, shrink=0.6, aspect=5, label='T (°C)')
'''
# Heatmap for Matrix Inversion Solution
ax_heatmap = fig_heatmap.add_subplot(1,3,3)
plt.title('Heatmap\n- Matrix Inversion -')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
heatmap = plt.imshow(T_grid_mi, extent=extent, cmap=cm.coolwarm, origin='lower')
plt.colorbar(heatmap, shrink=0.6, aspect=5, label='T (°C)')
'''
# Show heatmap
plt.show(heatmap)

'''
plt.savefig('plot.png', format='png')
plt.clf()
plt.cla()
plt.close()
'''
