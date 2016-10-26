import pdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode
# take x = 0 to be the point of zero displacement

def f(t, q, args):
  # q[0] = x, q[1] = v_x,
  # args[0] = k, args[1] = m

  dqdt = np.zeros(2)
  dqdt[0] = q[1]
  dqdt[1] = -args[0] / args[1] * q[0]
  return dqdt


def get_motion(k=1, m=1, x0=1, T=20, N=100):
  q0 = np.array([x0, 0.0])
  t0 = 0.0
  
  # Set up the integrator
  integrator = ode(f).set_integrator('lsoda')
  
  # Set the initial conditions
  integrator.set_initial_value(q0, t0).set_f_params([k,m])
  
  # Integrate
  dt = 0.01
  #dt = T*1.0/(N-1.0)
  nstep = int(T/dt)
  #nstep = N
  t = np.arange(1+nstep)*dt
  qsol = np.zeros((1+nstep,2))
  qsol[0,:] = q0
  
  for i in range(nstep):
    qsol[i+1,:] = integrator.integrate(integrator.t+dt)
  
  result = np.zeros((N,3))
  # getting the N position and velocity values at equally spaced
  # time intervals.
  for i in range(N):
    result[i][0] = i*T/(N-1.0)
    closest_ix = np.ceil(result[i][0]/dt)
    result[i][1] = qsol[closest_ix][0]
    result[i][2] = qsol[closest_ix][1]

  return result 

def xsol(t, k=1.0, m=1.0, x0=1):
  #return np.sin(np.sqrt(k/m)*(t + np.sqrt(m/k)*np.arcsin(x0)))
  return x0 * np.cos(np.sqrt(k/m)*t)

number_of_points = 100
max_time = 5

ts = np.linspace(0,max_time,10*number_of_points)

#plotting results for k=50, x0=0.1, m=1.0
plt.plot(ts, xsol(ts, 50.0, 1.0, 0.1), label="Analytical position")
result = get_motion(50.0, 1.0, 0.1, max_time, number_of_points)
plt.plot(result[:,0], result[:,1], label="Simulated position")
plt.plot(result[:,0], result[:,2], label="Simulated velocity")
plt.legend()
plt.xlabel("Times (s)")
plt.ylabel("Velocity (m/s) and postion (m)")
plt.savefig("linearspring.png")
plt.show()
