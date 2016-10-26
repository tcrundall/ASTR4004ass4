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


def get_motion(k=1, m=1, x0=1, N=100, T=20):
  q0 = np.array([1.0, 0.0])
  t0 = 0.0
  
  # Set up the integrator
  integrator = ode(f).set_integrator('lsoda')
  
  # Set the initial conditions
  integrator.set_initial_value(q0, t0).set_f_params([k,m])
  
  # Integrate
  dt = 0.05
  #dt = T*1.0/(N-1.0)
  nstep = int(T/dt)
  #nstep = N
  t = np.arange(1+nstep)*dt
  qsol = np.zeros((1+nstep,2))
  qsol[0,:] = q0
  
  for i in range(nstep):
    qsol[i+1,:] = integrator.integrate(integrator.t+dt)
  
  print("Integration went well")
  # Plot
  plt.plot(qsol[:,0], qsol[:,1])
  #plt.axis('equal')
  plt.show()
  return qsol

print(get_motion())
