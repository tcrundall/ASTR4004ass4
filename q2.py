#! /usr/bin/env python
import ah
import matplotlib.pyplot as plt
import numpy as np

a_true = [0,1,1,1,0]
a_false = [0.5,1,2,1,0]

times, v_with_errors = ah.simulate_v()
x=np.linspace(min(times), max(times), 100)

plt.plot(times, v_with_errors, 'o')
plt.plot(x, ah.v_func(a_true, x))
plt.savefig("figure1.pdf")
