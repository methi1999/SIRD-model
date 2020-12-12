from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

T = 200
N = 100
S0, I0, R0, D0 = N-1, 1, 0, 0
beta, gamma, delta = 0.35, 0.3, 0.1

def SIRD_non_graph():
    
    # References: https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/    
    # A grid of time points (in days)
    
    t = np.linspace(0, T, T)

    # The SIR model differential equations.
    def deriv(y, t, N, beta, gamma, delta):
        S, I, R, D = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I - delta * I
        dRdt = gamma * I
        dDdt = delta * I
        return dSdt, dIdt, dRdt, dDdt

    # Initial conditions vector
    y0 = S0, I0, R0, D0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, delta))

    return ret.T

results = SIRD_non_graph()
print(results)
t = list(range(results.shape[1]))

colors = ['blue', 'red', 'green', 'black']
lines = [Line2D([0], [0], color=c, linewidth=3) for c in colors]
labels = ['Susceptible', 'Infected', 'Recovered', 'Dead']
color_codes = {'S': 'b', 'I': 'r', 'R': 'g', 'D': 'k'}


fig = plt.figure(facecolor='w', figsize=(8, 6))
ax = fig.add_subplot(111, axisbelow=True)

ax.legend(lines, labels)

ax.plot(t, results[0]/N, color_codes['S'], alpha=0.5, lw=2)
ax.plot(t, results[1]/N, color_codes['I'], alpha=0.5, lw=2)
ax.plot(t, results[2]/N, color_codes['R'], alpha=0.5, lw=2)
ax.plot(t, results[3]/N, color_codes['D'], alpha=0.5, lw=2)

plt.title("Beta = {}, Gamma = {}, Delta = {}".format(beta, gamma, delta))

ax.set_xlabel('Time step')
ax.set_ylabel('Fraction')

ax.set_ylim(1e-7, 2)
ax.set_yscale('log')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
# ax.grid(b=True, which='major', c='w', lw=2, ls='-')
ax.grid(True)

# for spine in ('top', 'right', 'bottom', 'left'):
    # ax.spines[spine].set_visible(False)
plt.show()
