import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad

hbar = 1.0
m = 1.0
p0 = 10
x0 = 0.0
sigma = 1
L = sigma*np.sqrt(-8*np.log(1e-10*sigma*np.sqrt(2*np.pi)))
n_max = 60  
t_max = 120
t_steps = 1000

def UE_odd(n, x, L):
    return np.sqrt(2/L) * np.cos(n * np.pi * x / L)

def UE_even(n, x, L):
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

def energy_n(n, L, hbar, m):
    return (np.pi**2) * (hbar**2) * (n**2) / (2 * m * (L**2))

def psi_0(x, p0, sigma, x0, hbar):
    return (1/(np.sqrt(np.sqrt(2*np.pi)*sigma))) * np.exp(1j * p0 * x / hbar) * np.exp(- (x - x0)**2 / (4 * (sigma**2)))

    
def coefficient_odd(n, p0, sigma, L, hbar): #c_n
    return (np.sqrt(np.sqrt(2*np.pi)/(L*np.pi*sigma))*quad(lambda x: np.cos((n*np.pi*x)/L)*np.cos(p0*x/hbar)*np.exp(-(x/(2*sigma))**2),-L/2,L/2)[0])

    # quad returns a tuple: 
    # (results of the integral, absolute error of the integral)

def coefficient_even(n, p0, sigma, L, hbar): #d_n
    return (1j*np.sqrt(np.sqrt(2*np.pi)/(L*np.pi*sigma))*quad(lambda x: np.sin((n*np.pi*x)/L)*np.sin(p0*x/hbar)*np.exp(-(x/(2*sigma))**2),-L/2,L/2)[0])

def wavefunction(x, t, n_max, p0, sigma,  L, hbar, m):
    #psi = np.zeros_like(x, dtype=complex) 
    psi = 0
    for n in range(1, n_max+1):
        E_n = energy_n(n, L, hbar, m)

        if n % 2 != 0:  # Odd
            c_n = coefficient_odd(n, p0, sigma, L, hbar)
            psi += c_n * UE_odd(n, x, L) * np.exp(-1j * E_n * t)

        else:  # Even
            d_n = coefficient_even(n, p0, sigma, L, hbar)
            psi += d_n * UE_even(n, x, L) * np.exp(-1j * E_n * t)
    return psi

def probability_density(x, t, n_max, p0, sigma,  L, hbar, m):
    psi = wavefunction(x, t, n_max, p0, sigma,  L, hbar, m)
    return np.conj(psi) * psi

def animate():
    x_values = np.linspace(-L/2, L/2, 250)
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    plt.gcf().axes[0].axvspan(L/2, L/2+2, alpha=0.2, color='dodgerblue')
    plt.gcf().axes[0].axvspan(-L/2-2, -L/2, alpha=0.2, color='dodgerblue')
    title = ax.set_title(f'Probability Density at t = {0:.2f} seconds')
    def init():
        ax.set_xlim(-L, L)
        ax.set_ylim(0, 1)
        return line,
    def update(frame):
        t = frame * 0.25  
        title.set_text(f'Probability Density at t = {t:.2f} seconds')
        y_values = probability_density(x_values, t, n_max, p0, sigma,  L, hbar, m)
        line.set_data(x_values, y_values)
        return line, title
    ax.plot(x_values, probability_density(x_values, 0, n_max, p0, sigma,  L, hbar, m), color='gray', linestyle='--') 
    ani = FuncAnimation(fig, update, frames=range(100), init_func=init)
    plt.show()

animate()