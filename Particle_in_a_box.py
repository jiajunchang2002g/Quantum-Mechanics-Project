import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#constants
L = 10
sigma = 1
s = sigma / L                               # 0.1
hbar = 1
p0 = 1e2
p = L * p0 / hbar                           # good p is 1e3
m = 1
w0 = np.pi**2 * hbar / ( 2 * m * L**2 )     # good w0 is 1e-2
N = 45                                      # good N is 45 for good check_sum to 1e-5
ANIMATION_SPD = 0.25

def cn_integral(n, s, p):
    y = np.linspace(-0.5, 0.5, 70)  
    cos_term = np.cos(n * np.pi * y)
    exp_term = np.exp( -( np.power( ( y / (2 * s ) ), 2) ) )
    integrand = cos_term * np.cos(p * y) * exp_term
    constant = np.sqrt(np.sqrt(2) / np.sqrt(np.pi)) * np.sqrt(1 / s)
    return  constant * np.trapz(integrand, y)

def dn_integral(n, s, p):
    y = np.linspace(-0.5, 0.5, 70)  
    sin_term = np.sin(n * np.pi * y)
    exp_term = np.exp(-np.power( (y / (2 * s) ), 2 ) )
    integrand = 1j * sin_term * np.sin(p * y) * exp_term
    constant = np.sqrt(np.sqrt(2) / np.sqrt(np.pi)) * np.sqrt(1 / s)
    return constant * np.trapz(integrand, y)

def check_sum():
    sum = 0
    for i in range(1,N):
        cn = cn_integral(2 * i - 1, s, p)
        dn = dn_integral(2 * i, s, p)
        sum += np.abs(cn)**2 + np.abs(dn)**2 
    print(sum)

def u(n, x):
    cos_term = np.cos(n * np.pi * x / L)
    coeff = np.sqrt(2 / L)
    return coeff * cos_term
    
def v(n, x):
    sin_term = np.sin(n * np.pi * x / L)
    coeff = np.sqrt(2 / L)
    return coeff * sin_term

def f(x, t):
    sum = 0.0
    for i in range (1, N+1):
        odd_i = 2 * i - 1
        even_i = 2 * i 
        ci = cn[odd_i]
        di = dn[even_i]
        ui = u(odd_i, x)
        vi = v(even_i, x)
        # 3 steps to prevent overflow
        sum += np.power(np.abs(ci), 2) * np.power(ui,2) 
        sum += np.power(np.abs(di), 2) * np.power(vi, 2) 
        temp = 2 * ci * abs(di)
        temp *= ui * vi
        sum -= temp * np.sin( (even_i + odd_i) * w0 * t)
        # sum -= 2 * ci * abs(di) * ui * vi * np.sin( (even_i + odd_i) * w0 * t)
        for j in range (1, i):
            odd_j = 2 * j - 1
            even_j = 2 * j
            cj = cn[odd_j]
            dj = dn[even_j]
            uj = u(odd_j, x)
            vj = v(even_j, x)
            sum += 2 * ci * cj * ui * uj * np.cos( odd_i**2 - odd_j**2 ) * w0 * t
            sum += 2 * np.abs(di) * np.abs(dj) * vi * vj * np.cos( even_i**2 - even_j**2 ) * w0 * t
            sum -= 2 * ci * np.abs(dj) * ui * vj * np.sin( odd_i**2 - even_j**2 ) * w0 * t
            sum -= 2 * cj * np.abs(di) * uj * vi * np.sin( odd_j**2 - even_i**2 ) * w0 * t
    return sum

def animate():
    x_values = np.linspace(-L, L, 250)
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    def init():
        ax.set_xlim(-L, L)
        ax.set_ylim(0, 1)
        return line,
    def update(frame):
        t = frame * ANIMATION_SPD  
        y_values = f(x_values, t)
        line.set_data(x_values, y_values)
        return line,
    ax.plot(x_values, f(x_values, 0), color='gray', linestyle='--') # plot backgroud f(x,0)
    ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True)
    plt.show()

#initialise arrays

cn = np.loadtxt('cn.txt')
dn = np.loadtxt('dn.txt', dtype=complex)

# cn = np.zeros(2 * N)  
# dn = np.empty(2 * N + 1, dtype=complex)  
# for  i in range (1, N):
#     odd = 2 * i - 1
#     even = 2 * i
#     cn[odd] = cn_integral(odd, s, p)
#     dn[even] = dn_integral(even, s, p)

# print(w0)
# print(p)
# check_sum()
animate()