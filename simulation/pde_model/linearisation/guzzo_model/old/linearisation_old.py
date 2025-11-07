# %%
import numpy as np
import compute_eigeinvalues
import parameters
par = parameters.Parameters()

def build_matrix(phi, CC, xi):
    """
    Olds matrix local density
    
    """
    # Matrices OFF
    R_pos = np.zeros((par.ns, par.ns), dtype='cfloat')
    R_A = np.zeros((par.ns, par.ns), dtype='cfloat')
    R_A = np.zeros((par.ns, par.ns), dtype='cfloat')
    R_neg = np.zeros((par.ns, par.ns), dtype='cfloat')

    # Matrices ON
    P_pos = np.zeros((par.ns, par.ns), dtype='cfloat')
    P_A = np.zeros((par.ns, par.ns), dtype='cfloat')
    P_A = np.zeros((par.ns, par.ns), dtype='cfloat')
    P_neg = np.zeros((par.ns, par.ns), dtype='cfloat')

    R = int(phi/par.ds)
    # Renormalisation for the sum of exponentials
    renorm = np.sum(par.ds * np.exp(-(np.arange(R,par.ns) - R) * par.ds))

    for i in range(1,par.ns):
        # Haut gauche
        R_pos[i,i] = 1j*xi + 1/par.ds*(i<par.ns-1)
        R_pos[i,i-1] = -1/par.ds
        if i >= R:
            R_pos[i,i] += 1
            for j in range(par.ns):
                R_pos[i,j] += CC * np.exp(-(i - R) * par.ds) * par.ds / renorm
                R_A[i,j] += CC * np.exp(-(i - R) * par.ds) * par.ds / renorm

        P_pos[i,i] = 1j*xi + 1/par.ds*(i<par.ns-1)
        P_pos[i,i-1] = -1/par.ds
        if i == R:
            P_pos[i,i] += 1
            for j in range(par.ns):
                P_pos[i,j] += CC * par.delta * par.ds
                P_A[i,j] += CC * par.delta * par.ds
        if i > R:
            P_pos[i,i] += 1

        # Bas par.dsoite
        R_neg[i,i] = -1j*xi + 1/par.ds*(i<par.ns-1)
        R_neg[i,i-1] = -1/par.ds
        if i >= R:
            R_neg[i,i] += 1
            for j in range(par.ns):
                R_neg[i,j] += CC * np.exp(-(i - R) * par.ds) * par.ds / renorm
                R_A[i,j] += CC * np.exp(-(i - R) * par.ds) * par.ds / renorm

        P_neg[i,i] = -1j*xi + 1/par.ds*(i<par.ns-1)
        P_neg[i,i-1] = -1/par.ds
        if i == R:
            P_neg[i,i] += 1
            for j in range(par.ns):
                P_neg[i,j] += CC * par.delta * par.ds
                P_A[i,j] += CC * par.delta * par.ds
        if i > R:
            P_neg[i,i] += 1

    # First line OFF 
    R_pos[0,0] = 1j*xi + 1/par.ds
    R_pos[0,:] -= CC

    R_A[0,:] -= CC
    R_A[0,R:] -= 1

    R_A[0,:] -= CC
    R_A[0,R:] -= 1

    R_neg[0,0] = -1j*xi + 1/par.ds
    R_neg[0,:] -= CC

    # First line ON
    P_pos[0,0] = 1j*xi + 1/par.ds
    P_pos[0,:] -= CC

    P_A[0,:] -= CC
    P_A[0,R:] -= 1

    P_A[0,:] = -CC
    P_A[0,R:] -= 1

    P_neg[0,0] = -1j*xi + 1/par.ds
    P_neg[0,:] -= CC

    # Concatenate OFF matrices
    A = np.concatenate((R_pos,R_A),axis = 1)
    B = np.concatenate((R_A,R_neg),axis = 1)
    M_off = np.concatenate((A,B),axis=0)

    # Concatenate ON matrices
    A = np.concatenate((P_pos,P_A),axis = 1)
    B = np.concatenate((P_A,P_neg),axis = 1)
    M_on = np.concatenate((A,B),axis=0)

    return M_off, M_on, R_pos, P_pos, R_A, P_A, R_A, P_A, R_neg, P_neg


def build_matrix_off(phi, CC, xi):
    """
    Olds matrix local density
    
    """
    # Matrices OFF
    R_pos = np.zeros((par.ns, par.ns), dtype='cfloat')
    R_A = np.zeros((par.ns, par.ns), dtype='cfloat')
    R_neg = np.zeros((par.ns, par.ns), dtype='cfloat')

    R = int(phi/par.ds)
    # Renormalisation for the sum of exponentials
    renorm = np.sum(par.ds * np.exp(-(np.arange(R,par.ns) - R) * par.ds))
    print(renorm)

    for i in range(1,par.ns):
    # for i in range(par.ns):
        # Haut gauche
        R_pos[i,i] = 1j*xi + 1/par.ds*(i<par.ns-1)
        R_pos[i,i-1] = -1/par.ds
        if i >= R:
            R_pos[i,i] += 1
            for j in range(par.ns):
                R_pos[i,j] += CC * np.exp(-(i - R) * par.ds) * par.ds / renorm
                R_A[i,j] += CC * np.exp(-(i - R) * par.ds) * par.ds / renorm

        # Bas par.dsoite
        R_neg[i,i] = -1j*xi + 1/par.ds*(i<par.ns-1)
        R_neg[i,i-1] = -1/par.ds
        if i >= R:
            R_neg[i,i] += 1
            for j in range(par.ns):
                R_neg[i,j] += CC * np.exp(-(i - R) * par.ds) * par.ds / renorm

    # First line OFF 
    R_pos[0,0] = 1j*xi + 1/par.ds
    R_pos[0,:] -= CC

    R_A[0,:] -= CC
    R_A[0,R:] -= 1

    R_neg[0,0] = -1j*xi + 1/par.ds
    R_neg[0,:] -= CC

    # Concatenate OFF matrices
    A = np.concatenate((R_pos,R_A),axis = 1)
    B = np.concatenate((R_A,R_neg),axis = 1)
    R = np.concatenate((A,B),axis=0)

    return -R, -R_pos, -R_neg, -R_A


def build_matrix_on(phi, CC, xi):
    """
    Olds matrix local density
    
    """
    # Matrices ON
    P_pos = np.zeros((par.ns, par.ns), dtype='cfloat')
    P_A = np.zeros((par.ns, par.ns), dtype='cfloat')
    P_neg = np.zeros((par.ns, par.ns), dtype='cfloat')

    R = int(phi/par.ds)
    # Renormalisation for the sum of exponentials

    for i in range(1,par.ns):
        P_pos[i,i] = 1j*xi + 1/par.ds*(i<par.ns-1)
        P_pos[i,i-1] = -1/par.ds
        if i == R:
            P_pos[i,i] += 1
            for j in range(par.ns):
                P_pos[i,j] += CC * par.delta * par.ds
                P_A[i,j] += CC * par.delta * par.ds
        if i > R:
            P_pos[i,i] += 1

        P_neg[i,i] = -1j*xi + 1/par.ds*(i<par.ns-1)
        P_neg[i,i-1] = -1/par.ds
        if i == R:
            P_neg[i,i] += 1
            for j in range(par.ns):
                P_neg[i,j] += CC * par.delta * par.ds
        if i > R:
            P_neg[i,i] += 1

    # First line ON
    P_pos[0,0] = 1j*xi + 1/par.ds
    P_pos[0,:] -= CC

    P_A[0,:] -= CC
    P_A[0,R:] -= 1

    P_neg[0,0] = -1j*xi + 1/par.ds
    P_neg[0,:] -= CC

    # Concatenate ON matrices
    A = np.concatenate((P_pos,P_A),axis = 1)
    B = np.concatenate((P_A,P_neg),axis = 1)
    P = np.concatenate((A,B),axis=0)

    return -P, -P_pos, -P_neg, -P_A

phi, CC , xi = 4.4, 0.2, 1.5
# M_off, M_on, R_pos, P_pos, R_A, P_A, R_A, P_A, R_neg, P_neg = build_matrix(phi, CC , xi)
P, P_pos, P_neg, P_A = build_matrix_on(phi, CC , xi)
R, R_pos, R_neg, R_A = build_matrix_off(phi, CC , xi)

print(np.sum(np.abs(np.sum(np.real(P), axis=0))))
print(np.sum(np.abs(np.sum(np.real(R), axis=0))))

# print('P_pos')
# print(np.real(P_pos))
print('R_pos')
print(np.real(R_pos))
# print('P_neg')
# print(np.real(P_neg))
print('R_neg')
print(np.real(R_neg))
# print('P_A')
# print(np.real(P_A))
print('R_A')
print(np.real(R_A))
# %%
lin = compute_eigeinvalues.Linearisation(a=1)
array_P_local, array_R_local, __ = lin.compute_eigeinvalues(f_matrix_p=build_matrix_on, f_matrix_r=build_matrix_off)
# %%
import matplotlib.pyplot as plt
import plot
plo = plot.Plot(fontsize=20, figsize=(10,10))
num_err = 10e-8
array_P_local[array_P_local<num_err] = np.nan
array_R_local[array_R_local<num_err] = np.nan

cmap = plt.get_cmap('plasma_r')
xmin = par.S_array[0]
xmax = par.S_array[-1]
ymin = par.C_array[0]
ymax = par.C_array[-1]
labelsize = 20
lambda_max = np.nanmax(array_P_local)
alpha = 1

fig, ax = plt.subplots()
plt.imshow(array_P_local, extent=[xmin,xmax,ymax,ymin], cmap=cmap, aspect=10, vmin=0, vmax=lambda_max, alpha=alpha)
ax.plot(par.S_array, lin.C_P(par.S_array), color='limegreen', linewidth=2, linestyle='dotted', alpha=1, label=r'$\~C=\frac{0.5\times\bar{S}}{1+\bar{S}}$')
ax.tick_params(labelsize=labelsize)
ax.set_xlabel(r'$\bar{S}=R^*\times \bar{F}$', fontsize=labelsize)
ax.set_ylabel(r'$\~C$', fontsize=labelsize)
plt.xticks(np.arange(0, xmax, step=2), fontsize=labelsize)
plt.yticks(np.arange(0, ymax, step=0.2), fontsize=labelsize)
plo.forceAspect(ax, aspect=1)
v1 = np.linspace(0, lambda_max, 4, endpoint=True)
cb = plt.colorbar(ticks=v1, shrink=1)
cb.ax.set_yticklabels(["{:3.1f}".format(i) for i in v1], fontsize='15')
plt.gca().invert_yaxis()
cb.set_label("$\lambda$", fontsize=labelsize)
plt.legend(fontsize=labelsize*0.7)

lambda_max = np.nanmax(array_R_local)

fig, ax = plt.subplots()
plt.imshow(array_R_local, extent=[xmin,xmax,ymax,ymin], cmap=cmap, aspect=10, vmin=0, vmax=lambda_max, alpha=alpha)
ax.plot(par.S_array, lin.C_R(par.S_array), color='limegreen', linewidth=2, linestyle='dotted', alpha=1, label=r'$\~C=\frac{0.5\times\bar{S}}{1+\bar{S}}$')
ax.tick_params(labelsize=labelsize)
ax.set_xlabel(r'$\bar{S}=R^*\times \bar{F}$', fontsize=labelsize)
ax.set_ylabel(r'$\~C$', fontsize=labelsize)
plt.xticks(np.arange(0, xmax, step=2), fontsize=labelsize)
plt.yticks(np.arange(0, ymax, step=0.2), fontsize=labelsize)
plo.forceAspect(ax, aspect=1)
v1 = np.linspace(0, lambda_max, 4, endpoint=True)
cb = plt.colorbar(ticks=v1, shrink=1)
cb.ax.set_yticklabels(["{:3.1f}".format(i) for i in v1], fontsize='15')
plt.gca().invert_yaxis()
cb.set_label("$\lambda$", fontsize=labelsize)
plt.legend(fontsize=labelsize*0.7)
# %%
