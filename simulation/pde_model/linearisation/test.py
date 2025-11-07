# %%
import numpy as np
from itertools import product

a = np.array([1, 2, 3])
b = 4.0
c = np.array([4, 5, 6, 7])
d = 6
e = 2

variables = [a, b, c, d, e]

# Créer une fonction lambda pour traiter dynamiquement chaque variable
def process_variable(var):
    if isinstance(var, np.ndarray):
        return np.atleast_1d(var)
    else:
        return [var]

# Appliquer la fonction lambda à chaque variable
processed_variables = list(map(process_variable, variables))

# Générer toutes les combinaisons possibles des valeurs des variables
combinaisons = list(product(*processed_variables))

# Construire le tableau combiné
combined_array = np.array(combinaisons)

print("Combined Array:")
print(combined_array)
print("Shape:", combined_array.shape)

# %%
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import numpy as np

# Initialiser R_pos en utilisant le format LIL
R_pos = sp.lil_matrix((1000, 1000), dtype='cfloat')
R_pos[-1, -1] = -1j * 5

# Utiliser scipy.sparse.bmat pour créer la matrice bloc
R = sp.bmat([[R_pos, R_pos], [R_pos, R_pos]], format='csr')

# Vérifiez la structure de la matrice R
print(R)

# Calcul des valeurs propres
w1, __ = eigsh(R)

# Affichage des valeurs propres
print("Valeurs propres de R:")
print(w1)
print(np.max(w1.real))

# %%
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt

# Initialiser R_pos en utilisant le format LIL
n = 1000
S_ind = 999
R_pos = sp.lil_matrix((n, n), dtype='cfloat')
R_pos[-1, -1] = -1j * 5
index_diag = np.arange(n)
R_pos[index_diag[S_ind:-1], index_diag[S_ind:-1]] = -1j*0.1 - 10 + 3 - 1
print(R_pos.shape)

# Utiliser scipy.sparse.bmat pour créer la matrice bloc
R = sp.bmat([[R_pos, R_pos], [R_pos, R_pos]], format='csr')

# %%
# Visualiser la structure de la matrice R
plt.figure(figsize=(10, 10))
plt.spy(R, markersize=1)
plt.title('Structure of Sparse Matrix R')
plt.show()

# %%
import numpy as np
n = 5
a = np.arange(n-1)
b = np.random.randint(low=0, high=10, size=(n, n))
print(b)

# %%
print(b[a[1:]+1, a[1:]])
print(b[(a+1)[1:], a[1:]])
# %%