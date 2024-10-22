# %%
import numpy as np
import matplotlib.pyplot as plt

def extract_bins_in_circle(data, grid, r):
    coords_list = []
    for i in range(data.shape[1]):
        x, y = data[0, i], data[1, i]
        center_x, center_y = int(x * grid.shape[0]), int(y * grid.shape[1])
        indices = np.indices(grid.shape)
        distances = np.sqrt((indices[0] - center_x)**2 + (indices[1] - center_y)**2)
        mask = distances <= r
        coords = np.transpose(np.where(mask))
        coords_list.append(coords)
    return coords_list

# Exemple d'utilisation
data = np.random.uniform(size=(2, 10))  # Exemple de données
grid = np.zeros((2000, 2000))  # Exemple de grille
r = 50  # Rayon du cercle

coords_list = extract_bins_in_circle(data, grid, r)
# Affichage des bins sous forme d'un histogramme 2D
x_circle = np.concatenate(coords_list)[:,0]
y_circle = np.concatenate(coords_list)[:,1]

fig, ax = plt.subplots(figsize=(10, 10))
plt.hist2d(x_circle, y_circle, bins=[grid.shape[0], grid.shape[1]], range=[[0, grid.shape[0]], [0, grid.shape[1]]], cmap='Blues')
plt.colorbar()
ax.scatter(data[0]*grid.shape[0], data[1]*grid.shape[1], c='red', marker='.', s=100)
ax.set_aspect('equal', adjustable='box')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def extract_bins_in_half_circle(data, grid, r, direction, angle_view):
    
    coords_list = []
    indices = np.indices(grid.shape)

    for i in range(data.shape[1]):
        x, y = data[0, i], data[1, i]
        center_x, center_y = int(x * grid.shape[0]), int(y * grid.shape[1])
        distances = np.sqrt((indices[0] - center_x)**2 + (indices[1] - center_y)**2)
        mask = distances <= r
        coords = np.transpose(np.where(mask))
        
        # Appliquer la direction pour extraire les bins du demi-cercle dans cette direction
        direction_rad = direction[i]

        # Filtrer les bins en fonction de l'angle de la direction
        scalar_product = (coords[:, 0] - center_x) * np.cos(direction_rad) + (coords[:, 1] - center_y) * np.sin(direction_rad)
        mask_direction = scalar_product >= np.cos(angle_view / 2) * distances[mask]
        
        print(coords.shape)
        print(mask_direction.shape)

        coords = coords[mask_direction]
        
        coords_list.append(coords)
    
    return coords_list


# Exemple d'utilisation
data = np.random.uniform(size=(2, 3))  # Exemple de données
grid = np.zeros((2000, 2000))  # Exemple de grille
r = 50  # Rayon du cercle
direction = np.random.uniform(-np.pi, np.pi, size=data.shape[1])  # Exemple de directions aléatoires en radians
angle_view = np.pi

print(data)
print(direction)

coords_list = extract_bins_in_half_circle(data, grid, r, direction, angle_view)
# Affichage des bins sous forme d'un histogramme 2D
x_circle = np.concatenate(coords_list)[:, 0]
y_circle = np.concatenate(coords_list)[:, 1]

fig, ax = plt.subplots(figsize=(10, 10))
hist,__,__ = np.histogram2d(x_circle, y_circle, bins=[grid.shape[0], grid.shape[1]], range=[[0, grid.shape[0]], [0, grid.shape[1]]])
plt.imshow(np.rot90(hist), extent=(0, grid.shape[0], 0, grid.shape[1]), cmap='Blues')
plt.colorbar()
ax.scatter(data[0]*grid.shape[0], data[1]*grid.shape[1], c='red', marker='.', s=100)
ax.set_aspect('equal', adjustable='box')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def extract_bins_in_half_circle(data, grid, r, direction, angle_view):
    x = data[0, :]
    y = data[1, :]
    center_x = (x * grid.shape[0]).astype(int)
    center_y = (y * grid.shape[1]).astype(int)
    indices = np.indices(grid.shape)
    distances = np.sqrt((indices[0] - center_x[:, np.newaxis, np.newaxis])**2 + (indices[1] - center_y[:, np.newaxis, np.newaxis])**2)
    mask = distances <= r
    coords = np.transpose(np.where(mask))

    direction_rad = direction[:, np.newaxis]
    scalar_product = (coords[:, 0] - center_x[:, np.newaxis, np.newaxis]) * np.cos(direction_rad) + (coords[:, 1] - center_y[:, np.newaxis, np.newaxis]) * np.sin(direction_rad)
    mask_direction = scalar_product >= np.cos(angle_view / 2) * distances[mask]
    print(coords.shape)
    print(mask_direction.shape)

    coords_list = np.split(coords[mask_direction], len(data[0]), axis=1)
    
    return coords_list


# Exemple d'utilisation
data = np.random.uniform(size=(2, 3))  # Exemple de données
grid = np.zeros((2000, 2000))  # Exemple de grille
r = 50  # Rayon du cercle
direction = np.random.uniform(-np.pi, np.pi, size=data.shape[1])  # Exemple de directions aléatoires en radians
angle_view = np.pi/2

print(data)
print(direction)

coords_list = extract_bins_in_half_circle(data, grid, r, direction, angle_view)
# Affichage des bins sous forme d'un histogramme 2D
x_circle = np.concatenate(coords_list)[:, 0]
y_circle = np.concatenate(coords_list)[:, 1]

fig, ax = plt.subplots(figsize=(10, 10))
hist, __, __ = np.histogram2d(x_circle, y_circle, bins=[grid.shape[0], grid.shape[1]], range=[[0, grid.shape[0]], [0, grid.shape[1]]])
plt.imshow(np.rot90(hist), extent=(0, grid.shape[0], 0, grid.shape[1]), cmap='Blues')
plt.colorbar()
ax.scatter(data[0]*grid.shape[0], data[1]*grid.shape[1], c='red', marker='.', s=100)
ax.set_aspect('equal', adjustable='box')
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

def extract_bins_in_half_circle(data, grid, r, direction, angle_view):
    x = data[0, :]
    y = data[1, :]
    center_x = (x * grid.shape[0]).astype(int)
    center_y = (y * grid.shape[1]).astype(int)
    indices = np.indices(grid.shape)
    distances = np.sqrt((indices[0] - center_x[:, np.newaxis, np.newaxis])**2 + (indices[1] - center_y[:, np.newaxis, np.newaxis])**2)
    mask = distances <= r
    coords = np.transpose(np.where(mask))

    direction_rad = direction[:, np.newaxis]
    scalar_product = (coords[:, 0] - center_x[:, np.newaxis, np.newaxis]) * np.cos(direction_rad) + (coords[:, 1] - center_y[:, np.newaxis, np.newaxis]) * np.sin(direction_rad)
    mask_direction = scalar_product >= np.cos(angle_view / 2) * distances[mask]
    
    # Ajustement de mask_direction pour avoir les mêmes dimensions que coords
    mask_direction_adjusted = np.swapaxes(mask_direction, 0, 2)

    print(coords.shape)
    print(mask_direction_adjusted.shape)
    print(coords[mask_direction_adjusted,:])

    coords_list = np.split(coords[mask_direction_adjusted], len(data[0]), axis=1)
    
    return coords_list


# Exemple d'utilisation
data = np.random.uniform(size=(2, 3))  # Exemple de données
grid = np.zeros((2000, 2000))  # Exemple de grille
r = 50  # Rayon du cercle
direction = np.random.uniform(-np.pi, np.pi, size=data.shape[1])  # Exemple de directions aléatoires en radians
angle_view = np.pi/2

print(data)
print(direction)

coords_list = extract_bins_in_half_circle(data, grid, r, direction, angle_view)
# Affichage des bins sous forme d'un histogramme 2D
x_circle = np.concatenate(coords_list)[:, 0]
y_circle = np.concatenate(coords_list)[:, 1]

fig, ax = plt.subplots(figsize=(10, 10))
hist, __, __ = np.histogram2d(x_circle, y_circle, bins=[grid.shape[0], grid.shape[1]], range=[[0, grid.shape[0]], [0, grid.shape[1]]])
plt.imshow(np.rot90(hist), extent=(0, grid.shape[0], 0, grid.shape[1]), cmap='Blues')
plt.colorbar()
ax.scatter(data[0]*grid.shape[0], data[1]*grid.shape[1], c='red', marker='.', s=100)
ax.set_aspect('equal', adjustable='box')
plt.show()



#%%
import numpy as np

def extract_squares(matrix, centers, size):
    num_centers = centers.shape[0]
    x_coords = centers[:, 0]
    y_coords = centers[:, 1]
    
    x_starts = x_coords - size // 2
    y_starts = y_coords - size // 2
    
    strides = matrix.strides  # Strides de la matrice d'origine
    square_views = np.lib.stride_tricks.as_strided(
        matrix,
        shape=(num_centers, size, size),
        strides=(strides[0], strides[0] * matrix.shape[1], strides[1])
    )
    
    return square_views

# Exemple d'utilisation
matrix = np.random.randint(0, 10, (10, 10))
centers = np.array([(5, 5), (3, 7)])
centers = np.array([(5, 5), (3, 7), (0,0)])
size = 3

squares = extract_squares(matrix, centers, size)

print("Matrice d'origine :")
print(matrix)
print("\nCarrés extraits :")
print(squares)


#%%
import numpy as np
def extract_local_squared(matrix, centers, size):
    """
    Extract and sum values of 
    
    """
    v = np.lib.stride_tricks.sliding_window_view(matrix, (size*2-1,size*2-1))
    results = v[centers[0, :], centers[1, :], :, :]

    return results


pili_length = 3 # in bins
edges_width = pili_length
size_matrix = 10
matrix = np.random.randint(0, 10, (size_matrix+(edges_width-1)*2, size_matrix+(edges_width-1)*2))
centers = np.array([(0, -1, 3), 
                    (0, -1, 5)])

results = extract_local_squared(matrix, centers, edges_width)

print(matrix)
print(results)
# print(v)







# %%
count = []
for i in range(len(coords_list)):
    count.append(len(coords_list[i]))
print(count)




#%%
import numpy as np
import matplotlib.pyplot as plt

def extract_bins_in_half_circle(data, grid, r, direction, angle_view):
    
    coords_list = []
    indices = np.indices(grid.shape)

    for i in range(data.shape[1]):
        x, y = data[0, i], data[1, i]
        center_x, center_y = int(x * grid.shape[0]), int(y * grid.shape[1])
        distances = np.sqrt((indices[0] - center_x)**2 + (indices[1] - center_y)**2)
        mask = distances <= r
        coords = np.transpose(np.where(mask))
        
        # Appliquer la direction pour extraire les bins du demi-cercle dans cette direction
        direction_rad = direction[i]

        # Filtrer les bins en fonction de l'angle de la direction
        scalar_product = (coords[:, 0] - center_x) * np.cos(direction_rad) + (coords[:, 1] - center_y) * np.sin(direction_rad)
        mask_direction = scalar_product >= np.cos(angle_view / 2) * distances[mask]
        
        print(coords.shape)
        print(mask_direction.shape)

        coords = coords[mask_direction]
        
        coords_list.append(coords)
    
    return coords_list


# Exemple d'utilisation
data = np.random.uniform(size=(2, 3))  # Exemple de données
grid = np.zeros((2000, 2000))  # Exemple de grille
r = 50  # Rayon du cercle
direction = np.random.uniform(-np.pi, np.pi, size=data.shape[1])  # Exemple de directions aléatoires en radians
angle_view = np.pi

print(data)
print(direction)

coords_list = extract_bins_in_half_circle(data, grid, r, direction, angle_view)
# Affichage des bins sous forme d'un histogramme 2D
x_circle = np.concatenate(coords_list)[:, 0]
y_circle = np.concatenate(coords_list)[:, 1]

fig, ax = plt.subplots(figsize=(10, 10))
hist,__,__ = np.histogram2d(x_circle, y_circle, bins=[grid.shape[0], grid.shape[1]], range=[[0, grid.shape[0]], [0, grid.shape[1]]])
plt.imshow(np.rot90(hist), extent=(0, grid.shape[0], 0, grid.shape[1]), cmap='Blues')
plt.colorbar()
ax.scatter(data[0]*grid.shape[0], data[1]*grid.shape[1], c='red', marker='.', s=100)
ax.set_aspect('equal', adjustable='box')
plt.show()

# %%
import numpy as np
data = np.ones((2, 1000), dtype=np.float32)
a, _, _ = np.histogram2d(data[0], data[1])
a = a.astype(np.float32)
# print(a.dtype)
# print(data[0].dtype)
b = np.zeros((10, 10))
print(a.dtype)
a += b
print(a.dtype)
print(b.dtype)

# %% 
import numpy as np

a = np.ones(10) * 1.2e-45
print(a)
print(a.astype(np.float32))