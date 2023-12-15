import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Sample coordinates (latitude, longitude, horizontal FoV, vertical FoV)
coordinates = [
    (-158.0, 10.0, 16.0, 40.0),
    (169.0, 72.0, 65.0, 51.0, 0.0),
]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Spherical coordinates
latitudes = [coord[0] for coord in coordinates]
longitudes = [coord[1] for coord in coordinates]

# Convert latitude and longitude to radians
latitudes = np.radians(latitudes)
longitudes = np.radians(longitudes)

# Calculate X, Y, Z coordinates on the unit sphere
x = np.cos(longitudes) * np.sin(latitudes)
y = np.sin(longitudes) * np.sin(latitudes)
z = np.cos(latitudes)

# Plot points on the sphere
ax.scatter(x, y, z)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()