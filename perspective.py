import napari
import tifffile
import numpy as np

# Path to your TIF file
tif_path = 'light-field_demo.tif'

# Load the TIF image
image = tifffile.imread(tif_path)

spec_x = 193
spec_y = 230

dx = 18
dy = 10

lattice_0_x = spec_x % dx
lattice_0_y = spec_y % dy

lattice_1_x = (spec_x+9) % dx
lattice_1_y = (spec_y+5) % dy

lattice_0 = []
lattice_1 = []

while lattice_0_x < 512:
    while lattice_0_y < 512:
        lattice_0.append([lattice_0_y, lattice_0_x])
        lattice_0_y += dy
    
    lattice_0_y = spec_y % dy
    lattice_0_x += dx

while lattice_1_x < 512:
    while lattice_1_y < 512:
        lattice_1.append([lattice_1_y, lattice_1_x])
        lattice_1_y += dy
    
    lattice_1_y = (spec_y+5) % dy
    lattice_1_x += dx

all_points = np.vstack((lattice_0, lattice_1))

# Create a napari viewer
viewer = napari.view_image(image, name='TIF Image')

# Add a points layer with the specific coordinates and customized appearance
points_layer = viewer.add_points(
    all_points,  # Coordinates of the points
    name='Marked Point',
    size=1,                    # Size set to 1
    face_color='cyan',         # Example color (change to your preferred color)
    border_color='transparent',  # Remove border by making edge color transparent
    border_width=0,              # Set edge width to 0 for no border
    opacity=0.7,               # Set transparency (0.0 is fully transparent, 1.0 is opaque)
    symbol='square'            # You can also use 'disc' for a circular point
)

# Run the napari application (this will block until the window is closed)
napari.run()