import numpy as np
import napari
import tifffile
from skimage.measure import label, regionprops
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import os

def gaussian_2d(xy_mesh, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """
    2D Gaussian function
    """
    (x, y) = xy_mesh
    x0 = float(x0)
    y0 = float(y0)
    gauss = offset + amplitude * np.exp(-(
        (x - x0)**2 / (2 * sigma_x**2) + 
        (y - y0)**2 / (2 * sigma_y**2)
    ))
    return gauss.ravel()

def fit_gaussian_around_centroid(image, centroid, window_size=7):
    """
    Fit a 2D Gaussian around a centroid in the image
    """
    y0, x0 = int(centroid[0]), int(centroid[1])
    half_window = window_size // 2
    
    # Extract region around centroid, handling image boundaries
    y_min = max(0, y0 - half_window)
    y_max = min(image.shape[0], y0 + half_window + 1)
    x_min = max(0, x0 - half_window)
    x_max = min(image.shape[1], x0 + half_window + 1)
    
    region = image[y_min:y_max, x_min:x_max]
    
    # Skip fitting if region is too small
    if region.shape[0] < 3 or region.shape[1] < 3:
        return centroid
    
    # Create mesh grid for the region
    y, x = np.indices(region.shape)
    x = x + x_min
    y = y + y_min
    xy_mesh = (x, y)
    
    # Initial guess for parameters
    initial_guess = [
        np.max(region) - np.min(region),  # amplitude
        x0,                               # x0
        y0,                               # y0
        2.0,                              # sigma_x
        2.0,                              # sigma_y
        np.min(region)                    # offset
    ]
    
    try:
        # Fit the 2D Gaussian
        popt, _ = curve_fit(
            gaussian_2d, 
            xy_mesh, 
            region.ravel(), 
            p0=initial_guess,
            bounds=([0, x_min, y_min, 0.5, 0.5, 0], 
                    [np.inf, x_max, y_max, 10, 10, np.max(region)])
        )
        
        # Return the refined centroid (y, x)
        return (popt[2], popt[1])
    
    except (RuntimeError, ValueError):
        # If fitting fails, return original centroid
        return centroid

def process_image(tif_path, threshold=50, window_size=7):
    # Load image
    image = tifffile.imread(tif_path)
    
    # Create binary mask
    binary_mask = image > threshold

    # Label regions
    labeled_image = label(binary_mask)
    
    # Get region properties
    regions = regionprops(labeled_image)
    
    # Extract centroids
    initial_centroids = [prop.centroid for prop in regions]
    
    # Refine centroids using Gaussian fitting
    refined_centroids = [fit_gaussian_around_centroid(image, centroid, window_size) 
                         for centroid in initial_centroids]
    
    return image, np.array(initial_centroids), np.array(refined_centroids)

def extract_pixel_values(image, centroids, offset_y=0, offset_x=0):
    """
    Extract pixel values from the image at the given centroid positions with offsets
    """
    values = []
    valid_centroids = []
    
    for centroid in centroids:
        y, x = centroid
        # Apply offset and round to nearest integer for indexing
        y_int = int(round(y + offset_y))
        x_int = int(round(x + offset_x))
        
        # Check if within image bounds
        if 0 <= y_int < image.shape[0] and 0 <= x_int < image.shape[1]:
            values.append(image[y_int, x_int])
            valid_centroids.append(centroid)
    
    return np.array(values), np.array(valid_centroids)

def reorganize_to_square_grid_bilinear(centroids, values, grid_size=None):
    """
    Reorganize hexagonal grid values to a square grid using bilinear interpolation
    """
    if len(values) == 0:
        # Return empty grid if no values
        return np.zeros((grid_size, grid_size)) if grid_size else np.zeros((1, 1))
    
    if grid_size is None:
        # Estimate grid size
        grid_size = int(np.ceil(np.sqrt(len(values))))
    
    # Extract x,y coordinates from centroids (note: centroids are in (y,x) format)
    points = centroids[:, [1, 0]]  # Convert to [x, y] for griddata
    
    # Create a regular grid for interpolation
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    
    # Create regular grid coordinates
    grid_x = np.linspace(x_min, x_max, grid_size)
    grid_y = np.linspace(y_min, y_max, grid_size)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    
    # Perform bilinear interpolation
    square_grid = griddata(
        points,             # Original points (x,y)
        values,             # Values at those points
        (grid_xx, grid_yy), # Target grid points
        method='linear',    # Bilinear interpolation
        fill_value=0        # Use 0 for points outside the convex hull
    )
    
    return square_grid

def generate_offset_coordinates(radius=4):
    """
    Generate a list of (y,x) offset coordinates in a circle of given radius
    """
    offsets = []
    
    # Generate all coordinates in a square of side length 2*radius+1
    for y in range(-radius, radius+1):
        for x in range(-radius, radius+1):
            # Include only points within radius
            if np.sqrt(y**2 + x**2) <= radius:
                offsets.append((y, x))
    
    return offsets

if __name__ == "__main__":
    # Process the calibration image to get centroids
    calibration_path = 'mla_calibration.tif'
    cal_image, initial_centroids, refined_centroids = process_image(calibration_path)
    
    # Load the light field demo image
    lightfield_path = 'light-field_demo.tif'
    lightfield_image = tifffile.imread(lightfield_path)
    
    # Create output directory if it doesn't exist
    output_dir = "lightfield_views"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate offset coordinates in a circle with radius 4
    offsets = generate_offset_coordinates(radius=4)
    
    # Create a list to store all the generated images for visualization
    all_images = []
    filenames = []
    
    # Set grid size
    grid_size = 31
    
    # Process each offset
    for offset_y, offset_x in offsets:
        # Extract pixel values at the offset positions
        pixel_values, valid_centroids = extract_pixel_values(
            lightfield_image, refined_centroids, offset_y, offset_x
        )
        
        # Skip if no valid centroids
        if len(valid_centroids) == 0:
            continue
        
        # Reorganize to square grid
        square_grid = reorganize_to_square_grid_bilinear(
            valid_centroids, pixel_values, grid_size
        )
        
        # Convert to appropriate data type
        if np.issubdtype(lightfield_image.dtype, np.integer):
            square_grid = np.round(square_grid).astype(lightfield_image.dtype)
        else:
            square_grid = square_grid.astype(lightfield_image.dtype)
            
        # Create filename with offset information
        # Use + or - signs for clarity
        sign_y = "+" if offset_y >= 0 else ""
        sign_x = "+" if offset_x >= 0 else ""
        filename = f"image_{sign_y}{offset_y}_{sign_x}{offset_x}.tif"
        filepath = os.path.join(output_dir, filename)
        
        # Save as TIFF
        tifffile.imwrite(filepath, square_grid)
        print(f"Saved image with offset y={offset_y}, x={offset_x} to {filepath}")
        
        # Store for visualization
        all_images.append(square_grid)
        filenames.append(f"y={sign_y}{offset_y}, x={sign_x}{offset_x}")
    
    print(f"Generated {len(all_images)} images in {output_dir} directory")
    
    # Visualize with napari
    viewer = napari.Viewer()
    
    # Add original images
    viewer.add_image(cal_image, name='Calibration Image')
    viewer.add_image(lightfield_image, name='Light Field Demo')
    
    # Add all generated images as a single stack
    if all_images:
        image_stack = np.stack(all_images)
        viewer.add_image(
            image_stack, 
            name='Offset Views', 
            colormap='viridis',
            channel_axis=0
        )
        
    # Add the points
    viewer.add_points(initial_centroids, name='Initial Centroids', size=4)
    viewer.add_points(refined_centroids, name='Refined Centroids', size=4)
    
    napari.run()