import numpy as np
import tifffile
from skimage.measure import label, regionprops
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import os

def gaussian_2d(xy_mesh, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """2D Gaussian function"""
    (x, y) = xy_mesh
    x0, y0 = float(x0), float(y0)
    gauss = offset + amplitude * np.exp(-(
        (x - x0)**2 / (2 * sigma_x**2) + 
        (y - y0)**2 / (2 * sigma_y**2)
    ))
    return gauss.ravel()

def fit_gaussian_around_centroid(image, centroid, window_size=7):
    """Fit a 2D Gaussian around a centroid in the image"""
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
        x0, y0,                           # x0, y0
        2.0, 2.0,                         # sigma_x, sigma_y
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
    """Process calibration image to extract refined centroids"""
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
    """Extract pixel values from the image at the given centroid positions with offsets"""
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

def extract_integrated_pixel_values(image, centroids, radius=16):
    """Extract integrated pixel values within a given radius around each centroid"""
    values = []
    valid_centroids = []
    
    # Create a circular mask once
    y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
    circle_mask = x_grid**2 + y_grid**2 <= radius**2
    
    for centroid in centroids:
        y, x = centroid
        y_int, x_int = int(round(y)), int(round(x))
        
        # Initialize sum for this centroid - use float64 to prevent overflow
        integrated_value = 0.0
        
        # Get bounds for the region, ensuring we stay within image boundaries
        y_min = max(0, y_int - radius)
        y_max = min(image.shape[0], y_int + radius + 1)
        x_min = max(0, x_int - radius)
        x_max = min(image.shape[1], x_int + radius + 1)
        
        # Skip centroids too close to the edge
        if y_max - y_min < 3 or x_max - x_min < 3:
            continue
        
        # Extract the appropriate portion of the mask
        mask_y_min = radius - (y_int - y_min)
        mask_y_max = radius + (y_max - y_int)
        mask_x_min = radius - (x_int - x_min)
        mask_x_max = radius + (x_max - x_int)
        
        region_mask = circle_mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]
        region = image[y_min:y_max, x_min:x_max]
        
        # Verify shapes match before applying mask
        if region.shape == region_mask.shape:
            # Sum all pixel values within the circular region using vectorized operations
            integrated_value = float(np.sum(region * region_mask))
            values.append(integrated_value)
            valid_centroids.append(centroid)
    
    if not values:
        print("Warning: No valid integrated values found!")
        return np.array([]), np.array([])
    
    return np.array(values, dtype=np.float64), np.array(valid_centroids)

def reorganize_to_square_grid_bilinear(centroids, values, grid_size=51):
    """Reorganize hexagonal grid values to a square grid using bilinear interpolation"""
    if len(values) == 0:
        # Return empty grid if no values
        return np.zeros((grid_size, grid_size))
    
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

def generate_offset_coordinates(radius=16):
    """Generate a list of (y,x) offset coordinates in a circle of given radius"""
    offsets = []
    
    # Generate all coordinates in a square of side length 2*radius+1
    for y in range(-radius, radius+1):
        for x in range(-radius, radius+1):
            # Include only points within radius
            if np.sqrt(y**2 + x**2) <= radius:
                offsets.append((y, x))
    
    return offsets

def process_lightfield(calibration_path='mla_calibration.tif', 
                      lightfield_path='light-field_demo.tif',
                      output_dir='lightfield_views', 
                      radius=16,
                      grid_size=51,
                      generate_offsets=False, 
                      generate_integrated=True):
    """Main function to process lightfield images"""
    
    # Process the calibration image to get centroids
    print(f"Processing calibration image: {calibration_path}")
    _, _, refined_centroids = process_image(calibration_path)
    
    # Load the light field demo image
    print(f"Loading lightfield image: {lightfield_path}")
    lightfield_image = tifffile.imread(lightfield_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process offset views if requested
    if generate_offsets:
        print(f"Generating offset perspective views with radius {radius}...")
        
        # Generate offset coordinates in a circle with the specified radius
        offsets = generate_offset_coordinates(radius=radius)
        
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
            square_grid = np.round(square_grid).astype(np.uint8)
            
            # Create filename with offset information
            sign_y = "+" if offset_y >= 0 else ""
            sign_x = "+" if offset_x >= 0 else ""
            filename = f"image_{sign_y}{offset_y}_{sign_x}{offset_x}.tif"
            filepath = os.path.join(output_dir, filename)
            
            # Save as TIFF
            tifffile.imwrite(filepath, square_grid)
            print(f"Saved image with offset y={offset_y}, x={offset_x} to {filepath}")
    
    # Generate integrated image if requested
    if generate_integrated:
        print(f"Generating integrated image with radius {radius}...")
        integrated_values, valid_integrated_centroids = extract_integrated_pixel_values(
            lightfield_image, refined_centroids, radius=radius
        )
        
        # Skip if no valid centroids
        if len(valid_integrated_centroids) > 0:
            # Reorganize to square grid
            integrated_square_grid = reorganize_to_square_grid_bilinear(
                valid_integrated_centroids, integrated_values, grid_size
            )
            
            # Scale to 8-bit range
            max_value = np.max(integrated_square_grid)
            min_value = np.min(integrated_square_grid)
            print(f"Integrated values range: {min_value} to {max_value}")
            
            if max_value > min_value:  # Avoid division by zero
                # Scale to 0-255 range
                scaled_grid = (integrated_square_grid - min_value) * (255.0 / (max_value - min_value))
                integrated_square_grid = np.round(scaled_grid).astype(np.uint8)
                print(f"Scaled integrated image to 8-bit (0-255) range")
            else:
                integrated_square_grid = np.zeros(integrated_square_grid.shape, dtype=np.uint8)
            
            # Create filename for integrated image
            integrated_filename = "integrated_image.tif"
            integrated_filepath = os.path.join(output_dir, integrated_filename)
            
            # Save as 8-bit TIFF
            tifffile.imwrite(integrated_filepath, integrated_square_grid)
            print(f"Saved integrated image to {integrated_filepath}")
    
    print(f"Processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    # Change these flags to control what images to generate
    GENERATE_OFFSETS = True    # Set to True to generate offset perspective views
    GENERATE_INTEGRATED = True  # Set to True to generate integrated image
    
    # Call the main processing function with simple boolean flags
    process_lightfield(
        calibration_path='mla_calibration.tif',
        lightfield_path='light-field_demo.tif',
        output_dir='lightfield_views',
        radius=16,
        grid_size=51,
        generate_offsets=GENERATE_OFFSETS,
        generate_integrated=GENERATE_INTEGRATED
    )