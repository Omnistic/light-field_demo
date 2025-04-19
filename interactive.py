import numpy as np
import matplotlib.pyplot as plt
import tifffile
from matplotlib.patches import Circle
import os
import glob
from scipy.interpolate import griddata

class LightFieldViewer:
    def __init__(self, image_dir="lightfield_views", radius=4):
        self.image_dir = image_dir
        self.radius = radius
        self.current_image = None
        self.images = {}
        self.dragging = False
        self.current_position = None
        self.last_position = None
        self.positions_array = None
        self.load_images()
        
        # Create figure and setup
        self.fig = plt.figure(figsize=(12, 6))
        self.setup_ui()
        
    def load_images(self):
        """Load all offset images from the directory"""
        print(f"Looking for images in {self.image_dir}...")
        
        # Find all tiff files
        image_files = glob.glob(os.path.join(self.image_dir, "image_*.tif"))
        
        if not image_files:
            print(f"No images found in {self.image_dir}")
            return
            
        print(f"Found {len(image_files)} images")
        
        for image_file in image_files:
            # Extract offset values from filename (format: image_+Y_+X.tif)
            filename = os.path.basename(image_file)
            parts = filename.replace("image_", "").replace(".tif", "").split("_")
            
            try:
                y_offset = int(parts[0])
                x_offset = int(parts[1])
                
                # Load image
                image = tifffile.imread(image_file)
                self.images[(y_offset, x_offset)] = image
                print(f"Loaded image with offset y={y_offset}, x={x_offset}")
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing filename {filename}: {e}")
        
        # Create array of positions for interpolation
        if self.images:
            self.positions_array = np.array(list(self.images.keys()))
    
    def setup_ui(self):
        """Set up the user interface with circle selector and image display"""
        # Circle selector on the left
        self.ax_circle = self.fig.add_subplot(121)
        self.ax_circle.set_aspect('equal')
        self.ax_circle.set_xlim(-self.radius-1, self.radius+1)
        self.ax_circle.set_ylim(-self.radius-1, self.radius+1)
        
        # Draw main circle
        circle = Circle((0, 0), self.radius, fill=False, color='blue')
        self.ax_circle.add_patch(circle)
        
        # Add dots for available positions
        for pos in self.images.keys():
            y, x = pos
            self.ax_circle.plot(x, y, 'o', markersize=8)
        
        self.ax_circle.set_title('Click and drag to explore views')
        self.ax_circle.set_xlabel('X offset')
        self.ax_circle.set_ylabel('Y offset')
        self.ax_circle.grid(True)
        
        # Image display on the right
        self.ax_image = self.fig.add_subplot(122)
        self.ax_image.set_title('Light Field View')
        self.image_obj = None
        
        # Add mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Show central image if available
        if (0, 0) in self.images:
            self.display_image((0, 0))
        elif len(self.images) > 0:
            # Display first available image
            first_key = list(self.images.keys())[0]
            self.display_image(first_key)
    
    def on_press(self, event):
        """Handle mouse press event"""
        if event.inaxes != self.ax_circle:
            return
        
        # Get click coordinates
        x, y = event.xdata, event.ydata
        
        # Check if within circle
        if x**2 + y**2 > self.radius**2:
            return
        
        # Start dragging
        self.dragging = True
        self.last_position = (y, x)
        
        # Update display based on position
        self.update_from_position(y, x)
    
    def on_release(self, event):
        """Handle mouse release event"""
        self.dragging = False
    
    def on_motion(self, event):
        """Handle mouse motion event"""
        if not self.dragging or event.inaxes != self.ax_circle:
            return
        
        # Get current coordinates
        x, y = event.xdata, event.ydata
        
        # Check if within circle
        if x**2 + y**2 > self.radius**2:
            # Clamp to circle edge if outside
            angle = np.arctan2(y, x)
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
        
        # Only update if position has changed significantly
        if self.last_position is None or \
           np.sqrt((y - self.last_position[0])**2 + (x - self.last_position[1])**2) > 0.05:
            self.last_position = (y, x)
            # Update display based on position
            self.update_from_position(y, x)
    
    def update_from_position(self, y, x):
        """Update display based on current mouse position"""
        # Use interpolation
        self.display_interpolated_image(y, x)
    
    def find_nearest_neighbors(self, y, x, k=4):
        """Find k nearest neighbors for interpolation"""
        if not self.images or len(self.positions_array) < k:
            return None, None
        
        # Calculate distances to all positions
        distances = np.sqrt((self.positions_array[:, 0] - y)**2 + 
                           (self.positions_array[:, 1] - x)**2)
        
        # Find k nearest indices
        indices = np.argsort(distances)[:k]
        neighbors = [tuple(self.positions_array[i]) for i in indices]
        neighbor_distances = distances[indices]
        
        return neighbors, neighbor_distances
    
    def display_interpolated_image(self, y, x):
        """Display interpolated image at the given position"""
        if not self.images:
            return
            
        try:
            # Use inverse distance weighted interpolation with 4 nearest neighbors
            neighbors, distances = self.find_nearest_neighbors(y, x, k=4)
            if neighbors is None:
                return
            
            # Compute weights based on inverse distance
            weights = 1.0 / (distances + 1e-10)  # Avoid division by zero
            weights /= np.sum(weights)  # Normalize
            
            # Interpolate images
            img_shape = next(iter(self.images.values())).shape
            interpolated_img = np.zeros(img_shape, dtype=np.float32)
            
            for i, neighbor in enumerate(neighbors):
                interpolated_img += weights[i] * self.images[neighbor]
            
            # Display the interpolated image
            self.ax_image.clear()
            self.image_obj = self.ax_image.imshow(interpolated_img, cmap='viridis')
            
            # Update circle display
            self.update_circle_display(y, x)
            
            # Update title
            self.ax_image.set_title(f'Interpolated View (y={y:.1f}, x={x:.1f})')
            
            # Refresh display
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Interpolation error: {e}")
            # Fall back to nearest neighbor
            neighbors, _ = self.find_nearest_neighbors(y, x, k=1)
            if neighbors:
                self.display_image(neighbors[0])
    
    def update_circle_display(self, y, x):
        """Update the circle display with current position"""
        self.ax_circle.clear()
        
        # Redraw circle
        circle = Circle((0, 0), self.radius, fill=False, color='blue')
        self.ax_circle.add_patch(circle)
        
        # Add dots for all positions
        for pos in self.images.keys():
            pos_y, pos_x = pos
            self.ax_circle.plot(pos_x, pos_y, 'o', markersize=8, color='blue')
        
        # Add current position marker
        self.ax_circle.plot(x, y, 'x', markersize=10, color='red')
        
        # Add title and labels
        self.ax_circle.set_title('Click and drag to explore views')
        self.ax_circle.set_xlabel('X offset')
        self.ax_circle.set_ylabel('Y offset')
        self.ax_circle.set_xlim(-self.radius-1, self.radius+1)
        self.ax_circle.set_ylim(-self.radius-1, self.radius+1)
        self.ax_circle.grid(True)
    
    def display_image(self, position):
        """Display the image at the given position"""
        if position not in self.images:
            print(f"No image available at position {position}")
            return
            
        # Clear previous image
        self.ax_image.clear()
        
        # Display new image
        image = self.images[position]
        self.image_obj = self.ax_image.imshow(image, cmap='viridis')
        
        # Update circle display
        y, x = position
        self.update_circle_display(y, x)
        
        # Update title
        self.ax_image.set_title(f'Light Field View (y={y}, x={x})')
        
        # Refresh display
        self.fig.canvas.draw_idle()

# Run the viewer
if __name__ == "__main__":
    viewer = LightFieldViewer()
    plt.tight_layout()
    plt.show()