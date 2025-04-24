import numpy as np
import matplotlib.pyplot as plt
import tifffile
from matplotlib.patches import Circle
import os
import glob
from scipy.ndimage import gaussian_filter

class LightFieldViewer:
    def __init__(self, image_dir="lightfield_views", radius=16, smoothing=False):
        self.image_dir = image_dir
        self.radius = radius
        self.smoothing = smoothing  # Option to enable/disable smoothing transitions
        self.current_image = None
        self.images = {}
        self.dragging = False
        self.current_position = None
        self.last_position = None
        self.last_displayed_position = None
        self.positions_array = None
        self.throttle_counter = 0  # For throttling updates
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
        
        for image_file in sorted(image_files):  # Sort for consistent loading
            # Extract offset values from filename (format: image_+Y_+X.tif)
            filename = os.path.basename(image_file)
            parts = filename.replace("image_", "").replace(".tif", "").split("_")
            
            try:
                y_offset = int(parts[0])
                x_offset = int(parts[1])
                
                # Load image
                image = tifffile.imread(image_file)
                
                # Apply slight blur for smoother transition if enabled
                if self.smoothing:
                    image = gaussian_filter(image, sigma=0.5)
                    
                self.images[(y_offset, x_offset)] = image
                print(f"Loaded image with offset y={y_offset}, x={x_offset}")
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing filename {filename}: {e}")
        
        # Create array of positions for nearest neighbor search
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
            self.ax_circle.plot(x, y, 'o', markersize=4)  # Smaller dots for clarity
        
        self.ax_circle.set_title('Click and drag to explore views')
        self.ax_circle.set_xlabel('X offset')
        self.ax_circle.set_ylabel('Y offset')
        self.ax_circle.grid(True)
        
        # Image display on the right
        self.ax_image = self.fig.add_subplot(122)
        self.ax_image.set_title('Light Field View')
        self.image_obj = None
        
        # Add mouse events with better throttling
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Show central image if available
        if (0, 0) in self.images:
            self.display_nearest_image(0, 0)
        elif len(self.images) > 0:
            # Display first available image
            first_key = list(self.images.keys())[0]
            self.display_nearest_image(first_key[0], first_key[1])
    
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
        """Handle mouse motion event with throttling"""
        if not self.dragging or event.inaxes != self.ax_circle:
            return
        
        # Throttle updates for smoother performance
        self.throttle_counter += 1
        if self.throttle_counter % 2 != 0:  # Process every other event
            return
            
        # Get current coordinates
        x, y = event.xdata, event.ydata
        
        # Check if within circle
        if x**2 + y**2 > self.radius**2:
            # Clamp to circle edge if outside
            angle = np.arctan2(y, x)
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
        
        # Only update if position has changed significantly to avoid jitter
        if self.last_position is None or \
           np.sqrt((y - self.last_position[0])**2 + (x - self.last_position[1])**2) > 0.3:  # Increased threshold
            self.last_position = (y, x)
            # Update display based on position
            self.update_from_position(y, x)
    
    def update_from_position(self, y, x):
        """Update display based on current mouse position"""
        # Just display the nearest image - no interpolation needed with many views
        self.display_nearest_image(y, x)
    
    def find_nearest_position(self, y, x):
        """Find the nearest available position to the given coordinates"""
        if not self.images:
            return None
            
        # Calculate distances to all positions
        distances = np.sqrt((self.positions_array[:, 0] - y)**2 + 
                           (self.positions_array[:, 1] - x)**2)
        
        # Find nearest index
        nearest_idx = np.argmin(distances)
        nearest_pos = tuple(self.positions_array[nearest_idx])
        
        return nearest_pos
    
    def display_nearest_image(self, y, x):
        """Display the nearest image to the given position"""
        if not self.images:
            return
            
        # Find the nearest position where we have an image
        nearest_pos = self.find_nearest_position(y, x)
        
        # Check if we're already displaying this position
        if self.last_displayed_position == nearest_pos:
            # Just update the marker in the circle
            self.update_circle_display(y, x, nearest_pos)
            return
            
        self.last_displayed_position = nearest_pos
        
        # Display the image
        self.display_image(nearest_pos, interactive_pos=(y, x))
    
    def update_circle_display(self, y, x, nearest_pos=None):
        """Update the circle display with current position and nearest image position"""
        self.ax_circle.clear()
        
        # Redraw circle
        circle = Circle((0, 0), self.radius, fill=False, color='blue')
        self.ax_circle.add_patch(circle)
        
        # Add dots for all positions
        for pos in self.images.keys():
            pos_y, pos_x = pos
            # Highlight the position of the current image
            if nearest_pos and pos == nearest_pos:
                self.ax_circle.plot(pos_x, pos_y, 'o', markersize=8, color='green')
            else:
                self.ax_circle.plot(pos_x, pos_y, 'o', markersize=4, color='blue')
        
        # Add current interactive position marker
        self.ax_circle.plot(x, y, 'x', markersize=10, color='red')
        
        # Draw line connecting interactive position to nearest image position
        if nearest_pos:
            self.ax_circle.plot([x, nearest_pos[1]], [y, nearest_pos[0]], '--', color='gray', alpha=0.5)
        
        # Add title and labels
        self.ax_circle.set_title('Click and drag to explore views')
        self.ax_circle.set_xlabel('X offset')
        self.ax_circle.set_ylabel('Y offset')
        self.ax_circle.set_xlim(-self.radius-1, self.radius+1)
        self.ax_circle.set_ylim(-self.radius-1, self.radius+1)
        self.ax_circle.grid(True)
    
    def display_image(self, position, interactive_pos=None):
        """Display the image at the given position"""
        if position not in self.images:
            print(f"No image available at position {position}")
            return
            
        # Clear previous image
        if self.image_obj is None:
            self.ax_image.clear()
            # Display new image
            image = self.images[position]
            self.image_obj = self.ax_image.imshow(image, cmap='gray')
        else:
            # Just update the data for smoother transitions
            self.image_obj.set_data(self.images[position])
        
        # Update circle display
        y, x = position
        if interactive_pos:
            self.update_circle_display(interactive_pos[0], interactive_pos[1], position)
        else:
            self.update_circle_display(y, x, position)
        
        # Update title
        if interactive_pos:
            int_y, int_x = interactive_pos
            self.ax_image.set_title(f'Light Field View (interactive: y={int_y:.1f}, x={int_x:.1f})\n(nearest view: y={y}, x={x})')
        else:
            self.ax_image.set_title(f'Light Field View (y={y}, x={x})')
        
        # Refresh display with minimal updates for better performance
        self.fig.canvas.draw_idle()

# Run the viewer
if __name__ == "__main__":
    viewer = LightFieldViewer(smoothing=True)  # Set to True for additional image smoothing
    plt.tight_layout()
    plt.show()

# Additional options that can be uncommented:
# - Enable smoothing for softer transitions: viewer = LightFieldViewer(smoothing=True)
# - Different directory: viewer = LightFieldViewer(image_dir="my_lightfield_data")
# - Different radius: viewer = LightFieldViewer(radius=20)