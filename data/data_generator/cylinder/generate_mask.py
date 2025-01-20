import numpy as np
import random
from scipy.ndimage import label
import cv2

def generate_mask(nx, ny, size, shape_counts):
    """
    Generate a 2D binary mask array with specified numbers of each shape placed randomly without overlapping.

    Parameters:
        nx (int): Number of rows in the mask array.
        ny (int): Number of columns in the mask array.
        size (int): Fixed size of each shape.
        shape_counts (dict): Dictionary specifying the number of each shape to place, e.g., {'circle': 5, 'square': 3}. Shape types can be chosen: 'circle', 'triangle', 'square', 'pentagon'

    Returns:
        np.ndarray: The generated binary mask array.

    Raises:
        RuntimeError: If unable to place a shape without overlapping after maximum attempts.
    """
    

    def create_shape(shape, size, angle=0):
        """
        Create a binary image of a specified shape with optional rotation.
        
        Parameters:
            shape (str): The type of shape ("circle", "triangle", "square", "pentagon").
            size (int): The size of the shape (e.g., diameter for circle, side length for square).
            angle (float, optional): The rotation angle in degrees. Defaults to 0.
        
        Returns:
            numpy.ndarray: A binary image of shape (size, size) with the specified shape drawn on it.
        """
        # Create a larger canvas to avoid clipping during rotation
        canvas_size = int(size * 2)  # Ensure the canvas is large enough
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        center = canvas_size // 2

        if shape == "circle":
            # Circles are rotation-invariant, so no need to rotate
            cv2.circle(canvas, (center, center), size // 2, 1, -1)
            # Define a bounding box for the circle (its circumscribed square)
            half_size = size // 2
            points = np.array([
                [center - half_size, center - half_size],
                [center - half_size, center + half_size],
                [center + half_size, center + half_size],
                [center + half_size, center - half_size]
            ], dtype=np.float32)  # Use float32 for rotation
        else:
            # Define the shape's points
            if shape == "triangle":
                points = np.array([
                    [center, center - size // 2],
                    [center - size // 2, center + size // 2],
                    [center + size // 2, center + size // 2]
                ], dtype=np.float32)  # Use float32 for rotation
            elif shape == "square":
                half_size = size // 2
                points = np.array([
                    [center - half_size, center - half_size],
                    [center - half_size, center + half_size],
                    [center + half_size, center + half_size],
                    [center + half_size, center - half_size]
                ], dtype=np.float32)  # Use float32 for rotation
            elif shape == "pentagon":
                angle_offset = -90  # Start from the top
                points = []
                for i in range(5):
                    x = center + (size // 2) * np.cos(np.radians(i * 72 + angle_offset))
                    y = center + (size // 2) * np.sin(np.radians(i * 72 + angle_offset))
                    points.append((x, y))
                points = np.array(points, dtype=np.float32)  # Use float32 for rotation
            else:
                raise ValueError(f"Unknown shape type: {shape}")

            # Rotate the points if angle is not 0
            if angle != 0:
                rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1)
                points = cv2.transform(np.array([points]), rotation_matrix).reshape(-1, 2)

            # Draw the shape
            cv2.fillPoly(canvas, [np.int32(points)], 1)

        # Crop the canvas to ensure the rotated image is not clipped
        def get_rotated_bounds(points, center, angle):
            """
            Calculate the bounding box of the rotated points.
            """
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_points = cv2.transform(np.array([points]), rotation_matrix).reshape(-1, 2)
            min_x, min_y = np.min(rotated_points, axis=0)
            max_x, max_y = np.max(rotated_points, axis=0)
            return min_x, min_y, max_x, max_y

        if angle != 0:
            # Calculate the bounding box of the rotated shape
            min_x, min_y, max_x, max_y = get_rotated_bounds(points, (center, center), angle)

            # Calculate the crop region
            start_x = max(0, int(min_x))
            start_y = max(0, int(min_y))
            end_x = min(canvas_size, int(max_x) + 1)
            end_y = min(canvas_size, int(max_y) + 1)

            # Crop the canvas
            canvas = canvas[start_y:end_y, start_x:end_x]

            # Resize the cropped canvas to the original size
            canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST)

        else:
            # If no rotation, simply crop the center
            start = (canvas_size - size) // 2
            end = start + size
            canvas = canvas[start:end, start:end]

        return canvas
    
    # Initialize mask array
    mask = np.zeros((nx, ny), dtype=int)
    
    max_attempts = 1000  # Maximum attempts to place each shape
    
    # Place each shape the specified number of times
    for shape, count in shape_counts.items():
        for _ in range(count):
            shape_img = create_shape(shape, size, random.uniform(0, 360))
            block_height, block_width = shape_img.shape
            attempts = 0
            placed = False
            while attempts < max_attempts and not placed:
                start_x = random.randint(int(0.1 * nx), int(0.9 * nx - block_height))
                start_y = random.randint(int(0.1 * ny), int(0.9 * ny - block_width))
                if np.sum(mask[start_x:start_x+block_height, start_y:start_y+block_width] & shape_img) == 0:
                    mask[start_x:start_x+block_height, start_y:start_y+block_width] |= shape_img
                    placed = True
                attempts += 1
            if not placed:
                raise RuntimeError(f"Unable to place {shape} without overlapping after {max_attempts} attempts.")
    
    return mask

if __name__ == '__main__':
    random.seed(416)
    nx = 2001
    ny = 1001
    size = 401
    shape_counts = { 'square':1}
    mask = generate_mask(nx,ny,size,shape_counts)
    
    import matplotlib.pyplot as plt
    plt.imshow(mask.T, cmap="gray")
    plt.title("Generated Mask with Random Shapes")
    plt.show()
    
