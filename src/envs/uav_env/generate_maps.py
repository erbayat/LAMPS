import numpy as np
import os

def generate_map_with_bisection_radius(rows, cols, target_mean=0.5, inside_prob=0.9, outside_prob=0.1, tol=1e-3, max_iter=100):
    """
    Generate a binary map with a random center and adjust the radius using bisection to meet the target mean.

    Args:
        rows (int): Number of rows in the map.
        cols (int): Number of columns in the map.
        target_mean (float): Desired mean probability of the map.
        inside_prob (float): Probability of an event inside the radius.
        outside_prob (float): Probability of an event outside the radius.
        tol (float): Tolerance for the difference between the current and target mean.
        max_iter (int): Maximum number of bisection iterations.

    Returns:
        np.ndarray: Generated binary map.
    """
    # Randomly select the center
    center_x = np.random.randint(0, rows)
    center_y = np.random.randint(0, cols)

    # Define the search range for the radius
    min_radius = 0
    max_radius = max(rows,cols)  # Diagonal of the map
    best_radius = None

    for _ in range(max_iter):
        # Calculate the midpoint radius
        radius = (min_radius + max_radius) / 2

        # Create grid
        x = np.arange(0, rows).reshape(-1, 1)
        y = np.arange(0, cols).reshape(1, -1)

        # Calculate distance from the center for each cell
        distance_map = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # Assign probabilities based on radius
        probability_map = np.where(distance_map <= radius, inside_prob, outside_prob)

        # Calculate the current mean
        current_mean = np.mean(probability_map)

        # Check if the current mean is close enough to the target
        if abs(current_mean - target_mean) < tol:
            best_radius = radius
            break

        # Adjust the search range
        if current_mean < target_mean:
            min_radius = radius  # Increase radius
        else:
            max_radius = radius  # Decrease radius

    # Finalize the radius if not set during the loop
    if best_radius is None:
        best_radius = radius

    # Generate the binary map with the final radius
    distance_map = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    probability_map = np.where(distance_map <= best_radius, inside_prob, outside_prob)
    binary_map = (np.random.rand(rows, cols) < probability_map).astype(int)

    return binary_map

def generate_uav_positions(rows, cols, num_uavs):
    """
    Generate UAV positions for a given map with unique (x, y) coordinates.

    Args:
        rows (int): Number of rows in the map.
        cols (int): Number of columns in the map.
        num_uavs (int): Number of UAVs to place.

    Returns:
        np.ndarray: Array of UAV positions.
    """
    positions = []
    for _ in range(num_uavs):
        while True:
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)
            # Ensure the position is not on an obstacle (map value = 0) and is unique
            if (x, y) not in positions:  # Free space and unique
                positions.append((x, y))
                break
    return np.array(positions)


if __name__ == "__main__":

    # Set dimensions and ratio
    num_uavs = 7

    rows, cols = 50,50
    ratio_of_ones = 0.5  # Example ratio
    # Generate 15 binary maps
    maps = {}
    uav_positions = {}
    for i in range(1, 105):
        maps[f"map_{i}"] = generate_map_with_bisection_radius(rows, cols,ratio_of_ones)#generate_binary_map_with_random_center_radius(rows, cols,ratio_of_ones)
        uav_positions[f"map_{i}"] = generate_uav_positions(rows, cols, num_uavs)

    # Directory to save the .npy files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(current_dir, str(rows)+'_'+str(cols))
    maps_directory = os.path.join(save_directory, 'maps')
    uavs_directory = os.path.join(save_directory, "uav_positions")

    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Create directories if they don't exist
    os.makedirs(maps_directory, exist_ok=True)
    os.makedirs(uavs_directory, exist_ok=True)

    # Save each map and UAV positions as .npy files
    for i in range(1, 105):
        map_file = os.path.join(maps_directory, f"map_{i}.npy")
        uav_file = os.path.join(uavs_directory, f"uav_positions_{i}.npy")
        np.save(map_file, maps[f"map_{i}"])
        np.save(uav_file, uav_positions[f"map_{i}"])
        print(f"Saved map_{i} and UAV positions_{i}.")

    print("All maps and UAV positions generated and saved.")
