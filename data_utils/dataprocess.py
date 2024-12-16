import os
import numpy as np
from tqdm import tqdm

def pc_normalize(pc):
    """Normalize the point cloud by centering and scaling."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Farthest point sampling.
    Args:
        point: Input point cloud, [N, D]
        npoint: Number of points to sample
    Returns:
        sampled points: [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return point[centroids.astype(np.int32)]

def save_to_ply(points, filepath):
    """
    Save point cloud to PLY format.
    Args:
        points: Point cloud data, [N, 3]
        filepath: Save path
    """
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

def preprocess_and_save(input_dir, output_dir, npoints, normalize=True):
    """
    Preprocess data and save as PLY format.
    Args:
        input_dir: Directory of raw CSV data
        output_dir: Directory to save processed point clouds
        npoints: Number of points to sample
        normalize: Whether to normalize the point cloud
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
    total_files = len(csv_files)
    assert total_files == 200, "Expected 200 CSV files, please check data integrity!"

    # Correct time step order
    correct_time_order = [20, 90, 130, 140, 60, 100, 30, 40, 70, 80, 0, 10, 120, 150, 50, 110]
    
    for file_idx, csv_file in enumerate(tqdm(csv_files, desc="Processing files")):
        filepath = os.path.join(input_dir, csv_file)
        person_dir = os.path.join(output_dir, f"person_{file_idx+1:03d}")
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        # Attempt to load data and catch errors if any
        try:
            data = np.loadtxt(filepath, delimiter=",", skiprows=1, usecols=(0, 2, 3, 4))  # Extract time and coordinates columns
        except IndexError as e:
            print(f"Error: {csv_file} has insufficient columns: {e}")
            continue

        # Remove NaN and Inf values
        valid_mask = np.all(np.isfinite(data), axis=1)
        data = data[valid_mask]
        
        for idx, t in enumerate(correct_time_order):
            time_data = data[data[:, 0] == t][:, 1:]  # Extract point cloud coordinates for the current time step
            if time_data.shape[0] == 0:
                print(f"Warning: {csv_file} has no data at time step {t}, skipping")
                continue

            # Randomly resample if points are insufficient
            if time_data.shape[0] < npoints:
                deficit = npoints - time_data.shape[0]
                sampled_indices = np.random.choice(time_data.shape[0], deficit, replace=True)
                time_data = np.vstack((time_data, time_data[sampled_indices]))

            # Normalize if required
            if normalize:
                time_data = pc_normalize(time_data)

            # Farthest point sampling
            time_data = farthest_point_sample(time_data, npoints)

            # Save as PLY file
            output_file = os.path.join(person_dir, f"time_{idx:03d}.ply")
            save_to_ply(time_data, output_file)
    
    print(f"Processing complete! All point clouds saved to directory: {output_dir}")

if __name__ == "__main__":
    # Configuration
    input_dir = "your path"  # Replace with your raw data path
    output_dir = "your path"  # Replace with your save path
    npoints = 1024  # Target number of points for farthest point sampling
    normalize = True  # Whether to normalize the point cloud

    preprocess_and_save(input_dir, output_dir, npoints, normalize)
