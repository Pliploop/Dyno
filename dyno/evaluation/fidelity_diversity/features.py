
import numpy as np
import torch
import torch.nn.functional as F


def clip(velocities, angles):
    return np.clip(velocities, 0, 1), np.clip(angles, -1, 1)

def bin_(velocities, angles, num_bins=64):
    # bin the velocities and angles into num_bins bins between 0 and 1 and -1 and 1 respectively
    
    velocities = np.digitize(velocities, np.linspace(0, 1+1e-6, num_bins))
    angles = np.digitize(angles, np.linspace(-1, 1+1e-6, num_bins))
    
    
    velocities = np.linspace(0, 1, num_bins)[velocities]    
    angles = np.linspace(-1, 1, num_bins)[angles]
    
    
    return velocities, angles

def histogram_2d(velocities, angles, num_bins=64):
    
    angle_bins = np.linspace(-1, 1, num_bins +1)
    velocity_bins = np.linspace(0, 1, num_bins +1)
    
    
    
    hist, _, _ = np.histogram2d(velocities, angles, bins=[velocity_bins, angle_bins])
    hist = hist / np.sum(hist)
    
    return hist

def postprocess(norm, cosine_sim, num_bins=32):
    
    norm, cosine_sim = clip(norm, cosine_sim)
    norm, cosine_sim = bin_(norm, cosine_sim, num_bins=num_bins)
    
    # print unique values
    features = histogram_2d(norm, cosine_sim, num_bins=num_bins)
    # assert sum of features is 1
    assert np.isclose(np.sum(features), 1), f"Sum of features is {np.sum(features)}"
    
    #sum to bin into angle bins
    features = features.sum(axis=0).flatten()
    
    # sum along the cosine similarity axis to get bins for each cosine similarity
    
    
    return features

def get_fixed_length_motion_features(data, fixed_length=64):
        
    
    velocity_norms = np.linalg.norm(np.diff(data, axis=0), axis=1)
    velocity_cosine_sim = [cosine_similarity(data[i], data[i+1]) for i in range(data.shape[0]-1)]
    
    acceleration_norm = np.linalg.norm(np.diff(np.diff(data, axis=0), axis=0), axis=1)
    velocity = np.diff(data, axis=0)
    acceleration_cosine_sim = [cosine_similarity(velocity[i], velocity[i+1]) for i in range(velocity.shape[0]-1)]
    
    #pad or truncate the velocity and acceleration with zeros if necessary
    new_velocity_norms = np.zeros((fixed_length,))
    new_velocity_cosine_sim = np.zeros((fixed_length,))
    new_acceleration_norm = np.zeros((fixed_length,))
    new_acceleration_cosine_sim = np.zeros((fixed_length,))
    
    if velocity_norms.shape[0] < fixed_length:
        new_velocity_norms[:velocity_norms.shape[0]] = velocity_norms
        new_velocity_cosine_sim[:len(velocity_cosine_sim)] = velocity_cosine_sim
    elif velocity_norms.shape[0] > fixed_length:
        new_velocity_norms = velocity_norms[:fixed_length]
        new_velocity_cosine_sim = velocity_cosine_sim[:fixed_length]
        
    if acceleration_norm.shape[0] < fixed_length:
        new_acceleration_norm[:acceleration_norm.shape[0]] = acceleration_norm
        new_acceleration_cosine_sim[:len(acceleration_cosine_sim)] = acceleration_cosine_sim
    elif acceleration_norm.shape[0] > fixed_length:
        new_acceleration_norm = acceleration_norm[:fixed_length]
        new_acceleration_cosine_sim = acceleration_cosine_sim[:fixed_length]
        
        
    # clamp the velocity and acceleration values to 1
    velocity = postprocess(new_velocity_norms, new_velocity_cosine_sim)
    acceleration = postprocess(new_acceleration_norm, new_acceleration_cosine_sim)
    
    motion_features = np.concatenate([velocity, acceleration])
    motion_features = motion_features.reshape(-1)
    
    
    return motion_features

def interpolate_to_fixed_length_torch(vector, fixed_length=64):
    """
    Interpolate a tensor from shape (T, d) to shape (fixed_length, d) in a vectorized way.

    Args:
        vector (torch.Tensor): Input tensor of shape (T, d).
        fixed_length (int): Desired number of time steps (fixed_length).
    
    Returns:
        torch.Tensor: Interpolated tensor of shape (fixed_length, d).
    """
    if isinstance(vector, np.ndarray):
        was_np = True
        vector = torch.from_numpy(vector)
    else:
        was_np = False
    T, d = vector.shape
    vector = vector.unsqueeze(0).permute(0, 2, 1).float()  # Reshape to (N=1, C=d, L=T)
    interpolated = F.interpolate(vector, size=fixed_length, mode='linear', align_corners=True)
    interpolated = interpolated.squeeze(0).permute(1, 0)  # Back to (fixed_length, d)
    interpolated = interpolated.numpy() if was_np else interpolated
    return interpolated

def get_variable_length_motion_features(data, fixed_length = 64):
    data = interpolate_to_fixed_length_torch(data, fixed_length)
    velocity = np.diff(data, axis=0)
    acceleration = np.diff(velocity, axis=0)
    
    
    
    velocity_cosine_sim = [cosine_similarity(data[i], data[i+1]) for i in range(data.shape[0]-1)]
    velocity_cosine_sim = np.array(velocity_cosine_sim)

    acceleration_cosine_sim = [cosine_similarity(velocity[i], velocity[i+1]) for i in range(velocity.shape[0]-1)]
    acceleration_cosine_sim = np.array(acceleration_cosine_sim)
    
    velocity_norms = np.linalg.norm(velocity, axis=1)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    
    
    # simple 1d interpolation
    velocity_norms = np.interp(np.linspace(0, 1, fixed_length), np.linspace(0, 1, velocity_norms.shape[0]), velocity_norms)
    velocity_cosine_sim = np.interp(np.linspace(0, 1, fixed_length), np.linspace(0, 1, velocity_cosine_sim.shape[0]), velocity_cosine_sim)
    acceleration_norm = np.interp(np.linspace(0, 1, fixed_length), np.linspace(0, 1, acceleration_norm.shape[0]), acceleration_norm)
    acceleration_cosine_sim = np.interp(np.linspace(0, 1, fixed_length), np.linspace(0, 1, acceleration_cosine_sim.shape[0]), acceleration_cosine_sim)
    
    
    
    
    velocity = postprocess(velocity_norms, velocity_cosine_sim)
    acceleration = postprocess(acceleration_norm, acceleration_cosine_sim)
    
    motion_features = np.concatenate([velocity, acceleration])
    
    return motion_features


def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    