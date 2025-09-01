import numpy as np
#import torch.distributions as dist
from scipy.spatial import cKDTree
from scipy.spatial.distance import mahalanobis
import scipy.special as sps
from scipy.stats import multivariate_normal

def get_cluster_indices_cov(points, cov_matrices, dist=10):    
    # Initialize clusters and visited points
    clusters = []
    visited = set()
    
    # Create clusters
    for point_index in range(len(points)):
        if point_index not in visited:
            cluster = find_points_within_threshold_cov(points, point_index, cov_matrices, visited, distance_threshold=dist)
            if len(cluster) > 1:
                visited.update(cluster)
                clusters.append(cluster)
                
    for point_index in range(len(points)):
        if point_index not in visited:
            visited.update([point_index])
            clusters.append([point_index])

    # Convert clusters to a list of indices
    cluster_indices = [list(set(cluster)) for cluster in clusters if cluster]  # Remove empty clusters
    return cluster_indices

def find_points_within_threshold(points,point_index,visited,distance_threshold = None,kdtree = None):
    neighbors = kdtree.query_ball_point(points[point_index], distance_threshold)
    return [neighbor for neighbor in neighbors if neighbor not in visited]
    
def get_cluster_indices(input_points, scale = np.array([1]), dist = 10):        
    # Build a KD-Tree for efficient nearest neighbor search
    points = input_points*scale[np.newaxis,:]
    kdtree = cKDTree(points)    
    # Initialize clusters and visited points
    clusters = []
    visited = set()
    # Iterate through the points to create clusters
    for point_index in range(len(points)):
        
        if point_index not in visited:
            cluster = find_points_within_threshold(points,point_index,visited,distance_threshold = dist, kdtree = kdtree)
            visited.update(cluster)
            clusters.append(cluster)    
    # Convert clusters to a list of indices
    cluster_indices = [list(cluster) for cluster in clusters]
    return cluster_indices     


def filter_try_points(fit_points,try_points,threshold = 0.1, metric = None):
    try_fit_distances = np.sqrt(np.sum((try_points[:, None] - fit_points) @ metric * (try_points[:, None] - fit_points), dim=2))    
    #try_fit_distances = np.sqrt((try_points[:, None] - fit_points) @ metric @ (try_points[:, None] - fit_points).transpose(1, 2)).squeeze(-1)
    exclude_mask = (try_fit_distances.min(dim=1).values >= threshold)
    filtered_try_points = try_points[exclude_mask]
    return filtered_try_points


def filter_repeat_points(matching_indices,searched_index):
    matching_set = set(matching_indices.tolist())
    searched_set = set(searched_index.tolist())
    filtered_indices = list(matching_set - searched_set) 
    return filtered_indices


def merge_arrays(original_lists, cluster_indices, element_num_list, log_like_list, latest_prob_index, cov_update_count):
    new_log_like_list = []
    exchange_num = min(latest_prob_index, element_num_list[0])
    ind1 = element_num_list[0] - exchange_num
    ind2 = element_num_list[0]
    ind1_pro = int(ind1/cov_update_count)
    ind2_pro = int(ind2/cov_update_count)
    # Initialize an empty list to store merged results for each original_list in original_lists
    all_merged_lists = [[] for _ in original_lists]  # List of lists to hold results for each original_list

    for cluster in cluster_indices:
        for idx, index in enumerate(cluster):
            if idx == 0:
                # Process the first item in each cluster for each original_list
                new_arrs_list = [original_list[index] for original_list in original_lists]
                new_log_like = log_like_list[index]
            #else:
            #    #select sample indices
            #    mask_samples = new_log_like[ind1:ind2] < log_like_list[index][ind1:ind2]                
            #    ratio = mask_samples.sum()/(ind2 - ind1)
            #    #select proposal indices
            #    indices = np.arange(0, ind2_pro - ind1_pro)
            #    np.random.shuffle(indices)
            #    mask_proposals = np.zeros(len(indices), dtype=bool)
            #    mask_proposals[indices[:int(ratio * len(indices))]] = True
            #    #select proposal components
            #    mask_com = np.repeat(mask_proposals,batch_point_num)
            #    # Apply mask to each corresponding original_list in original_lists
            #    for i, original_list in enumerate(original_lists):
            #        new_arr = new_arrs_list[i]
            #        if i < 2 and mask_samples.sum() > 0:     # sample and denominators         
            #            mask_here = mask_samples[:, np.newaxis] if len(original_list[index].shape) > 1 else mask_samples
            #            new_arr[ind1:ind2] = np.where(mask_here, original_list[index][ind1:ind2], new_arr[ind1:ind2])
            #            new_arrs_list[i] = new_arr
            #        elif i < 4 and mask_samples.sum() > 0: #cov and cov normalization term
            #            mask_here = mask_proposals[:, np.newaxis, np.newaxis] if len(original_list[index].shape) > 1 else mask_proposals
            #            new_arr[ind1_pro:ind2_pro] = np.where(mask_here, original_list[index][ind1_pro:ind2_pro], new_arr[ind1_pro:ind2_pro])
            #            new_arrs_list[i] = new_arr
            #        elif mask_samples.sum() > 0:
            #            mask_here = mask_com[:, np.newaxis] if len(original_list[index].shape) > 1 else mask_com
            #            new_arr[ind1:ind2] = np.where(mask_here, original_list[index][ind1:ind2], new_arr[ind1:ind2])
            #            new_arrs_list[i] = new_arr                        
            #            
            #    new_log_like[ind1:ind2] = np.where(mask_samples, log_like_list[index][ind1:ind2], new_log_like[ind1:ind2])

        # Append merged results for each original_list to the corresponding merged lists
        for i, new_arr in enumerate(new_arrs_list):
            all_merged_lists[i].append(new_arr)
        new_log_like_list.append(new_log_like)

    return all_merged_lists, new_log_like_list





def merge_cov_arrays(original_list, cluster_indices, element_num_list, cov_update_count):
    # Initialize an empty list to hold the merged lists
    merged_lists = []    
    # Iterate through each cluster in cluster_indices
    for cluster in cluster_indices:
        for idx, index in enumerate(cluster):
            if idx == 0:
                new_arr = original_list[index]
                last_element_num = int(element_num_list[index]/cov_update_count)
            #else:
            #    now_element_num = int(element_num_list[index]/cov_update_count)
            #    #if original_list[index][now_element_num][0,0] == 0:
            #    #    print(now_element_num, )
            #    new_arr[last_element_num:last_element_num+now_element_num] = original_list[index][:now_element_num]
            #    last_element_num += now_element_num
        merged_lists.append(new_arr)
    return merged_lists

def merge_max_list(max_loglike_list, cluster_indices):
    new_max_loglike_list = []    
    for cluster in cluster_indices:
        if len(cluster) > 1:
            #max_value = None
            #for index in cluster:
            #    # Update max_value with the maximum from the current sub-list
            #    if max_value is None or max_loglike_list[index] > max_value:
            #        max_value = max_loglike_list[index]
            #new_max_loglike_list.append(max_value)
            new_max_loglike_list.append(max_loglike_list[cluster[0]])
        else:
            new_max_loglike_list.append(max_loglike_list[cluster[0]])    
    return new_max_loglike_list

def merge_element_num_list(element_num_list, cluster_indices):
    new_element_num_list = []
    for cluster in cluster_indices:
        if len(cluster) > 1:
            #tol_num = 0
            #for index in cluster:
            #    tol_num += element_num_list[index]
            new_element_num_list.append(element_num_list[cluster[0]])
        else:
            new_element_num_list.append(element_num_list[cluster[0]])
    return new_element_num_list


def mahalanobis_batch(points_neighbors, point, inv_cov):
    """Vectorized Mahalanobis distance calculation for multiple points."""
    diffs = points_neighbors - point  # Shape: (n_neighbors, n_dims)
    # Perform (p - q)^T @ inv_cov @ (p - q) for all neighbors in one go
    mahal_dist_sq = np.einsum('ij,jk,ik->i', diffs, inv_cov, diffs)
    return np.sqrt(mahal_dist_sq)

def find_points_within_threshold_cov(points, point_index, cov_matrices, visited, distance_threshold=None):
    # Find all potential neighbors using KD-tree as a first filter
    neighbors = [idx for idx in range(len(points)) if idx not in visited]
    
    if not neighbors:
        return []  # No neighbors found

    # Now calculate the Mahalanobis distances for all neighbors
    point_cov = cov_matrices[point_index]
    point = points[point_index]
    inv_cov = np.linalg.inv(point_cov)  # Inverse of the covariance matrix
    
    # Get the points corresponding to the neighbor indices
    points_neighbors = np.array([points[idx] for idx in neighbors])
    
    # Calculate the Mahalanobis distances for all neighbors in one step
    mahal_dists = mahalanobis_batch(points_neighbors, point, inv_cov)
    
    # Filter neighbors based on distance threshold and visited status
    filtered_neighbors = [neighbors[i] for i, dist in enumerate(mahal_dists) 
                          if dist <= distance_threshold]

    #print('1',points_neighbors, point, inv_cov)
    #print('2',mahal_dists)
    #print('3',filtered_neighbors)    

    return filtered_neighbors
