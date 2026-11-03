"""Rigid-transform helpers shared by capture and live ROS perception."""
import numpy as np
from .preprocess import transform_points

def transform_matrix(transform):
    origin=transform_points(np.zeros((1,3),np.float32),transform)[0]
    basis=transform_points(np.eye(3,dtype=np.float32),transform)-origin
    matrix=np.eye(4,dtype=np.float32);matrix[:3,:3]=basis.T;matrix[:3,3]=origin
    return matrix.tolist()
