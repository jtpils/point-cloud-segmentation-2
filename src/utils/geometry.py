from typing import Collection
import math
import numpy as np
from numpy.linalg import svd
from utils.types import Point, Plane

def augment(points:Collection[Point]):
    aug = np.ones((len(points), 4))
    aug[:, :3] = np.asarray(points)
    return aug

def estimate_plane_svd(points:Collection[Point]):
    points = augment(points)
    return Plane(*np.linalg.svd(points)[-1][-1, :])

def normal_norm(plane:Plane):
    return math.sqrt(plane.A**2 + plane.B**2 + plane.C**2)

def is_inlier_3d(plane:Plane, pt:Point, th_dist):
    d = np.abs(np.asarray(plane).dot(augment([pt]).T))/normal_norm(plane)
    return d <= th_dist