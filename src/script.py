import math
import time
import random
from collections import namedtuple
from math import ceil
import numpy as np
from numpy.linalg import svd


Point = namedtuple('Point', ['x', 'y', 'z'])

Plane = namedtuple('Plane', ['A', 'B', 'C', 'D'])

def augment(points):
    aug = np.ones((len(points), 4))
    aug[:, :3] = np.asarray(points)
    return aug

def estimate_plane_svd(points):
    points = augment(points)
    return Plane(*np.linalg.svd(points)[-1][-1, :])

def normal_norm(plane):
    return math.sqrt(plane.A**2 + plane.B**2 + plane.C**2)

def is_inlier_3d(plane, pt, th_dist):
    d = np.abs(np.asarray(plane).dot(augment([pt]).T))/normal_norm(plane)
    return d <= th_dist


class Ransac(object):

  def __init__(self, estimate=None, is_inlier=None, verbose=True):
    """
    RANSAC algorithm

    Args:
      estimate: function to find plane coefficients
      is_inlier: function to check if point is inlier or not

    """
    self.verbose = verbose
    self.estimate = estimate if estimate is not None else estimate_plane_svd
    self.is_inlier = is_inlier if is_inlier is not None else is_inlier_3d
  
  def run(self,
          data,
          threshold, 
          sample_size=3,
          goal_inliers=0.5,
          max_iterations=1000,
          stop_at_goal=True,
          random_seed=None):
    """
    Run fitting procedure
  
    Args:
      data: points cloud
      
      sample_size: how many point to sample
      goal_inliers: minimum number of inliers required or ratio
      max_iterations: maximum number of iterations allowable
      stop_at_goal: stop iterations if goal of inliers count achieved
      random_seed: seed number to repeatable results
    
    Returns:
      best_plane: coefficients of the best found plane
      best_inliers: set of inlier points
    """

    best_plane = None
    best_inliers = []
    random.seed(random_seed)
    if isinstance(goal_inliers, float) and goal_inliers <= 1.0:
      goal_inliers = ceil(goal_inliers*len(data))
    else:
      goal_inliers = int(goal_inliers)

    data = list(data)
    for i in range(max_iterations):

      # Sample support points
      support = random.sample(data, int(sample_size))
      
      # Fit plane to support
      plane = self.estimate(support)

      # Found inliers
      inliers = []
      for j in range(len(data)):
        if self.is_inlier(plane, data[j], threshold):
          inliers.append(data[j])

      if self.verbose:
        print('Iter {}/{}. Best plane: {}. Inliers {}/{}'.format(
          i+1, max_iterations, plane, len(inliers), goal_inliers))

      # Check if goal achieved
      if len(inliers) > len(best_inliers):
        best_plane = plane
        best_inliers = inliers
        if (len(best_inliers) >= goal_inliers) and stop_at_goal:
          break
      
    if self.verbose:
      print('Process took: {} iterations. Inliers count: {}'.format(i+1, len(best_inliers)))

    return best_plane, best_inliers


def main(inp_file='input.txt', out_file='output.txt'):
  with open(inp_file, 'r') as fp:
    p = float(fp.readline())
    n = int(fp.readline())
    point_cloud = []
    for _ in range(n):
      point = Point(*map(float, fp.readline().split('\t')))
      point_cloud.append(point)

    # print(point_cloud)

    segmentor = Ransac(verbose=False)
    start = time.perf_counter()
    plane, inliers = segmentor.run(data=point_cloud, threshold=p)
    # print('Elapsed time: {:.3f}s'.format(time.perf_counter() - start))

    # print('Found plane: {}'.format(plane))

  with open(out_file, 'w') as fp:
    fp.write(' '.join(map(str, plane)))

main() 
  