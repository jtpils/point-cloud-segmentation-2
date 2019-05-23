# RANSAC algorithms for finding best-fitting plane ignoring outliers
import random
from typing import Iterable, Sized, Collection, List, Callable, Union
from math import ceil
from utils.types import Point, Plane
from utils.geometry import estimate_plane_svd, is_inlier_3d

class Ransac(object):

  def __init__(self, estimate:Callable=None, is_inlier:Callable=None, verbose:bool=True):
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
          data:Collection,
          threshold:float, 
          sample_size:int=3,
          goal_inliers:Union[int, float]=0.5,
          max_iterations:int=1000,
          stop_at_goal=True,
          random_seed=None) -> Plane:
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

    best_plane:Plane = None
    best_inliers:List[Point] = []
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
      inliers:List[Point] = []
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


      


