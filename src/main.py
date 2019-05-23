import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from utils.types import Point
from methods.ransac import Ransac 

def main(inp_file='input.txt', out_file='output.txt'):
  with open(inp_file, 'r') as fp:
    p = float(fp.readline())
    n = int(fp.readline())
    point_cloud = []
    for _ in range(n):
      point = Point(*map(float, fp.readline().split('\t')))
      point_cloud.append(point)

    print(point_cloud)

    segmentor = Ransac(verbose=True)
    start = time.perf_counter()
    plane, inliers = segmentor.run(data=point_cloud, threshold=p)
    print('Elapsed time: {:.3f}s'.format(time.perf_counter() - start))

    print('Found plane: {}'.format(plane))

    # Visualize results
    def plot_plane(a, b, c, d, xmin=-10, xmax=10, ymin=-10, ymax=10):
      xx, yy = np.mgrid[xmin:xmax, ymin:ymax]
      return xx, yy, (-a*xx - b*yy - d)/c

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    xyzs = np.array(inliers)
    ax.scatter3D(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2], c='g')
    outliers = [p for p in point_cloud if p not in set(inliers)]
    xyzs = np.array(outliers)
    ax.scatter3D(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2], c='r')
    xyzs = np.array(point_cloud)
    xx, yy, zz = plot_plane(*plane, xyzs[:, 0].min(), xyzs[:, 0].max(), xyzs[:, 1].min(), xyzs[:, 1].max())
    ax.plot_surface(xx, yy, zz)
    plt.show()

  with open(out_file, 'w') as fp:
    fp.write(' '.join(map(str, plane)))

if __name__ == '__main__':
  main(inp_file='../data/sdc_point_cloud.txt', out_file='../out/output.txt')  