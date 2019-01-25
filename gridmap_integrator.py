import numpy as np
import kitti_data_loader

class GridMapIntegrator:
    def __init__(self):
        self.integ_map = None
        self.prev_pose = None

    def integrate(self, grid_maps, curr_pose):
        self.integ_map = self.transform_map(self.integ_map, curr_pose)
        gmap_sum = np.sum(grid_maps, axis=0)
        self.integ_map = self.update_map(self.integ_map, gmap_sum)
        self.prev_pose = curr_pose

    def transform_map(self, integ_map, curr_pose):
        return integ_map

    def update_map(self, integ_map, gmap_sum):
        return integ_map


def main():
    data_loader = kitti_data_loader.DataLoader("2011_09_26", 15)
    for ind in range(len(data_loader)):
        imgs, gmaps = data_loader.load_data(ind, asgray=True, read_gmap=True)


if __name__ == "__main__":
    main()
