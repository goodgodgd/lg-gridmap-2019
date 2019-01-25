import numpy as np
import dataset_loader


class GridMapIntegrator:
    def __init__(self):
        self.integ_map = None
        self.prev_pose = None

    def integrate(self, grid_maps, curr_pose):
        self.integ_map = self.transform_map(self.integ_map, curr_pose)
        self.integ_map = self.update_map(self.integ_map, grid_maps)
        self.prev_pose = curr_pose

    def transform_map(self, integ_map, curr_pose):
        return integ_map

    def update_map(self, integ_map, grid_maps):
        return integ_map


def main():
    grid_mapper = GridMapIntegrator()
    kitti_loader = dataset_loader.KittiDataLoader("2011_09_26", 15)
    for ind in range(len(kitti_loader)):
        gmaps = kitti_loader.get_gmaps(ind)
        Tmat = kitti_loader.get_pose(ind)
        grid_mapper.integ_map(gmaps, Tmat)


if __name__ == "__main__":
    main()
