import numpy as np
import dataset_loader
import viewer
from timeit import default_timer as timer

verbosity = False


class GridMapIntegrator:
    def __init__(self):
        self.prev_pose = np.identity(4)
        self.GMCOLS = int(80)
        self.GMROWS = int(120)
        self.CELLSIZE = 0.5
        self.integ_map = np.zeros((self.GMROWS, self.GMCOLS), dtype=np.float)
        self.origin_cell = [self.GMROWS, self.GMCOLS/2]
        self.cell_inds = self.cell_indices()
        self.cell_coords = self.cell_coordinates(self.cell_inds)
        self.dviewer = viewer.DataViewer()

    def cell_indices(self):
        cells = np.indices((self.GMROWS, self.GMCOLS))
        cells = np.reshape(cells, (2, -1))
        return cells

    def cell_coordinates(self, cell_inds):
        # x: front, y: left
        x = (self.origin_cell[0] - cell_inds[0, :]) * self.CELLSIZE
        y = (self.origin_cell[1] - cell_inds[1, :]) * self.CELLSIZE
        z = np.zeros(x.shape)
        w = np.ones(x.shape)
        points = np.stack([x, y, z, w], axis=0)
        return points

    def set_pose_offset(self, Tmat):
        self.prev_pose = Tmat

    def integrate(self, rgb_img, grid_maps, curr_pose, verbosity=False):
        grid_maps["prev"] = self.transform_map(self.integ_map, curr_pose)
        grid_maps["merged"] = self.merge_maps(grid_maps)
        grid_maps["integ"] = self.update_map(grid_maps["prev"], grid_maps["merged"])
        self.integ_map = grid_maps["integ"]
        self.prev_pose = curr_pose
        self.dviewer.show(rgb_img, grid_maps, 1200, 0)

    def transform_map(self, prev_integ_map, curr_pose):
        tmatrix = np.matmul(np.linalg.inv(self.prev_pose), curr_pose)

        # transform coordinates of current cells w.r.t the previous pose
        prev_coords = np.matmul(tmatrix, self.cell_coords)
        prev_cell_inds = self.posit_to_cell(prev_coords)
        curr_integ_map = np.zeros((self.GMROWS, self.GMCOLS))

        if verbosity:
            print("transform matrix=\n{}".format(tmatrix))
            print(self.cell_inds.shape, self.cell_coords.shape)
            print("metric coordinates of the same cell [cur_x, cur_y, prv_x, prv_y]\n",
                  np.concatenate([self.cell_coords[:2, 0:200:40].T, prev_coords[:2, 0:200:40].T], axis=1))
            print("grid coordinates of the same cell [cur_row, cur_col, prv_r, prv_c]\n",
                  np.concatenate([self.cell_inds[:2, 0:200:40].T, prev_cell_inds[:2, 0:200:40].T], axis=1))

        # transform grid map onto the current pose by interpolation
        start = timer()
        for prvcell, curcell in zip(prev_cell_inds.T, self.cell_inds.T):
            curr_integ_map[curcell[0], curcell[1]] = self.interpolate(prev_integ_map, prvcell)
        print("interpolation time:", timer() - start)

        return curr_integ_map

    def posit_to_cell(self, points):
        grid_rows = -points[0, :] / self.CELLSIZE + self.origin_cell[0]
        grid_cols = -points[1, :] / self.CELLSIZE + self.origin_cell[1]
        return np.array([grid_rows, grid_cols])

    def interpolate(self, gmap, cell):
        major_cell = np.round(cell).astype(np.int32)
        if self.outside_gridmap(major_cell):
            return 0

        relposits = np.array([[-1, -1], [-1, 0], [-1, 1],
                              [0, -1], [0, 0], [0, 1],
                              [1, -1], [1, 0], [1, 1]])
        weights = []
        value = 0
        for rel_pos in relposits:
            chk_cell = major_cell + rel_pos
            pos_err = np.abs(chk_cell - cell)
            if self.outside_gridmap(chk_cell) or np.any(pos_err > 1):
                # print("invalid neighbor", rel_pos, cell, chk_cell, pos_err)
                weights.append(False)
                continue

            weight = (1 - pos_err[0])*(1 - pos_err[1])
            weights.append(weight)
            value += gmap[chk_cell[0], chk_cell[1]]

        # skip if all neighbor cells are blank
        if value == 0:
            return 0

        weights = np.array(weights)
        wsum = np.sum(weights)
        weights = weights / wsum

        value = 0
        for rel_pos, weight in zip(relposits, weights):
            if weight:
                abs_pos = major_cell + rel_pos
                value += gmap[abs_pos[0], abs_pos[1]] * weight
        return value

    def outside_gridmap(self, cell):
        return (cell[0] < 0 or cell[0] >= self.GMROWS
                or cell[1] < 0 or cell[1] >= self.GMCOLS)

    def merge_maps(self, grid_maps):
        # TODO: weighted mean
        return np.mean(list(grid_maps.values()), axis=0)

    def update_map(self, before_map, new_map):
        # TODO: update log likelihood
        return np.mean(np.array([before_map, new_map]), axis=0)


def main():
    verbosity = True
    grid_mapper = GridMapIntegrator()
    kitti_loader = dataset_loader.KittiDataLoader("2011_09_26", "0015")
    Tmat = kitti_loader.get_pose(0)
    grid_mapper.set_pose_offset(Tmat)
    for ind in range(len(kitti_loader)):
        images = kitti_loader.get_rgb(ind)
        gmaps = kitti_loader.get_gmaps(ind)
        Tmat = kitti_loader.get_pose(ind)
        if verbosity:
            print("\nindex={}, Tmat:\n{}".format(ind, Tmat))
        grid_mapper.integrate(images["rgbL"], gmaps, Tmat)


if __name__ == "__main__":
    main()
