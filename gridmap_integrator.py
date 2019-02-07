import numpy as np
import dataset_loader
import viewer
from timeit import default_timer as timer

verbosity = True


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
        cells = np.reshape(cells, (2, -1)).T
        return cells

    def cell_coordinates(self, cell_inds):
        # x: front, y: left
        x = (self.origin_cell[0] - cell_inds[:, 0]) * self.CELLSIZE
        y = (self.origin_cell[1] - cell_inds[:, 1]) * self.CELLSIZE
        z = np.zeros(x.shape)
        w = np.ones(x.shape)
        points = np.stack([x, y, z, w], axis=1)
        return points

    def set_pose_offset(self, transform):
        self.prev_pose = transform

    def integrate(self, rgb_img, grid_maps, curr_pose):
        grid_maps["previous"] = self.transform_map(self.integ_map, curr_pose)
        grid_maps["current"] = self.merge_curr_maps(grid_maps)
        grid_maps["integrated"] = self.update_map(grid_maps["previous"], grid_maps["current"])
        self.integ_map = grid_maps["integrated"]
        self.prev_pose = curr_pose
        self.dviewer.show(rgb_img, grid_maps, 1200, 0)

    def transform_map(self, prev_integ_map, curr_pose):
        tmatrix = np.matmul(np.linalg.inv(self.prev_pose), curr_pose)

        # transform coordinates of current cells w.r.t the previous pose
        prev_coords = np.matmul(tmatrix, self.cell_coords.T).T
        prev_cell_inds = self.posit_to_cell(prev_coords)

        start = timer()
        curr_integ_map = self.interpolate(prev_integ_map, prev_cell_inds, self.cell_inds)
        if verbosity:
            print("transform matrix=\n{}".format(tmatrix))
            print(self.cell_inds.shape, self.cell_coords.shape)
            print("metric coordinates of the same cell [cur_x, cur_y, prv_x, prv_y]\n",
                  np.concatenate([self.cell_coords[0:200:40, :2], prev_coords[0:200:40, :2]], axis=1))
            print("grid coordinates of the same cell [cur_row, cur_col, prv_r, prv_c]\n",
                  np.concatenate([self.cell_inds[0:200:40, :], prev_cell_inds[0:200:40, :]], axis=1))
            print("interpolation time:", timer() - start)

        return curr_integ_map

    def posit_to_cell(self, points):
        grid_rows = -points[:, 0] / self.CELLSIZE + self.origin_cell[0]
        grid_cols = -points[:, 1] / self.CELLSIZE + self.origin_cell[1]
        return np.stack([grid_rows, grid_cols], axis=1)

    def interpolate(self, gridmap, src_cells, dst_cells):
        # round up floating src cell coordinates
        int_cells = np.round(src_cells).astype(np.int32)
        # find valid cell indices within grid map
        valid_inds = np.all((1 <= int_cells[:, 0], int_cells[:, 0] < self.GMROWS-1,
                             1 <= int_cells[:, 1], int_cells[:, 1] < self.GMCOLS-1), axis=0)
        if verbosity:
            print("valid inds\n{}\n{}".format(int_cells[3000:3200:20].T, valid_inds[3000:3200:20].T))
        # extract valid src cells and integer cells
        valid_src_cells = src_cells[valid_inds]
        valid_int_cells = int_cells[valid_inds]

        # relative positions of neighbor cells
        relposits = np.array([[-1, -1], [-1, 0], [-1, 1],
                              [0, -1], [0, 0], [0, 1],
                              [1, -1], [1, 0], [1, 1]])
        # calculate 9 neighbor cell coordinates and their weights for each valid cell
        neighbor_values = []
        weights = []
        for relpos in relposits:
            # calculate neighbor cells of all valid cells by relative position 'relpos'
            neighbor_cells = valid_int_cells + np.tile(relpos, (len(valid_int_cells), 1))
            # calculate neighbor weights that is close to 1 when it is close to src cell
            nei_weight = 1 - np.abs(neighbor_cells - valid_src_cells)
            # print("valid_src_cells\n{}\nneighbor cells\n{}\nweights of u, v\n{}".
            #       format(valid_src_cells[100:300:20].T, neighbor_cells[100:300:20].T,
            #              nei_weight[100:300:20].T))
            # ignore weight < 0
            nei_weight[nei_weight < 0] = 0
            # multiply row weight and column weight
            nei_weight = nei_weight[:, 0] * nei_weight[:, 1]
            weights.append(nei_weight)
            # print("weights", nei_weight[:10])

            # extract grid map values of current neighbors
            neighbor_inds = tuple([tuple(cell) for cell in neighbor_cells.T])
            cell_values = gridmap[neighbor_inds]
            # print("cell values\n", cell_values.shape, cell_values[:10])
            neighbor_values.append(cell_values)

        # nomalize weights
        weights = np.stack(weights, axis=1)
        weight_sum = np.sum(weights, axis=1)
        weights = weights / np.tile(weight_sum, (len(relposits), 1)).T
        # interpolated value is weight sum of neighbor cell values
        neighbor_values = np.stack(neighbor_values, axis=1)
        dst_values = np.sum(neighbor_values * weights, axis=1)
        if verbosity:
            print("neighbor_values\n{}\nweights\n{}\ndst_values\n{}".format(
                neighbor_values[3000:3200:20], weights[3000:3200:20], dst_values[3000:3200:20]))

        interp_map = np.zeros(gridmap.shape)
        # extract dst cells and set dst_values onto dst cells
        valid_dst_cells = dst_cells[valid_inds]
        dst_inds = tuple([tuple(cell) for cell in valid_dst_cells.T])
        interp_map[dst_inds] = dst_values
        return interp_map

    def outside_gridmap(self, cell):
        return (cell[0] < 0 or cell[0] >= self.GMROWS
                or cell[1] < 0 or cell[1] >= self.GMCOLS)

    @staticmethod
    def merge_curr_maps(grid_maps):
        # TODO: invert grid_maps["cal_yol"]
        # TODO: expand nonzero regions and append it as ["cam_yol_map"]
        grid_maps["cal_yol"] = 1 - grid_maps["cal_yol"]
        return np.mean(list(grid_maps.values()), axis=0)

    def update_map(self, before_map, new_map):
        # TODO: update log likelihood
        return np.mean(np.array([before_map, new_map]), axis=0)


def main():
    np.set_printoptions(precision=4, suppress=True)
    grid_mapper = GridMapIntegrator()
    kitti_loader = dataset_loader.KittiDataLoader("2011_09_26", "0015")
    transform = kitti_loader.get_pose(0)
    grid_mapper.set_pose_offset(transform)
    for ind in range(len(kitti_loader)):
        images = kitti_loader.get_rgb(ind)
        gmaps = kitti_loader.get_gmaps(ind)
        transform = kitti_loader.get_pose(ind)
        if verbosity:
            print("\n\nindex={}".format(ind))

        grid_mapper.integrate(images["rgbL"], gmaps, transform)


if __name__ == "__main__":
    main()
