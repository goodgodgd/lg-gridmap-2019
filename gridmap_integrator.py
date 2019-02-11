import numpy as np
import dataset_loader
import viewer
from timeit import default_timer as timer
import copy

VERBOSITY = True
VEHICLE_ROW_RANGE = (-2, 4)     # x(front): -1m ~ 2m
VEHICLE_COL_RANGE = (-2, 2)     # y(left):  -1m ~ 1m
CELL_VALUE_LIMIT = 5


class GridMapIntegrator:
    def __init__(self):
        self.prev_pose = np.identity(4)
        self.GMCOLS = int(80)
        self.GMROWS = int(120)
        self.CELLSIZE = 0.5
        self.cur_logodd_stat = np.zeros((self.GMROWS, self.GMCOLS), dtype=np.float)
        self.cur_logodd_dyna = np.zeros((self.GMROWS, self.GMCOLS), dtype=np.float)
        self.cur_logodd_merge = np.zeros((self.GMROWS, self.GMCOLS), dtype=np.float)
        self.cur_ogm = np.ones((self.GMROWS, self.GMCOLS), dtype=np.float) * 0.5
        self.prv_logodd_stat = np.zeros((self.GMROWS, self.GMCOLS), dtype=np.float)
        self.prv_logodd_dyna = np.zeros((self.GMROWS, self.GMCOLS), dtype=np.float)
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
        # fill grid_maps with logodd maps before and after transformation
        grid_maps = self.transform_map(curr_pose, grid_maps)
        # fill grid_maps with updated logodd and occupancy map
        grid_maps = self.update_map(grid_maps)
        self.prev_pose = curr_pose
        self.dviewer.show(rgb_img, grid_maps, 1200, 0)

    def transform_map(self, curr_pose, grid_maps):
        start = timer()
        # transform coordinates of current cells w.r.t the previous pose
        tmatrix = np.matmul(np.linalg.inv(self.prev_pose), curr_pose)
        prev_coords = np.matmul(tmatrix, self.cell_coords.T).T
        prev_cell_inds = self.posit_to_cell(prev_coords)

        self.prv_logodd_stat = self.interpolate(self.cur_logodd_stat, prev_cell_inds, self.cell_inds)
        # TODO: applay motion model for dynamic objects
        self.prv_logodd_dyna = self.interpolate(self.cur_logodd_dyna, prev_cell_inds, self.cell_inds)
        grid_maps["T_logodd_static"] = copy.deepcopy(self.prv_logodd_stat)
        grid_maps["T_logodd_dynamic"] = copy.deepcopy(self.prv_logodd_dyna)

        if VERBOSITY:
            print("transform_map took:", timer() - start)
            print("transform matrix=\n{}".format(tmatrix))
            print(self.cell_inds.shape, self.cell_coords.shape)
            print("metric coordinates of the same cell [cur_x, cur_y, prv_x, prv_y]\n",
                  np.concatenate([self.cell_coords[0:200:40, :2], prev_coords[0:200:40, :2]], axis=1))
            print("grid coordinates of the same cell [cur_row, cur_col, prv_r, prv_c]\n",
                  np.concatenate([self.cell_inds[0:200:40, :], prev_cell_inds[0:200:40, :]], axis=1))

        return grid_maps

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
        if VERBOSITY:
            print("valid inds\n{}\n{}".format(int_cells[3000:3200:20].T, valid_inds[3000:3200:20].T))
        # extract valid src cells and integer cells
        valid_src_cells = src_cells[valid_inds]
        valid_int_cells = int_cells[valid_inds]

        # calculate 9 neighbor cell coordinates and their weights for each valid cell
        neighbor_values, weights = \
            self.find_neighbors_n_weights(gridmap, valid_src_cells, valid_int_cells)
        # calculate weighted sum of neighbor values as dst cell values
        dst_values = self.weight_sum(neighbor_values, weights)

        interp_map = np.zeros(gridmap.shape)
        # extract dst cells and set dst_values onto dst cells
        valid_dst_cells = dst_cells[valid_inds]
        dst_inds = tuple([tuple(cell) for cell in valid_dst_cells.T])
        interp_map[dst_inds] = dst_values
        return interp_map

    @staticmethod
    def find_neighbors_n_weights(gridmap, src_cells, int_cells):
        # relative positions of neighbor cells
        relposits = np.array([[-1, -1], [-1, 0], [-1, 1],
                              [0, -1], [0, 0], [0, 1],
                              [1, -1], [1, 0], [1, 1]])
        neighbor_values = []
        weights = []
        for relpos in relposits:
            # calculate neighbor cells of all valid cells by relative position 'relpos'
            neighbor_cells = int_cells + np.tile(relpos, (len(int_cells), 1))
            # calculate neighbor weights that is close to 1 when it is close to src cell
            nei_weight = 1 - np.abs(neighbor_cells - src_cells)
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

        return neighbor_values, weights

    @staticmethod
    def weight_sum(values, weights):
        # nomalize weights
        weights = np.stack(weights, axis=1)
        weight_sum = np.sum(weights, axis=1)
        weights = weights / np.tile(weight_sum, (weights.shape[1], 1)).T
        # interpolated value is weight sum of neighbor cell values
        values = np.stack(values, axis=1)
        dst_values = np.sum(values * weights, axis=1)

        if VERBOSITY:
            print("values\n{}\nweights\n{}\ndst_values\n{}".format(
                values[3000:3200:20], weights[3000:3200:20], dst_values[3000:3200:20]))

        return dst_values

    def update_map(self, grid_maps, weights=None):
        if not weights:
            weights = {"cam_fcn": 1, "lid_seg": 1, "cam_yol_occ": 2}

        grid_maps["cam_yol_occ"] = self.detection_to_occupancy_log(grid_maps["cam_yol"])

        # update static map
        self.cur_logodd_stat = self.prv_logodd_stat
        static_keys = ["cam_fcn", "lid_seg"]
        for key in static_keys:
            self.cur_logodd_stat += grid_maps[key] * weights[key]
        self.cur_logodd_stat = np.clip(self.cur_logodd_stat, -CELL_VALUE_LIMIT, CELL_VALUE_LIMIT)

        # attenuate dynamic map
        self.cur_logodd_dyna = np.minimum(self.prv_logodd_dyna + 1, 0)
        print("dyna minumum", np.min(self.prv_logodd_dyna), np.min(self.cur_logodd_dyna))
        # update dynamic map with new objects
        dynamic_keys = ["cam_yol_occ"]
        for key in dynamic_keys:
            self.cur_logodd_dyna -= grid_maps[key] * weights[key]
        grid_maps["cur_logodd_dynamic"] = copy.deepcopy(self.cur_logodd_dyna)

        # merge static and dynamic maps
        self.cur_logodd_merge = np.where(self.cur_logodd_dyna < 0,
                                         self.cur_logodd_dyna, self.cur_logodd_stat)

        # self.cur_logodd_merge = self.cur_logodd_stat + self.cur_logodd_dyna
        # # overwrite current object measurement on merged amp
        # for key in dynamic_keys:
        #     self.cur_logodd_merge = np.where(grid_maps[key] > 0,
        #                                      -grid_maps[key] * weights[key],
        #                                      self.cur_logodd_merge)

        self.cur_ogm = self.logodd_to_prob(self.cur_logodd_merge)
        grid_maps["updated_logodd"] = self.cur_logodd_merge
        grid_maps["occupancy_map"] = copy.deepcopy(self.cur_ogm)
        return grid_maps

    def detection_to_occupancy_log(self, detection):
        row_inds, col_inds = np.where(detection > 0.0001)
        det_log = np.zeros(detection.shape, dtype=np.int32)
        for row, col in zip(row_inds, col_inds):
            rmin = max([row + VEHICLE_ROW_RANGE[0], 0])
            rmax = min([row + VEHICLE_ROW_RANGE[1], self.GMROWS-1])
            cmin = max([col + VEHICLE_COL_RANGE[0], 0])
            cmax = min([col + VEHICLE_COL_RANGE[1], self.GMCOLS - 1])
            det_log[rmin:rmax, cmin:cmax] = 1
        return det_log

    @staticmethod
    def logodd_to_prob(logodd):
        prob = 1 - 1/(1 + np.exp(logodd))
        return prob


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
        if VERBOSITY:
            print("\n\nindex={}".format(ind))

        grid_mapper.integrate(images["rgbL"], gmaps, transform)


if __name__ == "__main__":
    main()
