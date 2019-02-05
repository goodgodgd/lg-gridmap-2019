import numpy as np
import dataset_loader
import viewer


class GridMapIntegrator:
    def __init__(self):
        self.prev_pose = np.identity(4)
        self.GMCOLS = 80
        self.GMROWS = 120
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

    def integrate(self, index, rgb_img, grid_maps, curr_pose):
        if index < 2:
            self.integ_map = self.update_map(self.integ_map, grid_maps)
            print("update map")
        else:
            self.integ_map = self.transform_map(self.integ_map, curr_pose)
        self.prev_pose = curr_pose
        grid_maps["integ"] = self.integ_map
        print("integrate end\n", self.integ_map[0:120:20, 10:80:20])
        self.dviewer.show(rgb_img, grid_maps, 1000, 0)

    def transform_map(self, prev_integ_map, curr_pose):
        tmatrix = np.matmul(np.linalg.inv(self.prev_pose), curr_pose)
        # print("transform map, tmatrix=\n{}\nmap=\n{}".format(tmatrix, prev_integ_map[0:120:20, 10:80:20]))
        tmatrix = np.identity(4)
        tmatrix[0, 3] = 1
        # find coordinates of current cells w.r.t the previous pose
        prev_coords = np.matmul(tmatrix, self.cell_coords)
        prev_cell_inds = self.posit_to_cell(prev_coords)
        curr_integ_map = np.zeros((self.GMROWS, self.GMCOLS))
        print(self.cell_coords.shape, prev_coords.shape)
        print(self.cell_inds.shape, prev_cell_inds.shape)
        print("cell coords\n", np.concatenate([self.cell_coords[:2, 0:100:20].T, prev_coords[:2, 0:100:20].T], axis=1))
        print("cell coords\n", np.concatenate([self.cell_inds[:2, 0:100:20].T, prev_cell_inds[:2, 0:100:20].T], axis=1))

        for prvcell, curcell in zip(prev_cell_inds.T, self.cell_inds.T):
            curr_integ_map[curcell[0], curcell[1]] = self.interpolate(prev_integ_map, prvcell)
            # if prvcell[0] < 30 and prvcell[1] == 30:
            #     prvcell = prvcell.astype(np.int32)
            #     print("cell move", prvcell, prev_integ_map[prvcell[0], prvcell[1]],
            #           curcell, curr_integ_map[curcell[0], curcell[1]])

        return curr_integ_map

    def posit_to_cell(self, points):
        grid_rows = -points[0, :] / self.CELLSIZE + self.origin_cell[0]
        grid_cols = -points[1, :] / self.CELLSIZE + self.origin_cell[1]
        return np.array([grid_rows, grid_cols])

    def interpolate(self, gmap, cell):
        if cell[0] < -1 or cell[0] > self.GMROWS+1 \
                or cell[1] < -1 or cell[1] > self.GMCOLS+1:
            return 0

        gh, gw = gmap.shape
        major_cell = np.round(cell).astype(np.int32)
        if major_cell[0] < 0 or major_cell[0] >= gh or \
                major_cell[1] < 0 or major_cell[1] >= gw:
            return 0

        return gmap[major_cell[0], major_cell[1]]

        major_cell = np.floor(cell)
        row_err = cell[0] - major_cell[0] - 0.5
        col_err = cell[1] - major_cell[1] - 0.5
        relposs = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
        weights = np.array([(1-abs(row_err))*(1-abs(col_err)),
                   row_err*(1-abs(col_err)),
                   -row_err*(1-abs(col_err)),
                   (1-abs(row_err))*col_err,
                   -(1-abs(row_err))*col_err])
        wsum = np.sum(weights)
        weights = weights / wsum

        value = 0
        for relpos, weight in zip(relposs, weights):
            abspos = major_cell + relpos
            value += gmap[abspos] * weight
        return value

    def update_map(self, integ_map, grid_maps):
        return np.mean(list(grid_maps.values()), axis=0)


def main():
    grid_mapper = GridMapIntegrator()
    kitti_loader = dataset_loader.KittiDataLoader("2011_09_26", "0015")
    Tmat = kitti_loader.get_pose(0)
    grid_mapper.set_pose_offset(Tmat)
    for ind in range(len(kitti_loader)):
        images = kitti_loader.get_rgb(ind)
        gmaps = kitti_loader.get_gmaps(ind)
        Tmat = kitti_loader.get_pose(ind)
        print("index={}, Tmat:\n{}".format(ind, Tmat))
        grid_mapper.integrate(ind, images["rgbL"], gmaps, Tmat)


if __name__ == "__main__":
    main()
