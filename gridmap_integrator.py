import numpy as np
import dataset_loader


class GridMapIntegrator:
    def __init__(self):
        self.prev_pose = np.identity(4)
        self.GMCOLS = 80
        self.GMROWS = 120
        self.CELLSIZE = 0.5
        self.integ_map = np.zeros((self.GMROWS, self.GMCOLS))
        self.origin_cell = [self.GMROWS, self.GMCOLS/2]
        self.cell_inds = self.cell_indices()
        self.cell_coords = self.cell_coordinates(self.cell_inds)

    def cell_indices(self):
        cells = np.indices((self.GMROWS, self.GMCOLS))
        cells = np.reshape(cells, (2, -1))
        cells = cells.transpose()
        print("cells", cells[0:200:10])
        return cells

    def cell_coordinates(self, cell_inds):
        # x: front, y: left
        x = (self.origin_cell[0] - cell_inds[:, 0]) * self.CELLSIZE
        y = (self.origin_cell[1] - cell_inds[:, 1]) * self.CELLSIZE
        z = np.zeros(x.shape)
        points = np.stack([x, y, z], axis=1)
        print("points", points[0:200:10])
        return points

    def integrate(self, index, grid_maps, curr_pose):
        self.integ_map = self.transform_map(self.integ_map, curr_pose)
        if index < 2:
            self.integ_map = self.update_map(self.integ_map, grid_maps)
        self.prev_pose = curr_pose
        # TODO: viewer띄워서 이동하는거 확인하기
        # ..

    def transform_map(self, prev_integ_map, curr_pose):
        tmatrix = np.invert(self.prev_pose) * curr_pose
        # find coordinates of current cells w.r.t the previous pose
        prev_coords = tmatrix * self.cell_coords
        prev_cell = self.posit_to_cell(prev_coords)
        curr_integ_map = np.zeros((self.GMROWS, self.GMCOLS))

        for prvcell, curcell in zip(prev_cell, self.cell_inds):
            curr_integ_map[curcell] = self.interpolate(prev_integ_map, prvcell)

        return curr_integ_map

    def posit_to_cell(self, points):
        rows = -points[:, 0] / self.CELLSIZE + self.origin_cell[0]
        cols = -points[:, 1] / self.CELLSIZE + self.origin_cell[1]
        return np.array([rows, cols])

    def interpolate(self, gmap, cell):
        if cell[0] < -1 or cell[0] > self.GMROWS+1 \
                or cell[1] < -1 or cell[1] > self.GMCOLS+1:
            return 0

        major_cell = np.round(cell).astype(np.int32)
        return gmap[major_cell]

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
        return integ_map


def main():
    grid_mapper = GridMapIntegrator()
    kitti_loader = dataset_loader.KittiDataLoader("2011_09_26", "0015")
    for ind in range(len(kitti_loader)):
        gmaps = kitti_loader.get_gmaps(ind)
        Tmat = kitti_loader.get_pose(ind)
        print("Tmat: {}\n{}".format(ind, Tmat))
        grid_mapper.integrate(ind, gmaps, Tmat)


if __name__ == "__main__":
    main()
