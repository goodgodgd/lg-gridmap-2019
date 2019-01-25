import cv2
import numpy as np

import kitti_data_loader


class DataViewer:
    def __init__(self):
        self.grid_cols = 0
        self.grid_rows = 0
    
    def show(self, data_loader, index, im_width):
        imL, imR, fcn_gmap, lidar_gmap, yolo_gmap = self.load_n_check(data_loader, index)
        imL, imR, fcn_gmap, lidar_gmap, yolo_gmap = \
            self.process_to_disp_img(imL, imR, fcn_gmap, lidar_gmap, yolo_gmap, im_width)

        # concatenate gmaps horizontally
        gmaps = np.concatenate([fcn_gmap, lidar_gmap, yolo_gmap], axis=1)
        # concatenate gamp with imL vertically
        disp_img = np.concatenate([imL, gmaps], axis=0)

        cv2.imshow("srcdata", disp_img)
        cv2.waitKey(10)

    def load_n_check(self, data_loader, index):
        imL, imR, fcn_gmap, lidar_gmap, yolo_gmap \
            = data_loader.load_data(index, False, True)
        print("loaded shapes: ", index, imL.shape, imR.shape,
              fcn_gmap.shape, lidar_gmap.shape, yolo_gmap.shape)

        assert (fcn_gmap.shape[0] == lidar_gmap.shape[0])
        assert (fcn_gmap.shape[0] == yolo_gmap.shape[0])
        self.grid_rows = fcn_gmap.shape[0]
        self.grid_cols = fcn_gmap.shape[1]
        
        return imL, imR, fcn_gmap, lidar_gmap, yolo_gmap

    def process_to_disp_img(self, imL, imR, fcn_gmap, lidar_gmap, yolo_gmap, im_width):
        grid_cols, grid_rows = fcn_gmap.shape
        grid_pixels = im_width // (grid_cols * 3)
        gmap_width = grid_pixels * grid_cols
        im_width = grid_pixels * (grid_cols * 3)

        # grid map to binary image
        fcn_gmap = self.prep_grid_map(fcn_gmap)
        lidar_gmap = self.prep_grid_map(lidar_gmap)
        yolo_gmap = self.prep_grid_map(yolo_gmap)

        imL = self.resize_by_width(imL, im_width)
        fcn_gmap = self.resize_by_width(fcn_gmap, gmap_width)
        lidar_gmap = self.resize_by_width(lidar_gmap, gmap_width)
        yolo_gmap = self.resize_by_width(yolo_gmap, gmap_width)

        print("resized shapes: ", imL.shape,
              fcn_gmap.shape, lidar_gmap.shape, yolo_gmap.shape)

        return imL, imR, fcn_gmap, lidar_gmap, yolo_gmap

    @staticmethod
    def prep_grid_map(gmap):
        gmap = gmap * 255
        gmap = cv2.cvtColor(gmap, cv2.COLOR_GRAY2BGR)
        return gmap

    @staticmethod
    def resize_by_width(image, width):
        ih, iw = image.shape[:2]
        height = (ih * width) // iw
        image = cv2.resize(image, (width, height), cv2.INTER_NEAREST)
        # print("reshape before:", ih, iw, "/ after:", image.shape)
        return image


def main():
    viewer = DataViewer()
    data_loader = kitti_data_loader.DataLoader("2011_09_26", 15)

    for i in range(len(data_loader)):
        viewer.show(data_loader, i, 720)


if __name__ == "__main__":
    main()
