import cv2
import numpy as np

import kitti_data_loader


class DataViewer:
    def __init__(self):
        self.grid_cols = 0
        self.grid_rows = 0
    
    def show(self, data_loader, index, im_width):
        imgs, gmaps = self.load_n_check(data_loader, index)
        imgs, gmaps = self.process_to_disp_img(imgs, gmaps, im_width)

        # concatenate gmaps horizontally
        gmaps = np.concatenate(gmaps, axis=1)
        # concatenate gamp with imgs[0] vertically
        disp_img = np.concatenate([imgs[0], gmaps], axis=0)

        cv2.imshow("srcdata", disp_img)
        cv2.waitKey(10)

    def load_n_check(self, data_loader, index):
        imgs, gmaps = data_loader.load_data(index, False, True)
        print("loaded shapes: ", index, imgs[0].shape, imgs[1].shape,
              gmaps[0].shape, gmaps[1].shape, gmaps[2].shape)

        assert (gmaps[0].shape[0] == gmaps[1].shape[0])
        assert (gmaps[0].shape[0] == gmaps[2].shape[0])
        self.grid_rows = gmaps[0].shape[0]
        self.grid_cols = gmaps[0].shape[1]
        
        return imgs, gmaps

    def process_to_disp_img(self, imgs, gmaps, im_width):
        grid_cols, grid_rows = gmaps[0].shape
        grid_pixels = im_width // (grid_cols * 3)
        gmap_width = grid_pixels * grid_cols
        im_width = grid_pixels * (grid_cols * 3)

        # grid map to binary image
        for i in range(3):
            gmaps[i] = self.prep_grid_map(gmaps[i])

        # resize to be concatenated to a single image
        for i in range(2):
            imgs[i] = self.resize_by_width(imgs[i], im_width)
        for i in range(3):
            gmaps[i] = self.resize_by_width(gmaps[i], gmap_width)
        print("resized shapes: ", imgs[0].shape,
              gmaps[0].shape, gmaps[1].shape, gmaps[2].shape)

        return imgs, gmaps

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
