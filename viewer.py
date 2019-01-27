import cv2
import numpy as np

import dataset_loader


class DataViewer:
    def __init__(self):
        self.grid_cols = 0
        self.grid_rows = 0
    
    def show(self, data_loader, index, im_width):
        imgs, gmaps = self.load_n_check(data_loader, index)
        imgs, gmaps = self.process_to_disp_img(imgs, gmaps, im_width)

        # concatenate gmaps horizontally
        cancat_gmap = np.concatenate(list(gmaps.values()), axis=1)
        # concatenate gamp with imgs["rgbL"] vertically
        disp_img = np.concatenate([imgs["rgbL"], cancat_gmap], axis=0)

        cv2.imshow("srcdata", disp_img)
        cv2.waitKey(10)

    def load_n_check(self, data_loader, index):
        imgs = data_loader.get_rgb(index)
        gmaps = data_loader.get_gmaps(index)
        print("loaded shapes: ", index, imgs["rgbL"].shape,
              gmaps["cam_fcn"].shape, gmaps["cam_yol"].shape, gmaps["lid_seg"].shape)

        assert (gmaps["cam_fcn"].shape[0] == gmaps["cam_yol"].shape[0])
        assert (gmaps["cam_fcn"].shape[0] == gmaps["lid_seg"].shape[0])
        assert np.sum(gmaps["cam_fcn"] > 1) == 0 \
               and np.sum(gmaps["cam_yol"] > 1) == 0 \
               and np.sum(gmaps["lid_seg"] > 1) == 0
        
        return imgs, gmaps

    def process_to_disp_img(self, imgs, gmaps, im_width):
        grid_cols, grid_rows = gmaps["cam_fcn"].shape
        grid_pixels = im_width // (grid_cols * 3)
        gmap_width = grid_pixels * grid_cols
        im_width = grid_pixels * (grid_cols * 3)

        # grid map to binary image
        for key in gmaps.keys():
            gmaps[key] = self.prep_grid_map(gmaps[key])

        # resize to be concatenated to a single image
        for key in imgs.keys():
            imgs[key] = self.resize_by_width(imgs[key], im_width)
        for key in gmaps.keys():
            gmaps[key] = self.resize_by_width(gmaps[key], gmap_width)
        print("resized shapes: ", imgs["rgbL"].shape,
              gmaps["cam_fcn"].shape, gmaps["cam_yol"].shape, gmaps["lid_seg"].shape)

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

    # TODO
    def show_gmaps(self, gmaps):
        pass


def main():
    viewer = DataViewer()
    kitti_loader = dataset_loader.KittiDataLoader("2011_09_26", "0015")

    for i in range(len(kitti_loader)):
        viewer.show(kitti_loader, i, 720)


if __name__ == "__main__":
    main()
