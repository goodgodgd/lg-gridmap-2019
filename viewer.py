import cv2
import numpy as np

import dataset_loader


class DataViewer:
    def __init__(self):
        pass

    def show(self, image, gmaps, wnd_width, wait=0):
        gridmap = self.gmaps_to_image(gmaps)
        gridmap = self.fit_to_window(gridmap, wnd_width, int_ratio=True)
        image = self.fit_to_window(image, gridmap.shape[1], max_height=200)
        dispimg = np.concatenate([image, gridmap], axis=0)
        cv2.imshow("viewer", dispimg)
        cv2.waitKey(wait)

    @staticmethod
    def gmaps_to_image(gmaps):
        grid_list = []
        for key, gmap in gmaps.items():
            assert np.max(gmap) < 1.001
            grid_list.append((gmap * 255).astype(np.uint8))
        gridmap = np.concatenate(grid_list, axis=1)
        gridmap = cv2.cvtColor(gridmap, cv2.COLOR_GRAY2BGR)
        return gridmap

    @staticmethod
    def fit_to_window(image, dst_width, int_ratio=False, max_height=None):
        im_height, im_width = image.shape[:2]
        if int_ratio:
            dst_width = int(round(dst_width / im_width)) * image.shape[1]
        dst_height = int(round(dst_width / im_width * im_height))

        if max_height and dst_height > max_height:
            adj_height = max_height
            adj_width = round(max_height / im_height * im_width)
            center = cv2.resize(image, (adj_width, adj_height), cv2.INTER_NEAREST)
            image = np.zeros(shape=(adj_height, dst_width, 3), dtype=np.uint8)
            begcol = (dst_width - adj_width) // 2
            image[:, begcol:(dst_width-begcol), :] = center
        else:
            image = cv2.resize(image, (dst_width, dst_height), cv2.INTER_NEAREST)
        return image


def main():
    kitti_loader = dataset_loader.KittiDataLoader("2011_09_26", "0015")
    viewer = DataViewer()

    for i in range(len(kitti_loader)):
        images = kitti_loader.get_rgb(i)
        gmaps = kitti_loader.get_gmaps(i)
        viewer.show(images["rgbL"], gmaps, 1000)


if __name__ == "__main__":
    main()
