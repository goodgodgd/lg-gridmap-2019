import cv2
import numpy as np

import dataset_loader
from gridmap_integrator import CELL_VALUE_LIMIT


class DataViewer:
    def __init__(self):
        pass

    def show(self, image, gmaps, wnd_width, wait=0):
        wnd_width, grid_ratio = self.adjust_width(wnd_width, gmaps)
        gridmap = self.gmaps_to_image(gmaps, grid_ratio)
        image = self.fit_to_window(image, wnd_width, max_height=200)
        dispimg = np.concatenate([image, gridmap], axis=0)
        cv2.imshow("grid maps", dispimg)
        cv2.waitKey(wait)

    @staticmethod
    def adjust_width(target_width, gmaps):
        grid_list = list(gmaps.values())
        grid_cols = grid_list[0].shape[1] * len(grid_list)
        dst_width = int(round(target_width / grid_cols)) * grid_cols
        return dst_width, int(dst_width/grid_cols)

    def gmaps_to_image(self, gmaps, ratio: int):
        grid_list = []

        for key, gmap in gmaps.items():
            if -0.0001 < np.max(gmap) < 1.0001:
                gmap = np.clip(gmap, 0, 1)
            else:
                print("gridmap min max", key, np.min(gmap), np.max(gmap))
                gmap = np.clip(gmap, -CELL_VALUE_LIMIT, CELL_VALUE_LIMIT)
                gmap = gmap.astype(np.float) / (2. * CELL_VALUE_LIMIT) + 0.5

            gh, gw = gmap.shape
            gmap = cv2.cvtColor((gmap * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            gmap = cv2.resize(gmap, (gw*ratio, gh*ratio), cv2.INTER_NEAREST)
            gmap = self.put_text(gmap, key)
            grid_list.append(gmap)

        gridmap = np.concatenate(grid_list, axis=1)
        return gridmap

    @staticmethod
    def put_text(image, text):
        coordinate = (10, 20)
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        yellow = (0, 255, 255)
        cv2.putText(image, text, coordinate, font, font_scale, yellow, thickness, cv2.LINE_AA)
        return image

    @staticmethod
    def fit_to_window(image, dst_width, max_height=None):
        im_height, im_width = image.shape[:2]
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
