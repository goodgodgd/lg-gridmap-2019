import os
import glob
import cv2
import numpy as np

import path_provider as pp


class DataLoader:
    def __init__(self, date: str, drive_id: int):
        self.data_path = pp.get_path(date, drive_id)
        print("datapath", self.data_path)
        self.default_image_dir = "image_00"
        if not os.path.isdir(os.path.join(self.data_path, self.default_image_dir)):
            self.default_image_dir = "image_02"
            if not os.path.isdir(os.path.join(self.data_path, self.default_image_dir)):
                raise FileNotFoundError("no image directory in {}".format(self.data_path))
        self.default_image_path = os.path.join(self.data_path, self.default_image_dir, "data")
        self.image_list = glob.glob(self.default_image_path + "/*.png")
        self.image_list.sort()

    def __len__(self):
        return len(self.image_list)

    def load_data(self, index: int, gray: bool, gmap: bool):
        assert index < len(self), "index out of range {} >= {}".format(index, len(self))
        
        default_image_path = self.image_list[index]
        if gray:
            grayL = default_image_path.replace(self.default_image_dir, "image_00")
            imL = cv2.imread(grayL)
            grayR = default_image_path.replace(self.default_image_dir, "image_01")
            imR = cv2.imread(grayR)
        else:
            rgbL = default_image_path.replace(self.default_image_dir, "image_00")
            imL = cv2.imread(rgbL)
            rgbR = default_image_path.replace(self.default_image_dir, "image_01")
            imR = cv2.imread(rgbR)

        if not gmap:
            return imL, imR

        fcn_gmap = self.read_grid_map(default_image_path, "Image_FCN")
        lidar_gmap = self.read_grid_map(default_image_path, "LiDAR_road")
        yolo_gmap = self.read_grid_map(default_image_path, "yolo_distance")

        return imL, imR, fcn_gmap, lidar_gmap, yolo_gmap

    def read_grid_map(self, default_path, gmap_name):
        gmap_file = default_path.replace(self.default_image_dir + "/data", gmap_name)
        gmap_file = gmap_file.replace(".png", ".csv")
        grid_map = np.loadtxt(gmap_file, dtype=np.int32, delimiter=",")
        return grid_map

