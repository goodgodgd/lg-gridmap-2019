import os
import glob
import cv2
import numpy as np

import path_provider as pp


class DataLoader:
    def __init__(self, date: str, drive_id: int):
        self.data_path = pp.get_path(date, drive_id)
        if (not os.path.isdir(os.path.join(self.data_path, "image_02"))) \
                and (not os.path.isdir(os.path.join(self.data_path, "image_00"))):
            raise FileNotFoundError("no image directory in {}".format(self.data_path))
        self.default_img_dir = "image_02"
        self.default_image_path = os.path.join(self.data_path, self.default_img_dir, "data")
        print("default image path:", self.default_image_path)
        self.image_list = glob.glob(self.default_image_path + "/*.png")
        self.image_list.sort()

    def __len__(self):
        return len(self.image_list)

    def load_data(self, index: int, asgray: bool, read_gmap: bool):
        assert index < len(self), "index out of range {} >= {}".format(index, len(self))
        default_image_path = self.image_list[index]

        if asgray:
            image_dirs = ["image_00", "image_01"]
        else:
            image_dirs = ["image_02", "image_03"]
        image_list = self.read_images(default_image_path, self.default_img_dir, image_dirs)
        if not read_gmap:
            return image_list

        gmap_dirs = ["Image_FCN", "LiDAR_road", "yolo_distance"]
        gmap_list = []
        for mapdir in gmap_dirs:
            gmap_list.append(self.read_grid_map(default_image_path, mapdir))

        return image_list, gmap_list

    @staticmethod
    def read_images(default_path, default_dir, image_dirs):
        image_list = []
        for imgdir in image_dirs:
            image = cv2.imread(default_path.replace(default_dir, imgdir))
            image_list.append(image)
        return image_list

    def read_grid_map(self, default_path, gmap_name):
        gmap_file = default_path.replace(self.default_img_dir + "/data", gmap_name)
        gmap_file = gmap_file.replace(".png", ".csv")
        grid_map = np.loadtxt(gmap_file, dtype=np.uint8, delimiter=",")
        return grid_map

