import os
import glob
import numpy as np
import pykitti

import path_provider as pp


class DataLoader:
    def __len__(self):
        raise NotImplementedError()

    def get_gray(self, index: int):
        raise NotImplementedError()

    def get_rgb(self, index: int):
        raise NotImplementedError()

    def get_pose(self, index: int):
        raise NotImplementedError()

    def get_gmaps(self, index: int):
        raise NotImplementedError()

    def get_all(self, index: int):
        raise NotImplementedError()


class KittiDataLoader(DataLoader):
    def __init__(self, date: str, drive_id: str):
        self.data_path = pp.get_path(date, drive_id)
        self.default_img_dir = "image_00"
        self.default_img_path = os.path.join(self.data_path, self.default_img_dir, "data")
        print("[KittiDataLoader] image path:", self.default_img_path)
        self.default_img_list = glob.glob(self.default_img_path + "/*.png")
        assert self.default_img_list, "no images in {}".format(self.default_img_path)
        self.default_img_list.sort()
        self.kitti = pykitti.raw(pp.get_data_root(), date, drive_id)

    def __len__(self):
        return len(self.default_img_list)

    def get_gray(self, index: int):
        imL, imR = self.kitti.get_gray(index)
        return {"grayL": np.asarray(imL), "grayR": np.asarray(imR)}

    def get_rgb(self, index: int):
        imL, imR = self.kitti.get_rgb(index)
        return {"rgbL": np.asarray(imL), "rgbR": np.asarray(imR)}

    def get_pose(self, index: int):
        Tmat = self.kitti.oxts[index].T_w_imu
        return Tmat

    def get_gmaps(self, index: int):
        gmap_dirs = {"cam_fcn": "Image_FCN", "cam_yol": "yolo_distance", "lid_seg": "LiDAR_road"}
        grid_maps = {}
        for key, mapdir in gmap_dirs.items():
            grid_maps[key] = self._read_grid_map(index, mapdir)
        return grid_maps

    def get_all(self, index: int):
        grays = self.get_gray(index)
        rgbs = self.get_rgb(index)
        Tmat = self.get_pose(index)
        gmaps = self.get_gray(index)
        grays.update(rgbs)
        return grays, Tmat, gmaps

    def _read_grid_map(self, index: int, gmap_name):
        default_img_path = self.default_img_list[index]
        gmap_file = default_img_path.replace(self.default_img_dir + "/data", gmap_name)
        gmap_file = gmap_file.replace(".png", ".csv")
        grid_map = np.loadtxt(gmap_file, dtype=np.uint8, delimiter=",")
        return grid_map

