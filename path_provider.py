import os


def get_data_root():
    return "/home/ian/workplace/lg_gridmap/igdata"


def get_path(date, drive_id):
    return os.path.join(get_data_root(), date, "{}_drive_{}_sync".format(date, drive_id))
