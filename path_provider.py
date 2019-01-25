import os

__DATA_ROOT = "/home/ian/workplace/lg_gridmap/igdata"


def get_path(date, drive_id):
    return os.path.join(__DATA_ROOT, date, "{}_drive_{:04d}_sync".format(date, drive_id))
