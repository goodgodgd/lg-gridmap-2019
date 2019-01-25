import data_loader

if __name__ == "__main__":
    dloader = data_loader.DataLoader("2011_09_26", 15)

    for i in range(len(dloader)):
        imL, imR, fcn_gmap, lidar_gmap, yolo_gmap = dloader.load_data(i, False, True)
        print("shapes: ", i, imL.shape, imR.shape, fcn_gmap.shape, lidar_gmap.shape, yolo_gmap.shape)
