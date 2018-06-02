from package.plate_detection.object_detection import ObjectDetection


class VehicleDetection(ObjectDetection):
    def __init__(self, thresh=.5):
        super(VehicleDetection, self).__init__(thresh=thresh, classes=[3, 6, 7, 8])
