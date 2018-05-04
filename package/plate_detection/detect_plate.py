import argparse
import numpy as np
from package.plate_detection.plate_localization import PlateLocalization
from scipy.misc import imshow
from package.plate_detection.misc import bound_to_box
from package.plate_detection.plate_non_plate_classifier import PlateClassifier
from package.plate_detection.vehicle_detection import VehicleDetection
from package.plate_detection.localization_abstract import LocalizationAbstract


class PlateDetection(LocalizationAbstract):
    def __init__(self, window_max_size=(30, 30), window_width_step=2, window_height_step=2):
        self.vd = VehicleDetection()
        self.pl = PlateLocalization(
            window_max_size=window_max_size,
            window_width_step=window_width_step,
            window_height_step=window_height_step
        )
        self.pc = PlateClassifier()

    def find(self, img):
        _, _, box, _, image = self.vd.get_largest_object(img=img)
        vehicle_image = image[
                            int(box[0] * image.shape[0]):int(box[2] * image.shape[0]),
                            int(box[1] * image.shape[1]):int(box[3] * image.shape[1])
                        ]
        bounds, img = self.pl.get_bounds(img=vehicle_image)
        return self.pc.get_largest_plate(np.asarray([bound_to_box(bound) for bound in bounds]), vehicle_image)


def run(args):
    pd = PlateDetection()
    box, img = pd.find(args.image)
    assert box is not None, "No plate was found in the given image."
    if args.view:
        imshow(img)
        imshow(img[box[2]:box[3], box[0]:box[1]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', '--img',
        type=str,
        required=True,
        help='Path to vehicle image.'
    )
    parser.add_argument(
        '--view',
        type=bool,
        default=True,
        help='Path to vehicle image.'
    )
    run(parser.parse_args())
