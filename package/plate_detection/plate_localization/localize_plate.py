import argparse
from scipy.misc import imsave, imshow
from package.plate_detection.misc import read_image, draw_contours_bounds, bound_to_box
import os
from package.plate_detection.plate_localization.sobel_method import sobel_edge_method


class PlateLocalization(object):
    def __init__(self, window_max_size=(30, 30), window_width_step=2, window_height_step=2):
        self.window_sizes = [(i, j) for i in range(2, window_max_size[0], window_width_step)
                             for j in range(2, window_max_size[1], window_height_step)]

    def get_bounds(self, img, area_filter=True, area_ratio=0.008, area_range=(1, 7)):
        img = read_image(img)
        bounds = sobel_edge_method(
            img=img,
            sizes=self.window_sizes,
            area_filter=area_filter,
            area_ratio=area_ratio,
            area_range=area_range,
        )
        return bounds, img


def run(args):
    pd = PlateLocalization(
        window_max_size=args.window_max_size,
        window_width_step=args.window_width_step,
        window_height_step=args.window_height_step
    )
    bounds, img = pd.get_bounds(
        img=args.image,
        area_filter=args.area_filter,
        area_ratio=args.area_ratio,
        area_range=args.area_range
    )
    img_name = os.path.basename(args.image)
    counter = 0
    if args.view:
        imshow(draw_contours_bounds(img.copy(), bounds))
    output_path = os.path.join(args.output_dir, img_name.split('.')[0])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for bound in bounds:
        counter += 1
        box = bound_to_box(bound)
        candidate = img[box[2]:box[3], box[0]:box[1]]
        imsave(
            os.path.join(output_path, img_name.split('.')[0] + "_" + str(counter) + "." + img_name.split('.')[1]),
            candidate
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', '--img',
        type=str,
        help='Path to vehicle image.'
    )
    parser.add_argument(
        '--output_dir', '--out',
        type=str,
        default='',
        help='Path to directory to store images in.'
    )
    parser.add_argument(
        '--window_width_step',
        type=int,
        default=2,
        help='Search window width step.'
    )
    parser.add_argument(
        '--window_height_step',
        type=int,
        default=2,
        help='Search window height step.'
    )
    parser.add_argument(
        '--window_max_size',
        type=tuple,
        default=(30, 30),
        help='Maximum search window size.'
    )
    parser.add_argument(
        '--area_ratio',
        type=float,
        default=0.008,
        help='How large the candidate area should be compared to image area.'
    )
    parser.add_argument(
        '--area_range',
        type=tuple,
        default=(1, 7),
        help='Allowed margin for candidate area ratio.'
    )
    parser.add_argument(
        '--area_filter',
        type=bool,
        default=True,
        help='Whether to filter plate candidates by area.'
    )
    parser.add_argument(
        '--view',
        type=bool,
        default=False,
        help='Whether to view plate image or only save it.'
    )
    run(parser.parse_args())
