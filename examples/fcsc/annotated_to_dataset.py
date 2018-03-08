import glob
import json
import os
import os.path as osp

import fcn
import labelme
import numpy as np
import skimage.io


class_names = [
    'background',# 0
    'ball', # 1
    'circle_box', # 2
    'rectangle_box', # 3
    'dumbbell_green', # 4
    'dumbbell_red', # 5
    'hammer', # 6
    'misumi_frame', # 7
    'scissors',# 8
    'screw_driver',# 9
    'towel',# 10
]

def main():
    out_dir = 'dataset'

    json_files = glob.glob('*.json')
    for json_file in sorted(json_files):
        print(json_file)

        data = json.load(open(json_file))
        img = labelme.utils.img_b64_to_array(data['imageData'])
        lbl, label_names = labelme.utils.labelme_shapes_to_label(
            img.shape[:2], data['shapes'])

        # convert local label_value global label_value / class_id
        lbl_global = lbl.copy()
        for label_value, label_name in enumerate(label_names):
            try:
                class_id = int(label_name)
            except ValueError:
                assert label_name == 'background'
                class_id = 0
            lbl_global[lbl == label_value] = class_id
        lbl = lbl_global

        viz = fcn.utils.label2rgb(lbl, img, label_names=class_names)
        import cv2
        cv2.imshow(__file__, viz[:, :, ::-1])
        cv2.waitKey(0)
        # skimage.io.imshow(viz)
        # skimage.io.show()

        basename = osp.splitext(osp.basename(json_file))[0]
        sub_dir = osp.join(out_dir, basename)
        os.makedirs(sub_dir)
        skimage.io.imsave(osp.join(sub_dir, 'image.jpg'), img)
        np.savez_compressed(osp.join(sub_dir, 'label.npz'), lbl)
        print('Saved to: %s' % sub_dir)


if __name__ == '__main__':
    main()
