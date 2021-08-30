import cv2
from skimage.transform import resize
from collab_source.demo import load_checkpoints
from skimage import img_as_ubyte
import datetime
import matplotlib

import torch
import numpy as np
from collab_source.animate import normalize_kp

from General import Constants


class Webcam:

    def __init__(self, camera_index, cap):

        self.camera_index = camera_index
        self.cap = cap

        self.init_height, self.init_width = self.get_init_dimensions()
        self.new_height, self.new_width, self.y_start, self.y_end, self.x_start, self.x_end = get_resize_values(
            self.init_height, self.init_width, Constants.image_target_height, Constants.image_target_width)

        self.frame = self.set_transformed_frame()

    def get_init_dimensions(self):
        ret, frame = self.cap.read()
        return frame.shape[0], frame.shape[1]

    def set_transformed_frame(self):
        ret, frame = self.cap.read()
        self.frame = cv2.resize(
            frame, (self.new_width, self.new_height), interpolation=cv2.INTER_AREA
        )[self.y_start:self.y_end, self.x_start:self.x_end]
        return self.frame


class Picture:

    def __init__(self, file_path):

        self.file_path = file_path
        self.init_np_array = cv2.imread(self.file_path)

        self.init_height, self.init_width = self.get_init_dimensions()
        self.new_height, self.new_width, self.y_start, self.y_end, self.x_start, self.x_end = get_resize_values(
            self.init_height, self.init_width, Constants.image_target_height, Constants.image_target_width)

        self.frame = self.set_transformed_frame()

    def get_init_dimensions(self):
        return self.init_np_array.shape[0], self.init_np_array.shape[1]

    def set_transformed_frame(self):
        self.frame = cv2.resize(
            self.init_np_array, (self.new_width, self.new_height), interpolation=cv2.INTER_AREA
        )[self.y_start:self.y_end, self.x_start:self.x_end]
        return self.frame


def get_webcam(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    return Webcam(camera_index, cap)


def get_resize_values(init_height, init_width, target_height, target_width):
    """ Values for transforming an image via scaling and cropping """
    new_height, new_width = int(target_height), int(init_width / init_height * target_width)
    y_start, y_end = 0, -1
    x_start, x_end = int((new_width - target_width) / 2), int(new_width - (new_width - target_width) / 2)
    return new_height, new_width, y_start, y_end, x_start, x_end


# ------ OLD CODE -------------------------------------------
def get_prediction_np_array(source_image, driver, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

    driving = torch.tensor(np.array([driver])[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
    kp_source = kp_detector(source)
    kp_driving_initial = kp_detector(driving[:, :, 0])

    driving_frame = driving[:, :, 0]
    if not cpu:
        driving_frame = driving_frame.cuda()
    kp_driving = kp_detector(driving_frame)
    kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                           kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                           use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
    out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

    return np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
# ------ OLD CODE -------------------------------------------


def main():

    webcam = get_webcam(1)
    face_picture = Picture("C:/Users/ericw/Desktop/nic.png")

    run_with_cpu = False

    generator, kp_detector = load_checkpoints(
        config_path='collab_source/config/vox-256.yaml',
        checkpoint_path=Constants.secrets_dir + '/FaceToImageTranslator/vox-cpk.pth.tar',
        cpu=run_with_cpu
    )

    while True:

        if webcam.set_transformed_frame() is None:
            continue

        start_datetime = datetime.datetime.now()
        face_warp_list = [
            get_prediction_np_array(face_picture.frame / 255, webcam.frame / 255, generator, kp_detector, relative=False, adapt_movement_scale=False, cpu=run_with_cpu),
            # get_prediction_np_array(face_picture.frame / 255, webcam.frame / 255, generator, kp_detector, relative=False, adapt_movement_scale=True, cpu=run_with_cpu),
            # get_prediction_np_array(face_picture.frame / 255, webcam.frame / 255, generator, kp_detector, relative=True, adapt_movement_scale=False, cpu=run_with_cpu),   # No change
            # get_prediction_np_array(face_picture.frame / 255, webcam.frame / 255, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=run_with_cpu)     # No change
        ]
        end_datetime = datetime.datetime.now()
        print("FPS: {}".format(1000000/(end_datetime - start_datetime).microseconds))

        cv2.imshow('Input', webcam.frame)
        cv2.imshow('Face', face_picture.frame)
        for i, face_warp in enumerate(face_warp_list):
            cv2.imshow("Output {}".format(i), face_warp)

        c = cv2.waitKey(1)
        if c == 27:
            break

    webcam.cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
