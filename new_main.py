import cv2
import pygame
from skimage.transform import resize
from collab_source.demo import load_checkpoints
from skimage import img_as_ubyte
import datetime

import torch
import numpy as np
from tqdm import tqdm
from collab_source.animate import normalize_kp

from General import Constants


class Webcam:

    def __init__(self, camera_index, cap):

        self.camera_index = camera_index
        self.cap = cap

        self.init_height, self.init_width = self.get_init_dimensions()
        self.resize_height, self.resize_width = self.get_resize_values()
        self.y_start, self.y_end, self.x_start, self.x_end = self.get_crop_values()

        self.frame = self.set_transformed_frame()

    def get_init_dimensions(self):
        ret, frame = self.cap.read()
        return frame.shape[0], frame.shape[1]

    def get_resize_values(self):
        return int(Constants.image_target_height), int(self.init_width/self.init_height * Constants.image_target_width)

    def get_crop_values(self):
        y_start, y_end = 0, -1
        x_start, x_end = int((self.resize_width - Constants.image_target_width)/2), \
            int(self.resize_width - (self.resize_width - Constants.image_target_width)/2)
        return y_start, y_end, x_start, x_end

    def set_transformed_frame(self):
        ret, frame = self.cap.read()
        self.frame = cv2.resize(frame, (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)
        self.frame = self.frame[self.y_start:self.y_end, self.x_start:self.x_end]
        return self.frame


def get_webcam(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    return Webcam(camera_index, cap)


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


def get_face_warp(webcam_frame, source_image, run_with_cpu, generator, kp_detector):
    driving_frame = resize(webcam_frame, (256, 256))[..., :3]
    prediction = get_prediction_np_array(source_image, driving_frame, generator, kp_detector, relative=False, adapt_movement_scale=True, cpu=run_with_cpu)
    return img_as_ubyte(prediction)


def face():
    run_with_cpu = False
    generator, kp_detector = load_checkpoints(config_path='collab_source/config/vox-256.yaml',
                                              checkpoint_path=Constants.secrets_dir + '/FaceToImageTranslator/vox-cpk.pth.tar',
                                              cpu=run_with_cpu)
    run = True
    while run:

        # webcam_image = camera.get_webcam_image()
        # webcam_image = pseudo_webcam_image.image_np

        start_datetime = datetime.datetime.now()
        temp = get_face_warp(webcam_frame=webcam_image.image_np, source_image=warper_image.image_np, run_with_cpu=run_with_cpu, generator=generator, kp_detector=kp_detector)
        PygameHelper.show_numpy_array(temp, display, x=670, y=30, x_scale=1, y_scale=1)
        end_datetime = datetime.datetime.now()
        print("generation speed (in seconds) target: {} | actual: {}".format(round(1/30, 3), (end_datetime - start_datetime).microseconds/1000000))
# ------ OLD CODE -------------------------------------------


def main():

    webcam = get_webcam(1)

    while True:

        if webcam.set_transformed_frame() is None:
            continue

        cv2.imshow('Input', webcam.frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    webcam.cap.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    main()
