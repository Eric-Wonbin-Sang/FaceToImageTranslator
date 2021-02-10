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

from Classes import Camera, Image, PygameHelper
from PygameClasses import EasyRect, EasyText

from General import Constants


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


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


def main():
    pygame.init()

    display_width, display_height = 1000, 320
    display = pygame.display.set_mode((display_width, display_height))

    white = (0, 0, 255)

    clock = pygame.time.Clock()
    run = True

    camera = Camera.Camera(index=2, upside_down=False)

    warper_image = Image.Image(file_path="face_sources/Screenshot 2021-02-04 193704.png", width=256, height=256)
    pseudo_webcam_image = Image.Image(file_path="face_sources/s_face_test.jpg", width=256, height=256)

    run_with_cpu = False
    generator, kp_detector = load_checkpoints(config_path='collab_source/config/vox-256.yaml', checkpoint_path=Constants.secrets_dir + '/FaceToImageTranslator/vox-cpk.pth.tar', cpu=run_with_cpu)

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        display.fill(white)

        webcam_image = camera.get_webcam_image()
        # webcam_image = pseudo_webcam_image.image_np

        PygameHelper.show_numpy_array(webcam_image.image_np, display, x=30, y=30, x_scale=1, y_scale=1)
        warper_image.to_pygame_display(display, x=350, y=30)

        start_datetime = datetime.datetime.now()
        temp = get_face_warp(webcam_frame=webcam_image.image_np, source_image=warper_image.image_np, run_with_cpu=run_with_cpu, generator=generator, kp_detector=kp_detector)
        PygameHelper.show_numpy_array(temp, display, x=670, y=30, x_scale=1, y_scale=1)
        end_datetime = datetime.datetime.now()
        print("generation speed (in seconds) target: {} | actual: {}".format(round(1/30, 3), (end_datetime - start_datetime).microseconds/1000000))

        pygame.display.update()
        # clock.tick(60)

    pygame.quit()


main()
