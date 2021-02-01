import pygame
import numpy
from matplotlib import image
from skimage.transform import resize
from demo import load_checkpoints
from skimage import img_as_ubyte
import cv2
import datetime

import torch
import numpy as np
from tqdm import tqdm
from animate import normalize_kp

from General import  Constants


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


def show_numpy_array(np_array, display, x=0, y=0, x_scale=1.0, y_scale=1.0):
    np_array = cv2.resize(np_array, (int(np_array.shape[1] * x_scale), int(np_array.shape[0] * y_scale)))
    np_array = numpy.rot90(np_array)
    pygame_surface = pygame.surfarray.make_surface(np_array)
    display.blit(pygame_surface, (x, y))


def get_webcam_frame(cap):
    ret, frame = cap.read()
    frame = cv2.flip(frame, -1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (int(frame.shape[1] * .6), int(frame.shape[0] * .6)))
    x = 75
    w = 256
    y = 0
    h = 256
    frame = frame[y:y + h, x:x + w]
    return frame


def main():
    pygame.init()

    display_width = 1000
    display_height = 320

    display = pygame.display.set_mode((display_width, display_height))

    white = (0, 0, 255)

    clock = pygame.time.Clock()
    run = True

    cap = cv2.VideoCapture(0)
    face_image_path = "a_face_test.png"
    source_image = image.imread(face_image_path)
    resized_source_image = resize(source_image, (256, 256))[..., :3]

    temp_face_np_array = (resize(image.imread("s_face_test.jpg"), (256, 256))[..., :3] * 255).astype('uint8')

    run_with_cpu = True
    generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', checkpoint_path=Constants.secrets_dir + '/FaceToImageTranslator/vox-cpk.pth.tar', cpu=run_with_cpu)

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        display.fill(white)

        webcam_frame = get_webcam_frame(cap)
        # webcam_frame = temp_face_np_array

        show_numpy_array(webcam_frame, display, x=30, y=30, x_scale=1, y_scale=1)
        show_numpy_array((resized_source_image * 255).astype('uint8'), display, x=350, y=30, x_scale=1, y_scale=1)

        start_datetime = datetime.datetime.now()
        temp = get_face_warp(webcam_frame=webcam_frame, source_image=resized_source_image, run_with_cpu=run_with_cpu, generator=generator, kp_detector=kp_detector)
        show_numpy_array(temp, display, x=670, y=30, x_scale=1, y_scale=1)
        end_datetime = datetime.datetime.now()
        print("generation speed (in seconds) target: {} | actual: {}".format(round(1/30, 3), (end_datetime - start_datetime).microseconds/1000000))

        pygame.display.update()
        # clock.tick(60)

    pygame.quit()


main()
