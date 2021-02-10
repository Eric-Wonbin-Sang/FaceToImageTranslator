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
from PygameClasses import EasyRect, EasyText, PygameMouse

from General import Constants, Functions


def get_choose_text():
    return EasyText.EasyText(
        font_file="PygameClasses/FontFolder/Product Sans Bold.ttf",
        size=30,
        text="choose_camera",
        x=0,
        y=0,
        color=(0, 255, 0)
    )


def get_rect_list(count):
    return [
        EasyRect.EasyRect(
            x=0, y=0,
            width=0, height=0,
            color=(0, 255, 0)
        ) for i in range(count)
    ]


def update_rect_list_to_rows(rect_list, largest_col_size, display_width, choose_text):
    rect_list_list = Functions.group_list_by_size(rect_list, largest_col_size)
    y_offset = 0
    for i, temp_rect_list in enumerate(rect_list_list):
        for u, temp_rect in enumerate(temp_rect_list):
            x_offset = (u + 1) * display_width / (len(temp_rect_list) + 1)
            temp_rect.update_coordinates(
                x=x_offset, y=display_width / 7 + choose_text.y + y_offset,
                width=display_width / 6, height=display_width / 6
            )
        y_offset += rect_list[i - 1].height + display_width / 40


def get_camera_list():
    camera_list = []
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            camera = Camera.Camera(index=i)
            try:
                camera.get_webcam_image()
                camera_list.append(Camera.Camera(index=i, upside_down=False))
            except:
                cap.release()
    return camera_list


def show_cameras(camera_list, rect_list, display):
    for i, camera in enumerate(camera_list):
        temp_rect = rect_list[i]
        temp_image = camera.get_webcam_image()

        new_width = temp_rect.width
        new_height = (new_width / temp_image.width) * temp_image.height
        PygameHelper.show_numpy_array(
            (255 * camera.get_webcam_image().update(
                width=new_width,
                height=new_height
            ).get_image_np()),
            display,
            x=temp_rect.x - temp_rect.width / 2,
            y=temp_rect.y - temp_rect.height / 2 + (temp_rect.height - new_height) / 2
        )


def main():

    pygame.init()
    display_width, display_height = 1000, 500
    display = pygame.display.set_mode((display_width, display_height), pygame.RESIZABLE)
    background_color = (0, 0, 0)
    run = True

    # camera_list = get_camera_list()
    # for camera in camera_list:
    #     print(camera)
    camera_list = [
        Camera.Camera(index=i) for i in [
            0,
            # 1,
            # 2,
            # 3
        ]
    ]

    mouse = PygameMouse.Mouse()
    choose_text = get_choose_text()
    rect_list = get_rect_list(len(camera_list))
    draw_list = [choose_text, *rect_list]

    stage = 0
    loop_count = 0
    main_camera = None

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.VIDEORESIZE:
                display = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                display_width, display_height = event.w, event.h

        display.fill(background_color)

        if stage == 0:
            for i, rect in enumerate(rect_list):
                if rect.is_left_clicked(mouse):
                    stage += 1
                    main_camera = camera_list[i]
                    # print(i, rect.is_left_clicked(mouse))

            choose_text.update_coordinates(x=display_width/2, y=display_width/18, size=display_width/16)
            update_rect_list_to_rows(rect_list, 3, display_width, choose_text)

            for draw in draw_list:
                draw.draw(display)
            show_cameras(camera_list, rect_list, display)

        if stage == 1:
            print(main_camera)

        pygame.display.update()
        mouse.update()
        print("loop: {}".format(loop_count))
        loop_count += 1


main()
