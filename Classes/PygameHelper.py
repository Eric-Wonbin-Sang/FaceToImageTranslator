import cv2
import numpy
import pygame


def show_numpy_array(np_array, display, x=0, y=0, width=None, height=None, x_scale=1.0, y_scale=1.0):
    np_array = cv2.resize(np_array, (int(np_array.shape[1] * x_scale), int(np_array.shape[0] * y_scale)))
    np_array = numpy.rot90(np_array)
    pygame_surface = pygame.surfarray.make_surface(np_array)
    display.blit(pygame_surface, (x, y))
