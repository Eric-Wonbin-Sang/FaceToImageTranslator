import cv2
import numpy
import pygame
import matplotlib.image
from skimage.transform import resize as sk_resize


class Image:

    def __init__(self, file_path=None, source_np=None, width=None, height=None, x_scalar=None, y_scalar=None):

        self.file_path = file_path
        self.source_np = matplotlib.image.imread(self.file_path) if file_path is not None else source_np

        self.width, self.height = (width, height) if file_path is not None else (self.source_np.shape[1], self.source_np.shape[0])
        self.x_scalar, self.y_scalar = x_scalar, y_scalar

        self.image_np = self.get_image_np()

    def update(self, **kwargs):
        if kwargs.get("width"):
            self.width = kwargs.get("width")
        if kwargs.get("height"):
            self.height = kwargs.get("height")
        if kwargs.get("x_scalar"):
            self.x_scalar = kwargs.get("x_scalar")
        if kwargs.get("y_scalar"):
            self.y_scalar = kwargs.get("y_scalar")
        return self

    def get_image_np(self):
        if self.width and self.height:
            return resize_wh(self.source_np, self.width, self.height)
        elif self.x_scalar and self.y_scalar:
            return resize_scalar(self.source_np, self.x_scalar, self.y_scalar)
        else:
            return self.source_np

    def to_pygame_display(self, display, x, y):
        image_np = (self.image_np * 255).astype('uint8')
        np_array = numpy.rot90(image_np)
        pygame_surface = pygame.surfarray.make_surface(np_array)
        display.blit(pygame_surface, (x, y))


def resize_wh(image_np, width, height):
    return sk_resize(image_np, (int(width), int(height)))[..., :3]


def resize_scalar(image_np, x_scale, y_scale):
    return sk_resize(image_np, int(image_np.shape[1] * x_scale), int(image_np.shape[0] * y_scale))
