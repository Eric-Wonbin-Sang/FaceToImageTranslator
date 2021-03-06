import pygame
from pygame import Rect


class EasyRect(Rect):

    def __init__(self, **kwargs):

        self.x = kwargs.get("x", 0)
        self.y = kwargs.get("y", 0)
        self.width = kwargs.get("width", 0)
        self.height = kwargs.get("height", 0)
        self.color = kwargs.get("color", (0, 0, 0))
        self.default_color = self.color
        self.border_thickness = kwargs.get("border_thickness", 0)
        self.draw_center = kwargs.get("draw_center", True)

        super().__init__(
            int(self.x),
            int(self.y),
            self.width,
            self.height
        )

    def update_coordinates(self, x, y, width=None, height=None):
        if width:
            self.width = width
        if height:
            self.height = height
        self.x = x
        self.y = y

    def is_hovered(self, mouse):
        if self.draw_center:
            return self.x - self.width/2 < mouse.x < self.x + self.width/2 and self.y - self.height/2 < mouse.y < self.y + self.height/2
        return self.x < mouse.x < self.x + self.width and self.y < mouse.y < self.y + self.height

    def is_left_clicked(self, mouse):
        return self.is_hovered(mouse) and mouse.left_click

    def is_right_clicked(self, mouse):
        return self.is_hovered(mouse) and mouse.right_click

    def draw(self, screen):
        if self.draw_center:
            pygame.draw.rect(
                screen, self.color,
                (self.x - self.width/2, self.y - self.height/2, self.width, self.height),
                self.border_thickness
            )
        else:
            pygame.draw.rect(
                screen, self.color,
                (self.x, self.y, self.width, self.height),
                self.border_thickness
            )