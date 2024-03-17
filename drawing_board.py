import myappkit
import numpy as np
import pygame
from skimage.measure import block_reduce
from copy import copy


class DrawingBoard(myappkit.Item):
    def __init__(self):
        super().__init__()
        self.pen_size = 30
        if self.pen_size % 2 == 0:
            self.pen_size += 1

        self.radius = self.pen_size // 2

        self.dot = np.array([[1]*self.pen_size]*self.pen_size)
        for y in range(len(self.dot)):
            for x in range(len(self.dot[0])):
                r = (abs(x - self.radius)**2 + abs(y - self.radius)**2)**0.5
                if r > self.radius:
                    self.dot[y][x] = 0

        self.padding_size = self.pen_size//2 + 1
        self.grid = np.zeros(shape=(28*12 + 2*self.padding_size, 28*12 + 2*self.padding_size))
        self.origin = (201, 1)
        self.rect = (self.origin[0], self.origin[1],
                     len(self.grid) - 2*self.padding_size,
                     len(self.grid[0]) - 2*self.padding_size)

        self.drawing = False
        self.font = pygame.font.Font('Courier Prime Code.ttf', 26)
        self.digit = -1
        self.probability = 0
        self.current_xy = None
        self.prev_xy = None

    def handleEvent(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if myappkit.coords_within_rect(pygame.mouse.get_pos(), self.rect):
                self.drawing = True

        if event.type == pygame.MOUSEBUTTONUP:
            self.drawing = False
            self.prev_xy = None

        if event.type == pygame.MOUSEMOTION and self.drawing:
            if not myappkit.coords_within_rect(pygame.mouse.get_pos(), self.rect):
                self.drawing = False
                self.prev_xy = None

    def draw(self, xy):
        (x, y) = xy
        x, y = x - self.origin[0] + self.padding_size, y - self.origin[1] + self.padding_size
        self.grid[y - self.radius: y + self.radius + 1, x - self.radius: x + self.radius + 1] += self.dot

    def fill_gap(self, p1, p2):
        mid_point = (
            (p1[0] + p2[0]) // 2,
            (p1[1] + p2[1]) // 2,
        )
        self.draw(mid_point)

        distance = np.sqrt(
            (p1[0] - mid_point[0]) ** 2 + (p1[1] - mid_point[1]) ** 2
        )
        if distance >= 0.5 * self.pen_size:
            self.fill_gap(p1, mid_point)

        distance = np.sqrt(
            (p2[0] - mid_point[0]) ** 2 + (p2[1] - mid_point[1]) ** 2
        )
        if distance >= 0.5 * self.pen_size:
            self.fill_gap(p2, mid_point)

    def update(self):
        if self.drawing:
            self.current_xy = pygame.mouse.get_pos()
            self.draw(self.current_xy)

            if self.prev_xy is not None:
                distance = np.sqrt(
                    (self.current_xy[0] - self.prev_xy[0]) ** 2 + (self.current_xy[1] - self.prev_xy[1]) ** 2
                )
                if distance >= 0.5 * self.pen_size:
                    self.fill_gap(self.current_xy, self.prev_xy)

            self.prev_xy = copy(self.current_xy)

    def render(self, window):
        border = (self.rect[0] - 1, self.rect[1] - 1, self.rect[2] + 2, self.rect[3] + 2)
        pygame.draw.rect(window, (255, 255, 255), rect=border, width=1)
        for x in range(self.rect[0], self.rect[0] + self.rect[2]):
            for y in range(self.rect[1], self.rect[1] + self.rect[3]):
                whiteness = self.grid[y - self.origin[1] + self.padding_size][x - self.origin[0] + self.padding_size]
                whiteness = min(whiteness, 1)
                c = (255 * whiteness,
                     255 * whiteness,
                     255 * whiteness)
                window.set_at((x, y), c)

        if self.digit != -1:
            # text = 'I am ' + str(self.probability) + '% sure that is a ' + str(self.digit) + '.'
            if self.probability > 50:
                text = 'That is a ' + str(self.digit) + '.'
            else:
                text = 'I am not sure.'

            text = self.font.render(text, False, (255, 255, 255))
            window.blit(text, (15, 352))

    def get_digit(self):
        digit_array = self.grid[self.padding_size: -self.padding_size, self.padding_size: -self.padding_size]
        digit_array = block_reduce(digit_array,
                                   block_size=(12, 12),
                                   func=np.mean)

        for x in range(28):
            for y in range(28):
                digit_array[y][x] = 255 * min(digit_array[y][x], 1)

        return digit_array

    def clear(self):
        self.digit = -1
        self.grid = np.zeros(shape=(28 * 12 + 2 * self.padding_size, 28 * 12 + 2 * self.padding_size))
