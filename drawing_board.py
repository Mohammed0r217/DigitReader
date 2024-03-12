import myappkit
import numpy as np
import pygame
from skimage.measure import block_reduce


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

    def handleEvent(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            xy = pygame.mouse.get_pos()
            if myappkit.coords_within_rect(xy, self.rect):
                self.drawing = True
                self.draw(xy)

        if event.type == pygame.MOUSEBUTTONUP:
            self.drawing = False

        if event.type == pygame.MOUSEMOTION and self.drawing:
            xy = pygame.mouse.get_pos()
            if myappkit.coords_within_rect(xy, self.rect):
                self.draw(xy)
            else:
                self.drawing = False

    def draw(self, xy):
        (x, y) = xy
        x, y = x - self.origin[0] + self.padding_size, y - self.origin[1] + self.padding_size
        self.grid[y - self.radius: y + self.radius + 1, x - self.radius: x + self.radius + 1] += self.dot

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
            text = 'That is a ' + str(self.digit) + '.'
            text = self.font.render(text, False, (255, 255, 255))
            window.blit(text, (15, 352))

    def get_digit(self):
        digit_array = self.grid[self.padding_size: -self.padding_size, self.padding_size: -self.padding_size]
        digit_array = block_reduce(digit_array,
                                   block_size=(12, 12),
                                   func=np.mean)

        for x in range(28):
            for y in range(28):
                digit_array[y][x] = min(digit_array[y][x], 1)

        return digit_array

    def clear(self):
        self.digit = -1
        self.grid = np.zeros(shape=(28 * 12 + 2 * self.padding_size, 28 * 12 + 2 * self.padding_size))
