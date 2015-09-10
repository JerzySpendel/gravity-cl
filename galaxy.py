import numpy as np
import planet
import kernel
import pyopencl as cl
import pygame
import sys
from gameclock import GameClock
import random

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
program = cl.Program(ctx, kernel.program).build()


class Galaxy:
    def __init__(self):
        mf = cl.mem_flags
        self.planets = []
        self.click = np.array([0, 0], dtype=np.float32)
        self.clicked = False
        self.add_planet(planet.Planet(r=[100,100], v=[5, 0]))
        self.add_planet(planet.Planet(r=[100, 90], v=[-5, 0]))
        for x in range(50):
            for y in range(50):
                self.add_planet(planet.Planet(r=[10*x, 10*y]))
        self.planets_np = self.kernel_data()
        self.planets_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.planets_np)
        self.planets_const_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.planets_np)

    def add_planet(self, _planet):
        self.planets.append(_planet)

    def kernel_data(self):
        d = []
        for _planet in self.planets:
            d.append(_planet._numpy_row())
        return np.array(d, dtype=np.float32)

    def apply_dt(self, dt):
        mf = cl.mem_flags
        if self.clicked:
            program.apply_dt(queue, self.planets_np.shape, None, self.planets_buffer, np.int32(len(self.planets_np)), self.click, np.float32(dt))
        else:
            zero_force = np.array([0, 0], dtype=np.float32)
            program.apply_dt(queue, self.planets_np.shape, None, self.planets_buffer, np.int32(len(self.planets_np)), zero_force, np.float32(dt))

    def load_from_gpu(self):
        cl.enqueue_copy(queue, self.planets_np, self.planets_buffer).wait()


class GalaxyView:
    def __init__(self):
        self.galaxy = Galaxy()
        self.screen = pygame.display.set_mode((500, 500))
        self.clock = GameClock(update_callback=self.update, max_ups=30)

    def update(self, dt):
        self.handle_events()
        self.screen.fill((30, 0, 0))
        for i in range(10):
            self.galaxy.apply_dt(dt/(2000*(i+1)))
        self.galaxy.load_from_gpu()
        for _planet in self.galaxy.planets_np:
            temp = _planet.astype(np.int)
            _planet_temp = planet.Planet(r=[temp[0], temp[1]])
            self.screen.set_at(_planet_temp.r, (255, 255, 255))
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.galaxy.clicked = True
                return
            elif event.type == pygame.MOUSEBUTTONUP:
                self.galaxy.clicked = False
                return
            elif event.type == pygame.MOUSEMOTION:
                if self.galaxy.clicked:
                    x, y = event.pos
                    y = 500-y
                    print(x, y)
                    self.galaxy.click = np.array([x, y], dtype=np.float32)

    def main_loop(self):
        while True:
            self.clock.tick()

if __name__ == '__main__':
    g = GalaxyView()
    g.main_loop()
