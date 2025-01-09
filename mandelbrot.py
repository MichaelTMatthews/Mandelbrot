import jax
import jax.numpy as jnp
import numpy as np
import pygame
from matplotlib import pyplot as plt


def make_mandelbrot_renderer(pixel_dims):
    xs = jnp.arange(pixel_dims[0]) - pixel_dims[0] // 2
    ys = jnp.arange(pixel_dims[1]) - pixel_dims[1] // 2

    xs = jnp.repeat(xs[:, None], repeats=pixel_dims[1], axis=1)
    ys = jnp.repeat(ys[None, :], repeats=pixel_dims[0], axis=0)

    uniform_xy = jnp.concatenate([xs[:, :, None], ys[:, :, None]], axis=-1)

    @jax.jit
    def render(position, step):
        xy = position[None, None, :] + (uniform_xy * step)
        z = jnp.zeros((*pixel_dims, 2))

        def complex_square(c):
            r = c[0] * c[0] - c[1] * c[1]
            i = 2 * c[0] * c[1]
            return jnp.array([r, i])

        def _apply_f(carry, t):
            z, diverged_n = carry

            new_z = jax.vmap(jax.vmap(complex_square))(z) + xy
            new_z_mag = jax.vmap(jax.vmap(jnp.linalg.norm))(new_z)

            diverged_n = jax.vmap(
                jax.vmap(
                    lambda x, y: jax.lax.select(x < 0, jax.lax.select(y > 2, t, x), x)
                )
            )(diverged_n, new_z_mag)

            return (new_z, diverged_n), None

        n = 1000
        (z, diverged_n), _ = jax.lax.scan(
            _apply_f, (z, -jnp.ones(pixel_dims, dtype=int)), jnp.arange(n)
        )

        c_centre = jnp.zeros(3, dtype=jnp.uint8)
        c1 = jnp.array([255, 0, 0], dtype=jnp.uint8)
        c2 = jnp.array([0, 255, 0], dtype=jnp.uint8)

        c1 = jnp.array([255, 100, 0], dtype=jnp.uint8)
        c2 = jnp.array([0, 0, 255], dtype=jnp.uint8)

        pixels = (diverged_n == -1)[:, :, None] * c_centre[None, None, :]
        pixels += ((diverged_n != -1)[:, :, None] * diverged_n[:, :, None] / n) * c1[
            None, None, :
        ]
        pixels += (
            (diverged_n != -1)[:, :, None]
            * (1 - diverged_n[:, :, None] / n)
            * c2[None, None, :]
        )

        return pixels

    return render


class MandelbrotRenderer:
    def __init__(self, size, pixel_render_size=1):
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []

        self.screen_size = (size * pixel_render_size, size * pixel_render_size)

        # Init rendering
        pygame.init()
        pygame.key.set_repeat(250, 75)

        self.screen_surface = pygame.display.set_mode(self.screen_size)

        self._render = make_mandelbrot_renderer((size, size))

        self.position = jnp.zeros(2)
        self.step = 0.005

    def update(self):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        # Update screen
        pygame.display.flip()
        # time.sleep(0.01)

    def render(self):
        # Clear
        self.screen_surface.fill((0, 0, 0))

        pixels = self._render(self.position, self.step)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)

        surface = pygame.surfarray.make_surface(np.array(pixels).transpose((1, 0, 2)))
        self.screen_surface.blit(surface, (0, 0))

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False

    def get_action_from_keypress(self):
        translate_speed = 10.0
        zoom_amount = 1.1

        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self.position += jnp.array([-self.step * translate_speed, 0])
                    return True
                if event.key == pygame.K_s:
                    self.position += jnp.array([self.step * translate_speed, 0])
                    return True
                if event.key == pygame.K_a:
                    self.position += jnp.array([0, -self.step * translate_speed])
                    return True
                if event.key == pygame.K_d:
                    self.position += jnp.array([0, self.step * translate_speed])
                    return True
                if event.key == pygame.K_EQUALS:
                    self.step /= zoom_amount
                    return True
                if event.key == pygame.K_MINUS:
                    self.step *= zoom_amount
                    return True

        return False


def main():
    size = 800

    renderer = MandelbrotRenderer(size)
    renderer.render()

    clock = pygame.time.Clock()

    while not renderer.is_quit_requested():
        action = renderer.get_action_from_keypress()

        if action:
            renderer.render()

        renderer.update()
        clock.tick(60)


if __name__ == "__main__":
    jit = True

    if jit:
        main()
    else:
        with jax.disable_jit():
            main()
