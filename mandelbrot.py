import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt


def make_mandelbrot_renderer(pixel_dims):
    xs = jnp.arange(pixel_dims[0])
    ys = jnp.arange(pixel_dims[1])

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

        n = 100
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


def main():
    size = 800
    pixel_dims = (size, size)
    step = 0.005
    position = jnp.zeros(2) - ((size * step) / 2)

    renderer = make_mandelbrot_renderer(pixel_dims)
    pixels = renderer(position, step)

    plt.imshow(pixels.astype(jnp.uint8))
    plt.show()


if __name__ == "__main__":
    jit = True

    if jit:
        main()
    else:
        with jax.disable_jit():
            main()
