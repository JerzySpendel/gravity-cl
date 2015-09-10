import pyopencl as cl
import kernel
import galaxy
g = galaxy.Galaxy()
for _ in range(10):
    g.apply_dt(0.15)
print(g.planets_np)
