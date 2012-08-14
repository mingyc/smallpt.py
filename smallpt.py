from math import sqrt, pi, sin, cos, fabs
from random import uniform
import sys
import logging
import numpy
from numpy import zeros, dot, cross

# alias 
vector = numpy.array
clamp = numpy.clip
ZERO = zeros([3])

# configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)




RAY_EPSILON = 1e-4
INF = 1e20

width = 1024
height = 768


def normalize(v):
    return v / sqrt(dot(v, v))


class Ray(object):
    def __init__(self, origin, direction):
        self.o = origin
        self.d = direction

    def intersect(self):
        """ Intersects ray with the scene. """
        t = INF
        idx = sphere_num
        for s in spheres[::-1]:
            d = s.intersect(self)
            if d != None and d < t:
                t = d
                idx -= 1
        return (t, idx) if t < INF else None


class Sphere(object):
    def __init__(self, radius, position, emission, color, reflection_type):
        self.r = radius
        self.pos = position
        self.e = emission
        self.c = color
        self.r_type = reflection_type

    def intersect(self, ray):
        """ Solve (D.D)t^2 + 2D.(O-C)t + (O-C).(O-C) - r^2 = 0,
            where ray = o + tD
        """
        ray_sphere_dir = self.pos - ray.o
        half_b = dot(ray_sphere_dir, ray.d)
        det = half_b*half_b - dot(ray_sphere_dir, ray_sphere_dir) + self.r*self.r
        if det < 0:
            return None
        det = sqrt(det)
        t = half_b - det
        if t > RAY_EPSILON:
            return t
        t = half_b + det
        if t > RAY_EPSILON:
            return t
        return None



spheres = (
        Sphere(1e5, vector([1e5+1, 40.8, 81.6]),   ZERO, vector([.75, .25, .25]), 'DIFF'),  # Left
        Sphere(1e5, vector([-1e5+99, 40.8, 81.6]), ZERO, vector([.25, .25, .75]), 'DIFF'),  # Right
        Sphere(1e5, vector([50, 40.8, 1e5]),       ZERO, vector([.75, .75, .75]), 'DIFF'),  # Back
        Sphere(1e5, vector([50, 40.8, -1e5+170]),  ZERO, ZERO,          'DIFF'),  # Front
        Sphere(1e5, vector([50, 1e5, 81.6]),       ZERO, vector([.75, .75, .75]), 'DIFF'),  # Bottom
        Sphere(1e5, vector([50, -1e5+81.6, 81.6]), ZERO, vector([.75, .75, .75]), 'DIFF'),  # Top
        Sphere(16.5,vector([27, 16.5, 47]),        ZERO, vector([1., 1., 1.])*999,'SPEC'),  # Mirror
        Sphere(16.5,vector([73, 16.5, 78]),        ZERO, vector([1., 1., 1.])*999,'REFR'),  # Glass
        Sphere(600, vector([50, 681.6-.27, 81.6]), vector([12, 12, 12]), ZERO,    'DIFF'),  # Lite
)
sphere_num = len(spheres)



def radiance(ray, depth, Xi, E=1.):
    if depth > 10:
        return ZERO
    hit = ray.intersect()
    if not hit:
        return ZERO

    t, sid = hit
    obj = spheres[sid]
    x = ray.o + ray.d*t  # ray-scene intersection point
    n = normalize(x - obj.pos)  # sphere normal
    nl = n if dot(n, ray.d) else -n
    f = obj.c  # sphere BRDF modulator
    p = f[0] if (f[0] > f[1] and f[0] > f[2]) else ( f[1] if f[1] > f[2] else f[2] )
    depth += 1
    if depth > 5 or not p:
        if uniform(0., 1.) < p:
            f *= 1./p
        else:
            return obj.e * E

    # ideal diffuse reflection
    if obj.r_type == 'DIFF':
        r1 = 2 * pi * uniform(0., 1.)  # sample an angle
        r2 = uniform(0., 1.)  
        r2s = sqrt(r2)  # sample a distance from center
        # create a random orthonomarl coordinate frame (w, u, v)
        w = nl
        u = normalize(cross((vector([0, 1., 0]) if fabs(w[0]) > .1 else vector([1., 0, 0])), w))
        v = cross(w, u)
        d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2))  # a random reflection ray
        return obj.e + f * radiance(Ray(x, d), depth, Xi)
    # ideal specular reflection
    elif obj.r_type == 'SPEC':
        return obj.e + f * radiance(Ray(x, ray.d-n*2*dot(n, ray.d)), depth, Xi)
    # ideal dielectric refraction
    refl_ray = Ray(x, ray.d-n*2*dot(n, ray.d))
    into = dot(n, nl) > 0  # if ray from outside in
    nc = 1
    nt = 1.5
    nnt = nc/nt if into else nt/nc
    ddn = dot(ray.d, nl)
    cos2t = 1 - nnt*nnt*(1 - ddn*ddn)
    # total internal relection
    if cos2t < 0:
        return obj.e + f * radiance(refl_ray, depth, Xi)
    # choose reflection/refraction
    tdir = normalize(ray.d*nnt - n*((1 if into else -1)*(ddn*nnt + sqrt(cos2t))))
    a = nt-nc
    b = nt+nc
    R0 = a*a / (b*b)
    c = 1 - (-ddn if into else dot(tdir, n))
    Re = R0 + (1-R0)*c**4
    Tr = 1 - Re
    P = .25 + .5*Re
    RP = Re/P
    TP = Tr / (1-P)
    # russian roulette
    return obj.e + f*( (radiance(refl_ray, depth, Xi)*RP if uniform(0., 1.)<P else 
                        radiance(Ray(x, tdir), depth, Xi)*TP
                       ) 
                       if depth>2 else radiance(refl_ray, depth, Xi)*Re+radiance(Ray(x, tdir), depth, Xi)*Tr )


    # loop over any lights


    return


def render(samples):
    img = [0]*(width * height)
    for y in range(height):
        logger.info('Rendering ({0} spp) {1}%'.format(samples*4, 100.*y/(height-1)))
        Xi = (0, 0, y*y*y)
        for x in range(width):
            i = (height - y - 1)*width + x
            for sy in range(2):
                r = ZERO
                for sx in range(2):
                    for s in range(samples):
                        r1 = 2 * uniform(0., 1.)
                        r2 = 2 * uniform(0., 1.)
                        dx = sqrt(r1)-1 if r1 < 1 else 1-sqrt(2-r1)
                        dy = sqrt(r2)-1 if r2 < 1 else 1-sqrt(2-r2)
                        d = camera_x * (((sx+.5 + dx)/2 + x)/width  - .5) + camera_y * (((sy+.5 + dy)/2 + y)/height - .5) + camera.d
                        r += radiance(Ray(camera.o+d*140, normalize(d)), 0, Xi) * (1./samples)
                img[i] += clamp(r, 0, 1) * .25
    return img





camera = Ray(vector([50, 52, 295.6]), normalize(vector([0, -0.042612, -1])))
camera_x = vector([width*.5135/height, 0, 0])
camera_y = normalize(cross(camera_x, camera.d)) * .5135


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: {0} sample_per_pixel'.format(sys.argv[0])
        exit()
    samples = int(sys.argv[1])/4 if len(sys.argv) == 2 else 1
    img = render(samples)
    # write to file
    with open('sample.ppm', 'w') as f:
        f.write('P3\n%d %d\n%d\n' % (width, height, 255))
        for px in img:
            f.write('%d %d %d ' % (px.x, px.y, px.z))
    print 'Done.'
