import math
import numpy as np

O = (0,0,0,1)

class Tuple:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    def create(x, y, z, w):
        return (x,y,z,w)

def vector(x,y,z):
    return Tuple.create(x,y,z,0.0)

def point(x,y,z):
    return Tuple.create(x,y,z,1.0)

class Vector:
    def __init__(self,x,y,z):
        return vector(x,y,z)

class Point:
    def __init__(self,x,y,z):
        return point(x,y,z)
    
EPSILON = 0.00001
OO = 100000000

def equal(a,b):
    if abs(a-b)<EPSILON:
        return 1
    else:
        return 0

def sumt(*item):
    x=0
    y=0
    z=0
    w=0
    for i in item:
            x+=i[0]
            y+=i[1]
            z+=i[2]
            w+=i[3]
    return (x,y,z,w)

def subt(*item):
    x=2*item[0][0]
    y=2*item[0][1]
    z=2*item[0][2]
    w=2*item[0][3]
    for i in item:
            x-=i[0]
            y-=i[1]
            z-=i[2]
            w-=i[3]
    return (x,y,z,w)

def negt(item):
    return (-item[0],-item[1],-item[2],-item[3])

def smult(tuple,scalar):
    return (np.round(scalar*tuple[0],6),np.round(scalar*tuple[1],6),np.round(scalar*tuple[2],6),np.round(scalar*tuple[3],6))

def sdivt(tuple,scalar):
    return (np.round(tuple[0]/scalar,6),np.round(tuple[1]/scalar,6),np.round(tuple[2]/scalar,6),np.round(tuple[3]/scalar,6))

def magn(tuple):
    preroot=0
    for i in tuple:
        preroot+=i**2
    return np.sqrt(preroot)

def norm(tuple):
    m = magn(tuple)
    if (m==0):
        return tuple
    else:
        return sdivt(tuple,m)

def dot(tuple1,tuple2):
    return sum([i*j for (i, j) in zip(tuple1, tuple2)])
        
def cross(a,b):
    return vector(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])

class Projectile:
    def __init__(self,pos:Point,vel:Vector):
        self.pos=pos
        self.vel=vel

class Environment:
    def __init__(self,gravity:Vector,wind:Vector):
        self.gravity=gravity
        self.wind=wind

def tick(env:Environment, proj:Projectile):
    position = sumt(proj.pos,proj.vel)
    velocity = sumt(proj.vel,env.gravity,env.wind)
    return Projectile(position,velocity)

# COLOR
class Color:
    def __init__(self,r,g,b):
        self.r=r
        self.g=g
        self.b=b
    def create(r,g,b):
        return (r,g,b)

def sumc(*item:Color):
    x=0
    y=0
    z=0
    for i in item:
            x+=i.r
            y+=i.g
            z+=i.b
    return Color(x,y,z)

def subc(*item:Color):
    x=2*item[0].r
    y=2*item[0].g
    z=2*item[0].b
    for i in item:
            x-=i.r
            y-=i.g
            z-=i.b
    return Color(x,y,z)

def smulc(c:Color,scalar):
    return Color(np.round(scalar*c.r,6),np.round(scalar*c.g,6),np.round(scalar*c.b,6))

def hprodc(c1:Color,c2:Color):
    r = c1.r * c2.r
    g = c1.g * c2.g
    b = c1.b * c2.b
    return Color(r,g,b)

#CANVAS
class Canvas:
    def __init__(self,width,height):
        self.width=width
        self.height=height
        row = [Color.create(0,0,0) for i in range(self.width)]
        mat = [list(row) for i in range(self.height)]
        self.mat=mat

def write_pixel(c:Canvas,x,y,color:Color):
    r = int(np.round(255*color.r,0))
    g = int(np.round(255*color.g,0))
    b = int(np.round(255*color.b,0))

    if (r>255):
        r = 255
    elif (r<0):
        r = 0
    if (g>255):
        g = 255
    elif (g<0):
        g = 0
    if (b>255):
        b = 255
    elif (b<0):
        b = 0
    c.mat[y][x]=(r,g,b)

def pixel_at(c:Canvas,x,y):
    print(c.mat[y][x])

def canvas_to_ppm(c:Canvas,name):
    header = "P3\n"+str(c.width)+" "+str(c.height)+"\n255\n"
    body=""
    count = 0
    for i in range(len(c.mat)):
        for j in range(len(c.mat[i])):
            body+=str(c.mat[i][j][0])+" "+str(c.mat[i][j][1])+" "+str(c.mat[i][j][2])+" "
            count+=1
        if (count==70):
            body+="\n"

    with open(str(name)+'.ppm', 'wb') as f:
        f.write(bytearray(header+body+"\n", 'ascii'))

#MATRIX
A = np.array([[1,2,3,4],[5,6,7,8],[9,8,7,6],[5,4,3,2]])
B = np.array([[-2,1,2,3],[3,2,1,-1],[4,3,6,5],[1,2,7,8]])
C = np.array([[1,2,3,4],[2,4,4,2],[8,6,4,1],[0,0,0,1]])
D = (1,2,3,1)
identity_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

def inversem(M:np.array):
    return np.linalg.inv(M)

def matrix_multiply(A:np.array,B:np.array):
    M = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            M[i][j] = A[i,0]*B[0,j]+A[i,1]*B[1,j]+A[i,2]*B[2,j]+A[i,3]*B[3,j]
    return M

def matrix_tuple(A:np.array,B):
    R = np.zeros(4)
    for i in range(4):
        R[i] = A[i][0]*B[0]+A[i][1]*B[1]+A[i][2]*B[2]+A[i][3]*B[3]
    return tuple(R)

def translation(x,y,z):
    return np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]])

def scaling(x,y,z):
    return np.array([[x,0,0,0],[0,y,0,0],[0,0,z,0],[0,0,0,1]])

def rotation_x(r):
    return np.array([[1,0,0,0],[0,math.cos(r),-math.sin(r),0],[0,math.sin(r),math.cos(r),0],[0,0,0,1]])

def rotation_y(r):
    return np.array([[math.cos(r),0,math.sin(r),0],[0,1,0,0],[-math.sin(r),0,math.cos(r),0],[0,0,0,1]])

def rotation_z(r):
    return np.array([[math.cos(r),-math.sin(r),0,0],[math.sin(r),math.cos(r),0,0],[0,0,1,0],[0,0,0,1]])

def shearing(xy,xz,yx,yz,zx,zy):
    return np.array([[1,xy,xz,0],[yx,1,yz,0],[zx,zy,1,0],[0,0,0,1]])

def tupletoint(tuple):
    return(int(tuple[0]),int(tuple[1]),int(tuple[2]),int(tuple[3]))

# RAYS
class Ray:
    def __init__(self,origin:Point=None,direction:Vector=None):
        self.origin = origin if origin is not None else O
        self.direction = direction if origin is not None else norm(vector(1,1,1))

def raypos(ray:Ray,t):
    return sumt(ray.origin,smult(ray.direction,t))

class Shape:
    def __init__(self):
        self.transform = identity_matrix
        self.material = Material()
        self.saved_ray = Ray()
    def localIntersect(self,r0:Ray):
        return np.array([])
    def localNormalAt(self,point:Point):
        p2v = list(point)
        p2v[3] = 0
        return tuple(p2v)
# SPHERE 
nSphere = 0
class Sphere(Shape):
    def __init__(self,origin:Point=None,radius=None,nSphere=nSphere):
        super().__init__()
        self.origin = origin if origin is not None else O
        self.radius = radius if origin is not None else 1
        nSphere += 1
        self.id = nSphere if id is not None else nSphere
    def localIntersect(self,r0:Ray):
        
        r = transform_ray(r0,inversem(self.transform))
        self.saved_ray = r
        sphere_to_ray = subt(r.origin,self.origin)
        a = dot(r.direction,r.direction)
        b = 2*dot(r.direction,sphere_to_ray)
        c = dot(sphere_to_ray,sphere_to_ray)-1
        delta = b**2 - 4*a*c

        if (delta<0):
            return ()
        else:
            t1 = ((-b - np.sqrt(delta))/(2*a))
            t2 = ((-b + np.sqrt(delta))/(2*a))
            xs = np.array([])
            xs = np.append(xs,Intersection(t1,self))
            xs = np.append(xs,Intersection(t2,self))
        return xs
    def localNormalAt(self, wp: Point):
        #super().localNormalAt(wp)
        #op = matrix_tuple(inversem(self.transform),wp)
        op = wp
        on = subt(op,O)
        wn = matrix_tuple(np.transpose(inversem(self.transform)),on)
        wnl = list(wn)
        wnl[3] = 0
        return norm(tuple(wn))

def check_axis(origin,direction):
    tmin_numerator = -1-origin
    tmax_numerator = 1-origin
    if (np.abs(direction)>=EPSILON):
        tmin = tmin_numerator/direction
        tmax = tmax_numerator/direction
    else:
        tmin = tmin_numerator * OO
        tmax = tmax_numerator * OO
    
    if (tmin>tmax):
        temp = tmin
        tmin = tmax
        tmax = temp
    
    return [tmin,tmax]

class Cube(Shape):
    def __init__(self,origin:Point=None):
        super().__init__()
        self.origin = origin if origin is not None else O
    def localIntersect(self,r0:Ray):
        xtmin = check_axis(r0.origin[0],r0.direction[0])[0]
        xtmax = check_axis(r0.origin[0],r0.direction[0])[1]
        ytmin = check_axis(r0.origin[1],r0.direction[1])[0]
        ytmax = check_axis(r0.origin[1],r0.direction[1])[1]
        ztmin = check_axis(r0.origin[2],r0.direction[2])[0]
        ztmax = check_axis(r0.origin[2],r0.direction[2])[1]

        tmin = max(xtmin,ytmin,ztmin)
        tmax = min(xtmax,ytmax,ztmax)

        if (tmin>tmax):
            return np.array([])
        
        xs = np.array([])
        xs = np.append(xs,Intersection(tmin,self))
        xs = np.append(xs,Intersection(tmax,self))

        return xs

    def localNormalAt(self, wp: Point):
        maxc = max(np.abs(wp[0]),np.abs(wp[1]),np.abs(wp[2]))

        if (maxc==np.abs(wp[0])):
            return vector(wp[0],0,0)
        elif (maxc==np.abs(wp[1])):
            return vector(0,wp[1],0)
        return vector(0,0,wp[2])
    
def checkCap(ray:Ray,t):
    x = ray.origin[0] + t*ray.direction[0]
    z = ray.origin[2] + t*ray.direction[2]
    return ((x**2+z**2)<=1)
class Cylinder(Shape):
    def __init__(self,origin:Point=None,minimum=None,maximum=None,closed=None):
        super().__init__()
        self.origin = origin if origin is not None else O
        self.minimum = minimum if minimum is not None else -OO
        self.maximum = maximum if maximum is not None else OO
        self.closed = closed if closed is not None else False
    def localIntersect(self,r0:Ray):
        xs = np.array([])
        a = r0.direction[0]**2+r0.direction[2]**2
        if (np.abs(a)<EPSILON):
            self.intersect_caps(r0,xs)
            return xs
        b = 2*r0.origin[0]*r0.direction[0]+2*r0.origin[2]*r0.direction[2]
        c = r0.origin[0]**2+r0.origin[2]**2 - 1
        disc = b**2-4*a*c
        if (disc < 0): return np.array([])
        t0 = (-b-np.sqrt(disc))/(2*a)
        t1 = (-b+np.sqrt(disc))/(2*a)
        
        if (t0>t1):
            temp = t0
            t0 = t1
            t1 = temp

        
        y0 = r0.origin[1] + t0*r0.direction[1]
        if ((self.minimum<y0) and (y0<self.maximum)):
            xs = np.append(xs,Intersection(t0,self))
        
        y1 = r0.origin[1] + t1*r0.direction[1]
        if ((self.minimum<y1) and (y1<self.maximum)):
            xs = np.append(xs,Intersection(t1,self))

        self.intersect_caps(r0,xs)
        return xs

    def localNormalAt(self, wp: Point):
        dist = wp[0]**2+wp[2]**2
        if ((dist<1) and (wp[1]>=(self.maximum-EPSILON))):
            return vector(0,1,0)
        elif ((dist<1) and (wp[1]<=(self.minimum+EPSILON))):
            return vector(0,-1,0)
        else:
            return vector(wp[0],0,wp[2])
    
    def intersect_caps(self,ray:Ray,xs):
        if ((self.closed==False) or (np.abs(ray.direction[1])<EPSILON)):
            return
        t = (self.minimum - ray.origin[1])/ray.direction[1]
        if checkCap(ray,t):
            xs = np.append(xs,Intersection(t,self))
            

def intersect(s:Shape, r0:Ray):
    local_ray = r0 #transform_ray(r0,inversem(s.transform))
    return s.localIntersect(local_ray)
        
# INTERSECTION
class Intersection:
    def __init__(self,t,obj:Shape): #luego objetos en general
        self.t = t
        self.obj = obj

def intersections(*intrs:Intersection):
    ts = np.array([])
    for i in intrs:
        ts = np.append(ts,i)
    return ts

def hit(xs):
    #dtype = [('t',float),('obj',Sphere)]
    if (isinstance(xs,Intersection)):
        xs = np.array([xs])
    nonneg = np.array([float('inf'),0])
    for i in xs:
        if (i.t>=0):
            nonneg = np.vstack([nonneg,[(i.t),(i.obj)]])
    if (len(nonneg)>2):
        nonneg = nonneg[nonneg[:,0].argsort()]  
        return Intersection(nonneg[0][0],nonneg[0][1])
    else:
        return [0,Sphere(O,0,nSphere)]
    
# RAY TRANSFORMATION
def transform_ray(r:Ray,M:np.array):
    newOrigin = matrix_tuple(M,r.origin)
    newDirection = matrix_tuple(M,r.direction)
    return Ray(newOrigin,newDirection)

def set_transform(shape:Shape,transform:np.array):
    shape.transform = transform

# SHADING
def normal_at(s:Shape, wp:Point):
    local_point = matrix_tuple(inversem(s.transform),wp)
    local_normal = s.localNormalAt(local_point)
    world_normal0 = list(matrix_tuple(np.transpose(inversem(s.transform)),local_normal))
    #world_normal0 = list(matrix_tuple(s.transform,local_normal))
    world_normal0[3] = 0
    world_normal = tuple(world_normal0)
    return norm(world_normal)

def reflect(v:Vector,n:Vector):
    return subt(v,smult(n,2*dot(v,n)))

def color2tuple(c:Color):
    return (c.r,c.g,c.b)

def tuple2color(a):
    return Color(a[0],a[1],a[2])

class PointLight:
    def __init__(self,position:Point,intensity:Color):
        self.position = position
        self.intensity = intensity

class Material:
    def __init__(
    self,
    pattern=None,
    color:Color=None,
    ambient=None,
    diffuse=None,
    specular=None,
    shininess=None,
    reflective=None,
    transparency=None,
    refractive_index=None):
        self.pattern = pattern if pattern is not None else None
        self.color = color if color is not None else Color(1,1,1)
        self.ambient = ambient if ambient is not None else 0.1
        self.diffuse = diffuse if diffuse is not None else 0.9
        self.specular = specular if specular is not None else 0.9
        self.shininess = shininess if shininess is not None else 200.0
        self.reflective = reflective if reflective is not None else 0.0
        self.transparency = transparency if transparency is not None else 0.0
        self.refractive_index = refractive_index if refractive_index is not None else 1.0

def lightning(material:Material,object:Shape,light:PointLight,point:Point,eyev:Vector,normalv:Vector,in_shadow:bool):
    if (material.pattern is not None):
        color = apply_pattern_at_object(material.pattern,object,point)
    else: color = material.color

    effective_color = hprodc(color,light.intensity)
    lightv = norm(subt(light.position,point))
    ambient = smulc(effective_color,material.ambient)
    light_dot_normal = dot(lightv,normalv)
    if (light_dot_normal<0):
        diffuse = Color(0,0,0)
        specular = Color(0,0,0)
    else:
        diffuse = smulc(effective_color,material.diffuse*light_dot_normal)
        reflectv = reflect(negt(lightv),normalv)
        reflect_dot_eye = dot(reflectv,eyev)
        if (reflect_dot_eye<=0):
            specular = Color(0,0,0)
        else:
            factor = np.power(reflect_dot_eye,material.shininess)
            specular = smulc(light.intensity,material.specular*factor)
    if (in_shadow):
        return ambient
    else:
        return sumc(ambient,diffuse,specular)

# WORLD
class World:
    def __init__(self,objs=None,lsource:PointLight=None):
        self.objs = objs if objs is not None else None
        self.lsource = lsource if objs is not None else None

# Default World
def default_world():
    light = PointLight(point(-10,10,-10),Color(1,1,1))
    s1 = Sphere()
    s1.material.color = Color(0.8,1.0,0.6)
    s1.material.diffuse = 0.7
    s1.material.specular = 0.2

    s2 = Sphere()
    set_transform(s2,scaling(0.5,0.5,0.5))

    objs = np.array([])
    objs = np.append(objs,s1)
    objs = np.append(objs,s2)
    return World(objs,light)

def intersect_world(w:World,r:Ray):
    xs = []
    for i in w.objs:
        arr = intersect(i,r)
        for j in arr:
            xs.append(j)
    
    xs.sort(key=lambda x: x.t)
    return xs

class Computation:
    def __init__(self,t,obj:Shape,point:Point,eyev:Vector,normalv:Vector,reflectv:Vector,inside:bool) -> None:
        self.t = t
        self.obj = obj
        self.point = point
        self.eyev = eyev
        self.normalv = normalv
        self.reflectv = reflectv
        self.inside = inside
        self.over_point = O
        self.under_point = O
        self.n1 = 0.5
        self.n2 = 0.5
"""
def prepare_computations(i:Intersection,r:Ray):
    comp = Computation(i.t,i.obj,raypos(r,i.t),negt(r.direction),normal_at(i.obj,raypos(r,i.t)),normal_at(i.obj,raypos(r,i.t)),False)
    if (dot(comp.normalv,comp.eyev)<0):
        comp.inside = True
        comp.normalv = negt(comp.normalv)
        comp.reflectv = reflect(r.direction,comp.normalv)
        comp.over_point = sumt(comp.point,smult(comp.normalv,EPSILON))
    else:
        comp.inside = False
        comp.over_point = sumt(comp.point,smult(comp.normalv,EPSILON))
    return comp
"""
def prepare_computations(i,r:Ray):
    if (isinstance(i,Intersection)):
        comp = Computation(i.t,i.obj,raypos(r,i.t),negt(r.direction),normal_at(i.obj,raypos(r,i.t)),normal_at(i.obj,raypos(r,i.t)),False)
        containers = np.array([])
        if (i == hit(i)):
            if (not np.any(containers)):
                  comp.n1 = 1.0
            else:
                comp.n1 = containers[-1].material.refractive_index
        if (i.obj in containers):
            containers = np.setdiff1d(containers,np.array([i.obj]))
        else:
            containers = np.append(containers,i.obj)
        if (i == hit(i)):
            if (not np.any(containers)):
                comp.n2 = 1.0
            else:
                comp.n2 = containers[-1].material.refractive_index
        comp.point = raypos(r,comp.t)
        if (dot(comp.normalv,comp.eyev)<0):
            comp.inside = True
            comp.normalv = negt(comp.normalv)
            comp.reflectv = reflect(r.direction,comp.normalv)
            comp.over_point = sumt(comp.point,smult(comp.normalv,EPSILON))
            comp.under_point= subt(comp.point,smult(comp.normalv,EPSILON))
        else:
            comp.inside = False
            comp.over_point = sumt(comp.point,smult(comp.normalv,EPSILON))
            comp.under_point= subt(comp.point,smult(comp.normalv,EPSILON))
        return comp
    else:
        containers = np.array()
        for x in i:
            comp = Computation(x.t,x.obj,raypos(r,x.t),negt(r.direction),normal_at(x.obj,raypos(r,x.t)),normal_at(x.obj,raypos(r,x.t)),False)
            if (x == hit(i)):
                if (not np.any(i)):
                    comp.n1 = 1.0
                else:
                    comp.n1 = containers[-1].material.refractive_index
            if (x.obj in containers):
                containers = np.setdiff1d(containers,np.array([x.obj]))
            else:
                containers = np.append(containers,x.obj)
            if (x == hit(i)):
                if (not np.any(i)):
                    comp.n2 = 1.0
                else:
                    comp.n2 = containers[-1].material.refractive_index
                break


def is_shadowed(world:World, point:Point):
    v = subt(world.lsource.position,point)
    distance = magn(v)
    direction = norm(v)
    r = Ray(point,direction)
    xs = intersect_world(world,r)
    h = hit(xs)
    if (not isinstance(h,list)):
        if (h.t<distance):
            return True
        else:
            return False
    else:
        return False

def shade_hit(world:World, comps:Computation):
    shadowed = is_shadowed(world,comps.over_point)
    surface = lightning(comps.obj.material,comps.obj,world.lsource,comps.point,comps.eyev,comps.normalv,shadowed)
    reflected = reflected_color(world,comps)
    refracted = refracted_color(world,comps,5)
    return sumc(surface,reflected,refracted)

def color_at(w:World, r:Ray):
    xs = intersect_world(w,r)
    hits = hit(xs)
    if isinstance(hits,list):
        return Color(0,0,0)
    else:
        comps = prepare_computations(hits,r)
        return shade_hit(w,comps)

def view_transform(_from:Point, _to:Point, _up:Vector):
    forward = norm(subt(_to,_from))
    left = cross(forward, norm(_up))
    true_up = cross(left, forward)

    orientation = np.array([[left[0],left[1],left[2],0],[true_up[0],true_up[1],true_up[2],0],[-forward[0],-forward[1],-forward[2],0],[0,0,0,1]])
    return matrix_multiply(orientation,translation(-_from[0],-_from[1],-_from[2]))

# CAMERA
class Camera:
    def __init__(self,hsz:int,vsz:int,fov:float,transform=None) -> None:
        self.hsz = hsz
        self.vsz = vsz
        self.fov = fov
        self.transform = transform if transform is not None else identity_matrix
        
        half_view = np.tan(self.fov/2)
        aspect = self.hsz / self.vsz
        if (aspect >= 1):
            self.hw = half_view
            self.hh = half_view / aspect
        else:
            self.hw = half_view * aspect
            self.hh = half_view
        
        self.pixelsize = (self.hw*2)/hsz

def rayforpixel(camera:Camera, px:int, py:int):
    xoff = (px + 0.5) * camera.pixelsize
    yoff = (py + 0.5) * camera.pixelsize

    wx = camera.hw - xoff
    wy = camera.hh - yoff

    pixel = matrix_tuple(inversem(camera.transform),point(wx,wy,-1))
    origin = matrix_tuple(inversem(camera.transform),point(0,0,0))
    direction = norm(subt(pixel,origin))

    return Ray(origin,direction)

def render(c: Camera, w: World):
    image = Canvas(c.hsz,c.vsz)

    for y in range(c.vsz):
        for x in range(c.hsz):
            ray = rayforpixel(c,x,y)
            color = color_at(w,ray)
            write_pixel(image,x,y,color)
    
    return image

# PLANE
class Plane(Shape):
    def __init__(self):
        super().__init__()
    def localNormalAt(self, point: Point):
        return vector(0,1,0)
    def localIntersect(self, r0: Ray):
        r = transform_ray(r0,inversem(self.transform))
        self.saved_ray = r
        #r = r0
        if (abs(r.direction[1])<EPSILON):
            return ()
        t = -r.origin[1]/r.direction[1]
        xs = np.array([])
        xs = np.append(xs,Intersection(t,self))
        return xs

class Pattern:
    def __init__(self,transform=None) -> None:
        self.transform = transform if transform is not None else identity_matrix
    def applypattern(self,p:Point):
        return Color(p[0],p[1],p[2])

class StripePattern(Pattern):
    def __init__(self, a: Color, b: Color, transform=None) -> None:
        super().__init__(transform)
        self.a = a
        self.b = b
    def applypattern(self,p:Point):
        if (np.floor(p[0])%2==0): return self.a
        else: return self.b

class GradientPattern(Pattern):
    def __init__(self, a: Color, b: Color, transform=None) -> None:
        super().__init__(transform)
        self.a = a
        self.b = b
    def applypattern(self, p: Point):
        distance = subc(self.b,self.a)
        fraction = p[0]-np.floor(p[0])
        return sumc(self.a,smulc(distance,fraction))

class RingPattern(Pattern):
    def __init__(self, a: Color, b: Color, transform=None) -> None:
        super().__init__(transform)
        self.a = a
        self.b = b
    def applypattern(self, p: Point):
        if (np.floor(np.sqrt((p[0]**2)+(p[2]**2)))%2==0): return self.a
        else: return self.b

class CheckersPattern(Pattern):
    def __init__(self, a: Color, b: Color, transform=None) -> None:
        super().__init__(transform)
        self.a = a
        self.b = b
    def applypattern(self, p: Point):
        if ((np.floor(np.absolute(p[0]))+np.floor(np.absolute(p[1]))+np.floor(np.absolute(p[2])))%2==0): return self.a
        else: return self.b

class RadialGradientPattern(Pattern):
    def __init__(self, a: Color, b: Color, transform=None) -> None:
        super().__init__(transform)
        self.a = a
        self.b = b
    def applypattern(self, p: Point):
        r = np.sqrt((p[0]**2)+(p[2]**2))
        distance = subc(self.b,self.a)
        fraction = r-np.floor(r)
        grad1 = sumc(self.a,smulc(distance,fraction))
        distance2 = subc(self.a,self.b)
        fraction2 = r-np.floor(r)
        grad2 = sumc(self.a,smulc(distance2,fraction2))
        if (np.floor(r)%2==0): return grad1
        else: return grad1

class SolidPattern(Pattern):
    def __init__(self, color: Color, transform=None) -> None:
        super().__init__(transform)
        self.color = color
    def applypattern(self, p: Point):
        return self.color

def apply_pattern_at_object(pattern:Pattern,object:Shape,wp:Point):
    op = matrix_tuple(inversem(object.transform),wp)
    pp = matrix_tuple(inversem(pattern.transform),op)
    return pattern.applypattern(pp)

def interpolate(a0,a1,w):
    return (a1-a0)*w + a0

class vector2:
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y

def randomGradient(ix:int, iy:int):
    w:int = 64
    s:int = w // 2
    a:int = ix
    b:int = iy
    a *= 3284157443
    b ^= a << s | a >> w-s
    b *= 1911520717
    a ^= b << s | b >> w-s
    a *= 2048419325
    random = a * 3.14159265
    v = vector2(np.cos(random),np.sin(random))
    return v

def dotGridGradient(ix:int,iy:int,x:float,y:float):
    gradient = randomGradient(ix,iy)
    dx = x - ix
    dy = y - iy
    return (dx*gradient.x+dy*gradient.y)

def PerlinNoise(x:float, y:float):
    x0: int = int(np.floor(x))
    x1: int = x0 + 1
    y0: int = int(np.floor(y))
    y1: int = y0 + 1

    sx: float = x - float(x0)
    sy: float = y - float(y0)

    n0 = dotGridGradient(x0,y0,x,y)
    n1 = dotGridGradient(x1,y0,x,y)

    ix0 = interpolate(n0,n1,sx)
    
    n0 = dotGridGradient(x0,y1,x,y)
    n1 = dotGridGradient(x1,y1,x,y)

    ix1 = interpolate(n0,n1,sx)

    value = interpolate(ix0,ix1,sy)
    return value

class PerlinNoisifyPattern(Pattern):
    def __init__(self, pattern:Pattern, transform=None) -> None:
        super().__init__(transform)
        self.pattern = Pattern
    def applypattern(self, p: Point):
        p2 = point(p[0]+PerlinNoise(p[0],p[1]),p[1]+PerlinNoise(p[2],p[1]),p[2]+PerlinNoise(p[0],p[1]))
        return self.pattern.applypattern(self.pattern,p2)

def reflected_color(w:World,comps:Computation):
    if (comps.obj.material.reflective == 0.0):
        return Color(0,0,0)
    reflect_ray = Ray(comps.over_point,comps.reflectv)
    color = color_at(w,reflect_ray)
    return smulc(color,comps.obj.material.reflective)

def glass_sphere():
    s = Sphere()
    s.transform = identity_matrix
    s.material.transparency = 1.0
    s.material.refractive_index = 1.5
    return s

def glass_cube():
    c = Cube()
    c.transform = identity_matrix
    c.material.transparency = 1.0
    c.material.refractive_index = 1.5
    return c

def glass_cylinder():
    c = Cylinder(O,0,2,True)
    c.transform = identity_matrix
    c.material.transparency = 0
    c.material.refractive_index = 1
    return c

def refracted_color(world:World,comps:Computation,remaining):
    if (comps.obj.material.transparency == 0):
        return Color(0,0,0)
    n_ratio = comps.n1/comps.n2
    cos_i = dot(comps.eyev,comps.normalv)
    sin2_t = n_ratio**2*(1-cos_i**2)
    if (sin2_t>1):
        return Color(0,0,0)
    cos_t = sqrt(1.0-sin2_t)
    direction = subt(smult(comps.normalv,(n_ratio*cos_i*cos_t)),smult(comps.eyev,n_ratio))
    refracted_ray = Ray(comps.under_point,direction)
    color = smulc(color_at(world,refracted_ray),comps.obj.material.transparency)
    return color

def world_1():
    floor = Plane()
    floor.material = Material()
    floor.material.color = Color(1,0.9,0.9)
    floor.material.specular = 0
    floor.material.pattern = CheckersPattern(Color(0,0,0),Color(1,1,1))
    floor.material.reflective = 0.7

    left_wall = Plane()
    left_wall.transform = matrix_multiply(matrix_multiply(matrix_multiply(translation(0,0,5),rotation_y(-np.pi/4)),rotation_x(np.pi/2)),identity_matrix)
    left_wall.material = Material()

    right_wall = Plane()
    right_wall.transform = matrix_multiply(matrix_multiply(matrix_multiply(translation(0,0,5),rotation_y(np.pi/4)),rotation_x(np.pi/2)),identity_matrix)
    right_wall.material = Material()

    middle = Sphere()
    middle.transform = translation(-0.5,1,0.5)
    middle.material = Material()
    middle.material.color = Color(0.1,1,0.5)
    middle.material.diffuse = 0.7
    middle.material.specular = 0.3
    middle.material.pattern = CheckersPattern(Color(0,0,1),Color(1,0,1),scaling(0.5,0.5,0.5))


    right = Sphere()
    right.transform = matrix_multiply(translation(1.5,0.5,-0.5),scaling(0.5,0.5,0.5))
    right.material = Material()
    right.material.color = Color(0.1,1,0.5)
    right.material.diffuse = 0.7
    right.material.specular = 0.3
    right.material.pattern = middle.material.pattern

    left = Sphere()
    left.transform = matrix_multiply(translation(-1.5,0.33,-0.75),scaling(0.33,0.33,0.33))
    left.material = Material()
    left.material.color = Color(1,0.8,0.1)
    left.material.diffuse = 0.7
    left.material.specular = 0.3
    left.material.pattern = middle.material.pattern

    objs = np.array([])
    objs = np.append(objs,floor)
    objs = np.append(objs,left_wall)
    objs = np.append(objs,right_wall)
    objs = np.append(objs,left)
    objs = np.append(objs,middle)
    objs = np.append(objs,right)

    w = World(objs)

    w.lsource = PointLight(point(-10,10,-10),Color(1,1,1))

    camera = Camera(100,50,np.pi/3)
    camera.transform = view_transform(point(0,4.5,-5),point(0,1,0),vector(0,1,0))
    canvas = render(camera,w)
    canvas_to_ppm(canvas,1)

floor = Plane()
floor.material = Material()
floor.material.color = Color(1,0.9,0.9)
floor.material.specular = 0
floor.material.pattern = CheckersPattern(Color(0,0,0),Color(1,1,1))
floor.material.reflective = 0.0

left_wall = Plane()
left_wall.transform = matrix_multiply(matrix_multiply(matrix_multiply(translation(0,0,5),rotation_y(-np.pi/4)),rotation_x(np.pi/2)),identity_matrix)
left_wall.material = Material()
left_wall.material.pattern = RingPattern(Color(1,0,0),Color(1,0,1))

right_wall = Plane()
right_wall.transform = matrix_multiply(matrix_multiply(matrix_multiply(translation(0,0,5),rotation_y(np.pi/4)),rotation_x(np.pi/2)),identity_matrix)
right_wall.material = Material()
right_wall.material.pattern = StripePattern(Color(1,1,0),Color(0,1,0))

A = glass_cylinder()
A.transform = scaling(2,2,2)
A.material.refractive_index = 1.5
#A.material.color = Color(0.7,0,0.7)

B = glass_sphere()
B.transform = translation(0,0,-0.25)
B.material.refractive_index = 2.0

C = glass_sphere()
C.transform = translation(0,0,0.25)
C.material.refractive_index = 2.5

objs = np.array([])
objs = np.append(objs,floor)
objs = np.append(objs,left_wall)
objs = np.append(objs,right_wall)
objs = np.append(objs,A)
#objs = np.append(objs,B)
#objs = np.append(objs,C)


w = World(objs)
w.lsource = PointLight(point(-10,15,-10),Color(1,1,1))    
camera = Camera(200,100,np.pi/3)
camera.transform = view_transform(point(2,4.5,-4),point(0,1,0),vector(0,1,0))
canvas = render(camera,w)
canvas_to_ppm(canvas,1)
