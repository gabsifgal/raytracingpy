#include "objects.h"
#include <math.h>
#include "vector"


void Tuple::Vector(double _x, double _y, double _z){
    x = _x; y = _y; z = _z; w = 0;
}

void Tuple::Point(double _x, double _y, double _z){
    x = _x; y = _y; z = _z; w = 1;
}

// Compares two numbers to see if they're equal,
// within a margin of error of EPSILON (defined in line 4)
bool equal(double a, double b){
    return (abs(a-b)<EPSILON);
}

Tuple CVector(double x, double y, double z){
    Tuple r; r.Vector(x,y,z);
    return r;
}

Tuple CPoint(double x, double y, double z){
    Tuple r; r.Point(x,y,z);
    return r;
}

// Sums two 4-element arrays, item by item 
Tuple sumt(Tuple a, Tuple b){
    Tuple r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    r.z = a.z + b.z;
    r.w = a.w + b.w;
    return r;
}

// Substracts two 4-element arrays, item by item
Tuple subt(Tuple a, Tuple b){
    Tuple r;
    r.x = a.x - b.x;
    r.y = a.y - b.y;
    r.z = a.z - b.z;
    r.w = a.w - b.w;
    return r;
}

// Negates all items in a 4-element array
Tuple negate(Tuple arr){
    Tuple r;
    r.x = -arr.x;
    r.y = -arr.y;
    r.z = -arr.z;
    r.w = -arr.w;
    return r;
}

// Multiplies a 4-element array times a scalar
Tuple smul(Tuple arr, double scalar){
    Tuple r;
    r.x = arr.x*scalar;
    r.y = arr.y*scalar;
    r.z = arr.z*scalar;
    r.w = arr.w*scalar;
    return r;
}

// Divides a 4-element array by a scalar
Tuple sdiv(Tuple arr, double scalar){
    Tuple r;
    r.x = arr.x/scalar;
    r.y = arr.y/scalar;
    r.z = arr.z/scalar;
    r.w = arr.w/scalar;
    return r;
}

// Returns the magnitude of a vector
double magnitude(Tuple arr){
    return sqrt(pow(arr.x,2)+pow(arr.y,2)+pow(arr.z,2)+pow(arr.w,2));
}

// Normalizes a vector
Tuple normalize(Tuple arr){
    double m = magnitude(arr);
    if (m == 0){
        return arr;
    }
    else{
        return sdiv(arr,m);
    }
}

// Returns the dot product of two 4-element arrays
double dotproduct(Tuple a, Tuple b){
    return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w);
}

// Returns the cross product of two vectors as a vector
Tuple crossproduct(Tuple a, Tuple b){
    Tuple r;
    r.x = (a.y*b.z)-(a.z*b.y);
    r.y = (a.z*b.x)-(a.x*b.z);
    r.z = (a.x*b.y)-(a.y*b.x);
    r.w = 0;
}

void Projectile::SetPV(Tuple _pos, Tuple _vel){
    pos = _pos;
    vel = _vel;
}

void Environment::SetGW(Tuple _gravity, Tuple _wind){
    gravity = _gravity;
    wind = _wind;
}

Projectile tick(Environment env, Projectile proj){
    Projectile r;
    r.SetPV(sumt(proj.pos,proj.vel),
            sumt(sumt(proj.vel,env.gravity),env.wind));
    return r;
}

Color sumc(Color a, Color b){
    Color c;
    c.r = a.r + b.r; 
    c.g = a.g + b.g; 
    c.b = a.b + b.b;
    return c; 
}

Color subc(Color a, Color b){
    Color c;
    c.r = a.r - b.r; 
    c.g = a.g - b.g; 
    c.b = a.b - b.b;
    return c; 
}

Color smulc(Color c, double s){
    Color r;
    r.r = c.r * s;
    r.g = c.g * s;
    r.b = c.b * s;
    return r;
}

Color hprodc(Color a, Color b){
    Color r;
    r.r = a.r * b.r;
    r.g = a.g * b.g;
    r.b = a.b * b.b;
    return r;
}

void Canvas::init(int _width, int _height){
    width = _width;
    height = _height;
    std::vector<std::vector<Color>> m(height, std::vector<Color>(width));

    for (int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            m[i][j] = Color(0,0,0);
        }
    }
    mat = m;
}

void Canvas::write_pixel(int x, int y, Color color){
    int r = (int)round(255*color.r);
    int g = (int)round(255*color.g);
    int b = (int)round(255*color.b);

    Color c1; c1.r = r; c1.g = g; c1.b = b;
    if (r > 255){r = 255;} else if (r < 0){r = 0;}
    if (g > 255){g = 255;} else if (g < 0){g = 0;}
    if (b > 255){b = 255;} else if (b < 0){b = 0;}

    mat[y][x] = c1;
}

void Canvas::pixel_at(int x, int y){
    printf("(%lf, %lf, %lf)\n",
            mat[y][x].r,
            mat[y][x].g,
            mat[y][x].b);
}

void Canvas::canvas_to_ppm(std::string name){
    /*std::string header = "P3\n"+std::to_string(width)+" "+std::to_string(height)+"\n255\n";
    std::string body = "";
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            body+=std::to_string((int)mat[i][j].r)+" "+
                  std::to_string((int)mat[i][j].g)+" "+
                  std::to_string((int)mat[i][j].b)+" ";
        }
        body+="\n";
    }
    std::ofstream f(name+".ppm");
    f << header+body+"\n";
    f.close();*/
    std::string header = "P3\n"+std::to_string(width)+" "+std::to_string(height)+"\n255\n";
    std::string body = "";
    int count;
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            body+=std::to_string(((int)mat[i][j].r))+" "+
                  std::to_string(((int)mat[i][j].g))+" "+
                  std::to_string(((int)mat[i][j].b))+" ";
            count++;
        }
        if (count==70){
            body+="\n";
        }
    }
    std::ofstream f(name+".ppm");
    f << header+body+"\n";
    f.close();
}

double determinant(std::vector<std::vector<double>> M){
    if (M.size() == 2){
        return (M[0][0]*M[1][1]-M[0][1]*M[1][0]);
    }
    else{
        double det;
        for (int i = 0; i < M.size(); i++){
            det += M[0][i] * cofactor(M,0,i);
        }
        return det;
    }
    
}

std::vector<std::vector<double>> submatrix(std::vector<std::vector<double>> M, int row, int col){
    for (int i = 0; i < M.size(); i++){
        M[i].erase(M[i].begin() + col);
    }
    M.erase(M.begin() + row);
    return M;
}

double minor(std::vector<std::vector<double>> M, int row, int col){
    std::vector<std::vector<double>> N = submatrix(M,row,col);
    return determinant(N);
}

double cofactor(std::vector<std::vector<double>> M, int row, int col){
    if ((row+col)%2){
        return -minor(M,row,col);
    }
    else{
        return minor(M,row,col);
    }
}

std::vector<std::vector<double>> inversem(std::vector<std::vector<double>> M){
    std::vector<std::vector<double>> M2(M.size(),std::vector<double>(M.size()));
    double c;
    for (int i = 0; i < M.size(); i++){
        for (int j = 0; j < M.size(); j++){
            c = cofactor(M, i, j);
            M2[j][i] = c / determinant(M);
        }
    }
    return M2;
}
std::vector<std::vector<double>> matrix_multiply(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B){
    std::vector<std::vector<double>> R(A.size(),std::vector<double>(A.size()));
    for (int i = 0; i < A.size(); i++){
        for (int j = 0; j < A.size(); j++){
            R[i][j] = A[i][0]*B[0][j]+A[i][1]*B[1][j]+A[i][2]*B[2][j]+A[i][3]*B[3][j];
        }
    }
    return R;
}
Tuple matrix_tuple(std::vector<std::vector<double>> A, Tuple B){
    Tuple r; r.Vector(0,0,0);
    r.x = A[0][0]*B.x+A[0][1]*B.y+A[0][2]*B.z+A[0][3]*B.w;
    r.y = A[1][0]*B.x+A[1][1]*B.y+A[1][2]*B.z+A[1][3]*B.w;
    r.z = A[2][0]*B.x+A[2][1]*B.y+A[2][2]*B.z+A[2][3]*B.w;
    r.w = A[3][0]*B.x+A[3][1]*B.y+A[3][2]*B.z+A[3][3]*B.w;
    return r;
}
std::vector<std::vector<double>> translation(double x, double y, double z){
    std::vector<std::vector<double>> R = {{1,0,0,x},
                                {0,1,0,y},
                                {0,0,1,z},
                                {0,0,0,1}};
    return R;
}
std::vector<std::vector<double>> scaling(double x, double y, double z){
    std::vector<std::vector<double>> R = {{x,0,0,0},
                                {0,y,0,0},
                                {0,0,z,0},
                                {0,0,0,1}};
    return R;
}
std::vector<std::vector<double>> rotx(double r){
    std::vector<std::vector<double>> R = {{1,0,0,0},
                                {0,cos(r),-sin(r),0},
                                {0,sin(r),cos(r),0},
                                {0,0,0,1}};
    return R;
}
std::vector<std::vector<double>> roty(double r){
    std::vector<std::vector<double>> R = {{cos(r),0,sin(r),0},
                                {0,1,0,0},
                                {-sin(r),0,cos(r),0},
                                {0,0,0,1}};
    return R;
}
std::vector<std::vector<double>> rotz(double r){
    std::vector<std::vector<double>> R = {{cos(r),-sin(r),0,0},
                                {sin(r),cos(r),0,0},
                                {0,0,1,0},
                                {0,0,0,1}};
    return R;
}
std::vector<std::vector<double>> shearing(double xy, double xz, double yx, double yz, double zx, double zy){
    std::vector<std::vector<double>> R = {{1,xy,xz,0},
                                {yx,1,yz,0},
                                {zx,zy,1,0},
                                {0,0,0,1}};
    return R;
}

std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> b){
    if (b.size() == 0){
        std::vector<std::vector<double>> a = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
        return a;
    }
    std::vector<std::vector<double>> trans_vec(b[0].size(), std::vector<double>());

    for (int i = 0; i < b.size(); i++)
    {
        for (int j = 0; j < b[i].size(); j++)
        {
            trans_vec[j].push_back(b[i][j]);
        }
    }

    return trans_vec;
}

Tuple Ray::raypos(double t){
    return sumt(origin,smul(direction,t));
}

Ray transform_ray(Ray r, std::vector<std::vector<double>> M){
    Tuple newOrigin = matrix_tuple(M,r.origin);
    Tuple newDirection = matrix_tuple(M,r.direction);
    Ray r1; r1.origin = newOrigin; r1.direction = newDirection;
    return r1;
}
std::vector<Intersection> intersect(Sphere s, Ray r0){
    Ray r = transform_ray(r0,inversem(s.transform));
    Tuple sphere_to_ray = subt(r.origin,s.origin);
    double a = dotproduct(r.direction,r.direction);
    double b = 2*dotproduct(r.direction,sphere_to_ray);
    double c = dotproduct(sphere_to_ray,sphere_to_ray)-1;
    double delta = pow(b,2) - 4*a*c;

    if (delta < 0){
        Sphere snull; snull.origin = CPoint(0,0,0);
        snull.radius = 0;
        Intersection ix; ix.t = 13371337; ix.obj = snull;
        std::vector<Intersection> temp = {ix,ix};
        return temp;
    }
    else{
        double t1 = (-b - sqrt(delta))/(2*a);
        double t2 = (-b + sqrt(delta))/(2*a);
        Intersection i1, i2;
        i1.t = t1; i2.t = t2;
        i1.obj = s; i2.obj = s;
        std::vector<Intersection> temp = {i1,i2};
        return temp;
    }
}

Intersection hit(std::vector<Intersection> xs){
    std::vector<Intersection> nonneg;
    for (int i = 0; i < xs.size(); i++){
        //if (xs[i].t >= 0){
            nonneg.push_back(xs[i]);
       // }
    }
    if (nonneg.size() > 0){
        std::sort(nonneg.begin(),nonneg.end());
        return nonneg[0];
    }
    else{
        Sphere snull; snull.origin = CPoint(0,0,0);
        snull.radius = 0;
        Intersection ix; ix.t = 13371337; ix.obj = snull;
        return ix; 
    }
}

Tuple normal_at(Sphere s, Tuple wp){
    Tuple op = matrix_tuple(inversem(s.transform),wp);
    Tuple on = subt(op, CPoint(0,0,0));
    Tuple wn = matrix_tuple(transpose(inversem(s.transform)),on);
    wn.w = 0;
    return normalize(wn);
}

Tuple reflect(Tuple v, Tuple n){
    return subt(v,smul(n,2*dotproduct(v,n)));
}
Color lightning(Material material, PointLight light, Tuple point, Tuple eyev, Tuple normalv){
    Color effective_color = hprodc(material.color, light.intensity);
    Tuple lightv = normalize(subt(light.position, point));
    Color ambient = smulc(effective_color,material.ambient);
    Color diffuse = Color(0,0,0);
    Color specular = Color(0,0,0);
    double light_dot_normal = dotproduct(lightv,normalv);
    if (light_dot_normal < 0){
        diffuse = Color(0,0,0);
        specular = Color(0,0,0);
    }
    else{
        diffuse = smulc(effective_color,material.diffuse*light_dot_normal);
        Tuple reflectv = reflect(negate(lightv),normalv);
        double reflect_dot_eye = dotproduct(reflectv,eyev);
        if (reflect_dot_eye <= 0){
            specular = Color(0,0,0);
        } 
        else{
            double factor = pow(reflect_dot_eye,material.shininess);
            specular = smulc(light.intensity,material.specular*factor);
        }
    }
    return sumc(ambient,diffuse);
}

World default_world(void){
    PointLight light = PointLight(CPoint(-10,10,-10),Color(1,1,1));
    Sphere s1 = Sphere();
    s1.material.color = Color(0.8,1.0,0.6);
    s1.material.diffuse = 0.7;
    s1.material.specular = 0.2;

    Sphere s2 = Sphere();
    s2.transform = scaling(0.5,0.5,0.5);

    std::vector<Sphere> objs = {s1,s2};
    return World(objs,light);
}

std::vector<Intersection> intersect_world(World w, Ray r){
    std::vector<Intersection> xs;
    for (int i = 0; i < w.objs.size(); i++){
        std::vector<Intersection> arr = intersect(w.objs[i],r);
        for (int j = 0; j < arr.size(); j++){
            xs.push_back(arr[j]);
        }
    }
    std::sort(xs.begin(),xs.end());
    return xs;
}

Computation prepComp(Intersection i, Ray r){
    Computation comp = Computation(
                        i.t,
                        i.obj,
                        r.raypos(i.t),
                        negate(r.direction),
                        normal_at(i.obj,
                        r.raypos(i.t)),
                        false);
    if (dotproduct(comp.normalv,comp.eyev)<0){
        comp.inside = true;
        comp.normalv = negate(comp.normalv);
    } else{
        comp.inside = false;
    }
    return comp;
}

Color shade_hit(World world, Computation comps){
    return lightning(comps.obj.material,
                    world.lsource,
                    comps.point,
                    comps.eyev,
                    comps.normalv);
}
Color color_at(World w, Ray r){
    std::vector<Intersection> xs = intersect_world(w,r);
    Intersection hits = hit(xs);
    if (hits.t==13371337){
        return Color(0,0,0);
    }
    else{
        Computation comps = prepComp(hits,r);
        return shade_hit(w,comps);
    }
}
std::vector<std::vector<double>> view_transform(Tuple _from, Tuple _to, Tuple _up){
    Tuple forward = normalize(subt(_to,_from));
    Tuple left = crossproduct(forward, normalize(_up));
    Tuple true_up = crossproduct(left,forward);
    std::vector<std::vector<double>> orientation = {
        {left.x,left.y,left.z,0},
        {true_up.x,true_up.y,true_up.z,0},
        {-forward.x,-forward.y,-forward.z,0},
        {0,0,0,1}
    };
    return matrix_multiply(orientation,translation(-_from.x,-_from.y,-_from.z));
}

void Camera::init(int h, int v, double f, std::vector<std::vector<double>> t){
    hsz = h;
    vsz = v;
    fov = f;
    transform = t;
    double half_view = tan(fov/2);
    double aspect = hsz / vsz;
    if (aspect>=1){
        hw = half_view;
        hh = half_view / aspect;
    } else{
        hw = half_view * aspect;
        hh = half_view;
    }
    pixelsize = (hw*2)/hsz;
}

Ray rayforpixel(Camera camera, int px, int py){
    double wx = camera.hw - (px + 0.5) * camera.pixelsize;
    double wy = camera.hh - (py + 0.5) * camera.pixelsize;
    Ray r; 
    r.origin = matrix_tuple(inversem(camera.transform),CPoint(0,0,0)); 
    r.direction = normalize(subt(matrix_tuple(inversem(camera.transform),CPoint(wx,wy,-1)),matrix_tuple(inversem(camera.transform),CPoint(0,0,0))));
    return r;
}

Canvas render(Camera c, World w){
    Canvas image; image.init(c.hsz,c.vsz);
    Ray ray; Color color;
    for (int y = 0; y < c.vsz; y++){
        for (int x = 0; x < c.hsz; x++){
            ray = rayforpixel(c,x,y);
            color = color_at(w,ray);
            image.write_pixel(x,y,color);
        }
        //printf("%d\n",y);
    }
    return image;
}
