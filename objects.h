#ifndef OBJECTS_H
#define OBJECTS_H

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <math.h>

#define EPSILON 0.00001
#define PI 3.14159265

class Tuple{
    public:
        double x;
        double y;
        double z;
        double w;
        void Vector (double,double,double);
        void Point (double,double,double);
};

// Compares two numbers to see if they're equal,
// within a margin of error of EPSILON (defined in line 4)
bool equal(double a, double b);

Tuple CVector(double x, double y, double z);
Tuple CPoint(double x, double y, double z);

// Sums two 4-element arrays, item by item 
Tuple sumt(Tuple a, Tuple b);

// Substracts two 4-element arrays, item by item
Tuple subt(Tuple a, Tuple b);

// Negates all items in a 4-element array
Tuple negate(Tuple arr);

// Multiplies a 4-element array times a scalar
Tuple smul(Tuple arr, double scalar);

// Divides a 4-element array by a scalar
Tuple sdiv(Tuple arr, double scalar);

// Returns the magnitude of a vector
double magnitude(Tuple arr);

// Normalizes a vector
Tuple normalize(Tuple arr);

// Returns the dot product of two 4-element arrays
double dotproduct(Tuple a, Tuple b);

// Returns the cross product of two vectors as a vector
Tuple crossproduct(Tuple a, Tuple b);

class Projectile{
    public:
        Tuple pos;
        Tuple vel;
        void SetPV(Tuple _pos, Tuple _vel);
};

class Environment{
    public:
        Tuple gravity;
        Tuple wind;
        void SetGW(Tuple gravity, Tuple wind);
};

Projectile tick(Environment env, Projectile proj);

class Color{
    public:
        double r;
        double g;
        double b;
        
        Color(){}
        Color(double R, double G, double B): r(R),g(G),b(B){}
};

Color sumc(Color a, Color b);
Color subc(Color a, Color b);
Color smulc(Color c, double s);
Color hprodc(Color a, Color b);

class Canvas{
    public:
        int width;
        int height;
        std::vector<std::vector<Color>> mat;
        void init(int, int);
        void write_pixel(int, int, Color);
        void pixel_at(int, int);
        void canvas_to_ppm(std::string name);
};

double determinant(std::vector<std::vector<double>> M);
std::vector<std::vector<double>> submatrix(std::vector<std::vector<double>> M, int row, int col);
double minor(std::vector<std::vector<double>> M, int row, int col);
double cofactor(std::vector<std::vector<double>> M, int row, int col);
std::vector<std::vector<double>> inversem(std::vector<std::vector<double>> M);
std::vector<std::vector<double>> matrix_multiply(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B);
Tuple matrix_tuple(std::vector<std::vector<double>> A, Tuple B);
std::vector<std::vector<double>> translation(double x, double y, double z);
std::vector<std::vector<double>> scaling(double x, double y, double z);
std::vector<std::vector<double>> rotx(double r);
std::vector<std::vector<double>> roty(double r);
std::vector<std::vector<double>> rotz(double r);
std::vector<std::vector<double>> shearing(double xy, double xz, double yx, double yz, double zx, double zy);
std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> &b);

class Ray{
    public:
        Tuple origin, direction;
        Tuple raypos(double);
};

std::vector<std::vector<double>> IDENTITYMATRIX = {{1,0,0,0},
                                         {0,1,0,0},
                                         {0,0,1,0},
                                         {0,0,0,1}};

class Material{
    public:
        Color color;
        double ambient, diffuse, specular, shininess;

        Material(): color(Color(1,1,1)), 
                    ambient(0.1), diffuse(0.9),
                    specular(0.9), shininess(200.0){ }
        Material(Color c, double a, double d, double sp, double sh):
                color(c), ambient(a), diffuse(d), specular(sp), shininess(sh){ }
};

class Sphere{
    public:
        Tuple origin;
        double radius;
        std::vector<std::vector<double>> transform;
        Material material;

        Sphere(): origin(CVector(0,0,0)), radius(1), transform(IDENTITYMATRIX), material(Material()){ }
        Sphere(Tuple o, double r, std::vector<std::vector<double>> t, Material m): origin(o), radius(r), transform(t), material(m){ }

};

class Intersection{
    public:
        double t;
        Sphere obj;
        bool operator<( const Intersection& val ) const { 
            return t < val.t; 
        }   
};

Ray transform_ray(Ray r, std::vector<std::vector<double>> M);
std::vector<Intersection> intersect(Sphere s, Ray r0);
std::vector<Intersection> intersections(Intersection i1, Intersection i2);
Intersection hit(std::vector<Intersection> xs);

Tuple normal_at(Sphere s, Tuple wp);
Tuple reflect(Tuple v, Tuple n);

class PointLight{
    public:
        Tuple position;
        Color intensity;

        PointLight(Tuple p, Color c): position(p), intensity(c){ }
};



Color lightning(Material material, PointLight light, Tuple point, Tuple eyev, Tuple normalv);

class World{
    public:
        std::vector<Sphere> objs;
        PointLight lsource;
        World(std::vector<Sphere> o, PointLight p): objs(o), lsource(p) { }
};

World default_world(void);
std::vector<Intersection> intersect_world(World w, Ray r);

class Computation{
    public:
        double t;
        Sphere obj;
        Tuple point, eyev, normalv;
        bool inside;
        Computation(double a, Sphere b, Tuple c, Tuple d, Tuple e, bool f): t(a),obj(b),point(c),eyev(d),normalv(e),inside(f){ }
};

Computation prepComp(Intersection i, Ray r);
Color shade_hit(World world, Computation comps);
Color color_at(World w, Ray r);
std::vector<std::vector<double>> view_transform(Tuple _from, Tuple _to, Tuple _up);

class Camera{
    public:    
        int hsz, vsz;
        double fov;
        std::vector<std::vector<double>> transform;
        double hw, hh, pixelsize;
        void init(int, int, double, std::vector<std::vector<double>>);
        /*
        Camera(int h, int v, double f, std::vector<std::vector<double>> t):
            hsz(h), vsz(v), fov(f), transform(t),
            hw((tan(f/2)*((h/v)>=1))+(tan(f/2)*(h/v)*((h/v)<1))),
            hh((tan(f/2)/(h/v)*((h/v)>=1))+(tan(f/2)*((h/v)<1))),
            pixelsize(((tan(f/2)*((h/v)>=1))+(tan(f/2)*(h/v)*((h/v)<1)))*2/h){ }
        */
};

Ray rayforpixel(Camera camera, int px, int py);
Canvas render(Camera c, World w);
#endif
