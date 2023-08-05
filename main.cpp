#include <stdio.h>
#include <iostream>
#include "objects.h"
#include "objects.cpp"


int main(){
    Sphere floor = Sphere();
    floor.transform = scaling(10,0.01,10);
    floor.material = Material();
    floor.material.color = Color(1,0.9,0.9);
    floor.material.specular = 0;

    Sphere left_wall = Sphere();
    left_wall.transform = matrix_multiply(matrix_multiply(matrix_multiply(translation(0,0,5),roty(-PI/4)),rotx(PI/2)),scaling(10,0.01,10));
    left_wall.material = floor.material;

    Sphere right_wall = Sphere();
    right_wall.transform = matrix_multiply(matrix_multiply(matrix_multiply(translation(0,0,5),roty(PI/4)),rotx(PI/2)),scaling(10,0.01,10));
    right_wall.material = floor.material;

    Sphere middle = Sphere();
    middle.transform = translation(-0.5,1,0.5);
    middle.material = Material();
    middle.material.color = Color(0.1,1,0.5),
    middle.material.diffuse = 0.7;
    middle.material.specular = 0.3;

    Sphere right = Sphere();
    right.transform = matrix_multiply(translation(1.5,0.5,-0.5),scaling(0.5,0.5,0.5));
    right.material = Material();
    right.material.color = Color(0.1,1,0.5);
    right.material.diffuse = 0.7;
    right.material.specular = 0.3;

    Sphere left = Sphere();
    left.transform = matrix_multiply(translation(-1.5,0.33,-0.75),scaling(0.33,0.33,0.33));
    left.material = Material();
    left.material.color = Color(1,0.8,0.1);
    left.material.diffuse = 0.7;
    left.material.specular = 0.3;

    std::vector<Sphere> objs;
    objs.push_back(floor);
    objs.push_back(left_wall);
    objs.push_back(right_wall);
    objs.push_back(left);
    objs.push_back(middle);
    objs.push_back(right);

    World w = World(objs,PointLight(CPoint(-10,10,-10),Color(1,1,1)));

    Camera camera;
    camera.init(84,48,PI/3,view_transform(CPoint(0,1.5,-5),CPoint(0,1,0),CVector(0,1,0)));
    Canvas canvas = render(camera,w);
    canvas.canvas_to_ppm("1");
    
    w.objs[4].transform = matrix_multiply(translation(0.5,0.5,0.5),w.objs[4].transform);
    canvas = render(camera,w);
    canvas.canvas_to_ppm("2");

    w.objs[4].transform = matrix_multiply(translation(0.5,0.5,0.5),w.objs[4].transform);
    canvas = render(camera,w);
    canvas.canvas_to_ppm("3");

    w.objs[4].transform = matrix_multiply(translation(0.5,0.5,0.5),w.objs[4].transform);
    canvas = render(camera,w);
    canvas.canvas_to_ppm("4");

    w.objs[4].transform = matrix_multiply(translation(0.5,0.5,0.5),w.objs[4].transform);
    canvas = render(camera,w);
    canvas.canvas_to_ppm("5");

    printf("Done.\n");
    return 0;
}