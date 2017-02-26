# Cuda-Opengl-LBM
Lattice-Botlzmann fluid simulator using CUDA and OpenGL

Written by Tom Scherlis and Henry Friedlander

## Description:
CUDA and OpenGL packages were used to graphically accelerate the Lattice-Boltzmann Method
(LBM) Algorithms. Through the use of parallelization and efficient memory usage, considerable
optimization and efficiency is possible with LBM simulation. In this simulation, we model constant
2 dimensional fluid flow through a medium, and at every lattice node the fluid’s curl, probabilistic
velocity vectors, and pressure are calculated and available to be plotted on the screen.

## Links:
More information here:
http://tomscherlis.com/otw-portfolio/lbm/

Demo Vid:
https://www.youtube.com/watch?v=MSMGGoP24Hw

Technical Report:
http://tomscherlis.com/wordpress/wp-content/uploads/2017/02/Lattice-Boltzmann-Algorithm-Using-GPU-Acceleration.pdf

## Controls:
![controls](http://i.imgur.com/623OvSZ.png)

## Screenshots:
![screenshots 1](http://i.imgur.com/n6UrF8J.png)  
![screenshots 2](http://i.imgur.com/Ew07zWf.png)

## Known Bugs:
Fast mouse movement when drawing barriers can lead to gaps.

Combinations of low viscosity, high speed, large barriers, or fast barrier draw speed can lead to corruption due to supersonic microvelocities of the fluid. Be careful when drawing to avoid this!
