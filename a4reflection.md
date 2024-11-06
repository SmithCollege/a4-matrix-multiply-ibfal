when does it make sense to use the various approaches?
    I feel like the tiled and naive GPU versions make the most sense to use any sized data. The cublas one is great if you cant use GPU kernels necessarily because it performs better than just the standard C program.

How did your speed compare with cuBLAS?
    cuBLAS was faster than the c programs but not as fast as the two GPU, mmc/tiled aproaches.

What went well with this assignment?
    i found working with the matrix and the cublas functions to be quiet easy and simple to work with since i had the background. 

What was difficult?
    I was a little concerned with the cuBLAS lib not linking as well as i had some issues grasping the tiles, but once i did it made a lot more sense. I was a little confused because when i make the tile size 32 i started to get errors where the sums were like 980 for a 1000 size array. Which im assuming was because the 

How would you approach differently?
    I'm not sure. I felt pretty good about this one. 


Anything else you want me to know?


