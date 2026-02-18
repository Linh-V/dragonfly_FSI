// Mesh for TF FSI2 benchmark test - Automatic structured quad mesh

h_ref = 0.5;

L = 2.2; // m
H = 0.41; // m
c = 0.2; // m is the x-position of cylinder center
r = 0.05; // m radius of the cylinder
l = 0.35; // m length of the beam from the cylinder
h = 0.02; // m height of the beam

domain_x_min = 0;
domain_x_max = L;
domain_y_min = 0;
domain_y_max = H;

// Calculate x-position of beam attachment to ensure points are exactly on cylinder
x_attach = c + Sqrt(r*r - (h/2)*(h/2)); // Using Pythagorean theorem

////////////////////////////////////////////////////////////
// Points Definition
////////////////////////////////////////////////////////////
// Outer corners
Point(1) = {domain_x_min, domain_y_min, 0, h_ref}; // Bottom left
Point(2) = {domain_x_max, domain_y_min, 0, h_ref}; // Bottom right
Point(3) = {domain_x_max, domain_y_max, 0, h_ref}; // Top right
Point(4) = {domain_x_min, domain_y_max, 0, h_ref}; // Top left

// Cylinder center
Point(5) = {c, c, 0, h_ref/10};

// Beam attachment points - EXACTLY on the cylinder
Point(10) = {x_attach, c - h/2, 0, h_ref/20}; // Bottom of beam at cylinder
Point(11) = {x_attach, c + h/2, 0, h_ref/20}; // Top of beam at cylinder

// Beam end points
Point(12) = {0.6, c + h/2, 0, h_ref/20}; // Top right of beam
Point(13) = {0.6, c - h/2, 0, h_ref/20}; // Bottom right of beam

// points for meshing
Point(14) = {c - r*Cos(Pi/4), c - r*Sin(Pi/4), 0, h_ref/20}; // circle bottom-left
Point(15) = {c - r*Cos(Pi/4), c + r*Sin(Pi/4), 0, h_ref/20}; // circle top-left
Point(16) = {c + r*Cos(Pi/4), c + r*Sin(Pi/4), 0, h_ref/20}; // circle top-right
Point(17) = {c + r*Cos(Pi/4), c - r*Sin(Pi/4), 0, h_ref/20}; // circle bottom-right

Point(18) = {0.1, 0.1, 0, h_ref/20}; // bottom left
Point(19) = {0.1, 0.3, 0, h_ref/20}; // top left
Point(20) = {0.3, 0.3, 0, h_ref/20}; // top right
Point(21) = {0.3, 0.1, 0, h_ref/20}; // bottom right

Point(22) = {0.6, 0.3, 0, h_ref/20}; // top far right
Point(23) = {0.6, 0.1, 0, h_ref/20}; // bottom far right

Point(24) = {0.1, 0, 0, h_ref/20};
Point(25) = {0.6, 0, 0, h_ref/20};
Point(26) = {0.1, H, 0, h_ref/20};
Point(27) = {0.6, H, 0, h_ref/20};

Point(28) = {L, c + h/2, 0, h_ref/20};
Point(29) = {L, c - h/2, 0, h_ref/20}; 

Point(30) = {L, 0.3, 0, h_ref/20};
Point(31) = {L, 0.1, 0, h_ref/20};

Point(32) = {0.3, c + h/2, 0, h_ref/20};
Point(33) = {0.3, c - h/2, 0, h_ref/20}; 
Point(34) = {0.3, 0, 0, h_ref/20};
Point(35) = {0.3, H, 0, h_ref/20}; 

Point(36) = {0, 0.3, 0, h_ref/20}; 
Point(37) = {0, 0.1, 0, h_ref/20}; 

// Lines
Line(1) = {4, 26};
Line(2) = {26, 35};
Line(3) = {35, 27};
Line(4) = {27, 3};
Line(5) = {3, 30};
Line(6) = {30, 28};
Line(7) = {28, 29};
Line(8) = {29, 31};
Line(9) = {31, 2};
Line(10) = {2, 25};
Line(11) = {25, 34};
Line(12) = {34, 24};
Line(13) = {24, 1};
Line(14) = {1, 37};
Line(15) = {37, 36};
Line(16) = {36, 4};
Circle(17) = {11, 5, 16};
Circle(18) = {16, 5, 15};
Circle(19) = {15, 5, 14};
Circle(20) = {14, 5, 17};
Circle(21) = {17, 5, 10};
Line(22) = {11, 10};
Line(23) = {33, 10};
Line(24) = {33, 13};
Line(25) = {13, 12};
Line(26) = {12, 32};
Line(27) = {32, 11};
Line(28) = {32, 33};
Line(29) = {26, 19};
Line(30) = {19, 18};
Line(31) = {18, 24};
Line(32) = {35, 20};
Line(33) = {20, 32};
Line(34) = {33, 21};
Line(35) = {21, 34};
Line(36) = {27, 22};
Line(37) = {12, 22};
Line(38) = {13, 23};
Line(39) = {23, 25};
Line(40) = {19, 15};
Line(41) = {18, 14};
Line(42) = {21, 17};
Line(43) = {20, 16};
Line(44) = {36, 19};
Line(45) = {19, 20};
Line(46) = {20, 22};
Line(47) = {22, 30};
Line(48) = {12, 28};
Line(49) = {13, 29};
Line(50) = {37, 18};
Line(51) = {18, 21};
Line(52) = {21, 23};
Line(53) = {23, 31};

// Curve Loops and Surfaces
Curve Loop(1) = {1, 29, -44, 16};
Plane Surface(1) = {1};
Curve Loop(2) = {2, 32, -45, -29};
Plane Surface(2) = {2};
Curve Loop(3) = {32, 46, -36, -3};
Plane Surface(3) = {3};
Curve Loop(4) = {36, 47, -5, -4};
Plane Surface(4) = {4};
Curve Loop(5) = {44, 30, -50, 15};
Plane Surface(5) = {5};
Curve Loop(6) = {40, 19, -41, -30};
Plane Surface(6) = {6};
Curve Loop(7) = {45, 43, 18, -40};
Plane Surface(7) = {7};
Curve Loop(8) = {43, -17, -27, -33};
Plane Surface(8) = {8};
Curve Loop(9) = {46, -37, 26, -33};
Plane Surface(9) = {9};
Curve Loop(10) = {47, 6, -48, 37};
Plane Surface(10) = {10};
Curve Loop(11) = {48, 7, -49, 25};
Plane Surface(11) = {11};
Curve Loop(12) = {41, 20, -42, -51};
Plane Surface(12) = {12};
Curve Loop(13) = {-23, 34, 42, 21};
Plane Surface(13) = {13};
Curve Loop(14) = {24, 38, -52, -34};
Plane Surface(14) = {14};
Curve Loop(15) = {49, 8, -53, -38};
Plane Surface(15) = {15};
Curve Loop(16) = {50, 31, 13, 14};
Plane Surface(16) = {16};
Curve Loop(17) = {51, 35, 12, -31};
Plane Surface(17) = {17};
Curve Loop(18) = {52, 39, 11, -35};
Plane Surface(18) = {18};
Curve Loop(19) = {53, 9, 10, -39};
Plane Surface(19) = {19};

// BEAM 

Curve Loop(20) = {22, -23, - 28, 27};
Plane Surface(20) = {20};

Curve Loop(21) = {28, 24, 25,26};
Plane Surface(21) = {21};

//TRANSFINITE CURVE DEFINITIONS
N_mult = 1.2;
N1 = N_mult*4;
N2= N_mult*4;
N3 = N_mult*6;
N4 = N_mult*6;
N5 = N_mult*3;
N6 = N_mult*10;
N7 = N_mult*15;
pf = 1.1;
pf_db = 1/1.3;
// Block 1
//+
Transfinite Curve {16, 29} = N1 Using Progression 1;
//+
Transfinite Curve {1, 44} = N2 Using Progression 1;
//+
Transfinite Surface {1};
// Block 2
//+
Transfinite Curve {44, 50} = N2 Using Progression 1;
//+
Transfinite Curve {15, 30} = N3 Using Progression 1;
//+
Transfinite Surface {5};
// Block 3
//+
Transfinite Curve {14, 31} = N1 Using Progression 1;
//+
Transfinite Curve {50, 13} = N2 Using Progression 1;
//+
Transfinite Surface {16};
//+

// Block 4
//+
Transfinite Curve {29, 32} = N2 Using Progression 1;
//+
Transfinite Curve {2, 45} = N3 Using Progression 1;
//+
Transfinite Surface {2};

// Block 5 

//+
Transfinite Curve {45, 18} = N3 Using Progression 1;
//+
Transfinite Curve {40, 43} = N4 Using Progression pf_db;
//+
Transfinite Surface {7};

// Block 6 
//+
Transfinite Curve {30, 19} = N3 Using Progression 1;
//+
Transfinite Curve {40, 41} = N4 Using Progression pf_db;
//+
Transfinite Surface {6};


// Block 7
//+
Transfinite Curve {51, 20} = N3 Using Progression 1;
//+
Transfinite Curve {41, 42} = N4 Using Progression pf_db;
//+
Transfinite Surface {12};

// Block 8 
//+
Transfinite Curve {31, 35} = N2 Using Progression 1;
//+
Transfinite Curve {51, 12} = N3 Using Progression 1;
//+
Transfinite Surface {17};

// Block 9 
//+
Transfinite Curve {21, 34} = N3 Using Progression 1;
//+
Transfinite Curve {42, 23} = N4 Using Progression pf_db;
//+
Transfinite Surface {13};


// Block 11 
//+
Transfinite Curve {17, 33} = N3 Using Progression 1;
//+
Transfinite Curve {27, 43} = N4 Using Progression pf_db;
//+
Transfinite Surface {8};

// Block 12 
//+
Transfinite Curve {32, 36} = N1 Using Progression 1;
//+
Transfinite Curve {3, 46} = N6 Using Progression 1;
//+
Transfinite Surface {3};

// Block 13

//+
Transfinite Curve {33, 37} = N3 Using Progression 1;
//+
Transfinite Curve {46,26} = N6 Using Progression 1;
//+
Transfinite Surface {9};


// Block 15

//+
Transfinite Curve {34, 38} = N3 Using Progression 1;
//+
Transfinite Curve {24,52} = N6 Using Progression 1;
//+
Transfinite Surface {14};

// Block 16

//+
Transfinite Curve {35, 39} = N1 Using Progression 1;
//+
Transfinite Curve {52,11} = N6 Using Progression 1;
//+
Transfinite Surface {18};

// Block 17

//+
Transfinite Curve {36, 5} = N1 Using Progression 1;
//+
Transfinite Curve {4,47} = N7 Using Progression pf;
//+
Transfinite Surface {4};

// Block 18

//+
Transfinite Curve {37, 6} = N3 Using Progression 1;
//+
Transfinite Curve {47,48} = N7 Using Progression pf;
//+
Transfinite Surface {10};

// Block 19

//+
Transfinite Curve {25, 7} = N5 Using Progression 1;
//+
Transfinite Curve {48,49} = N7 Using Progression pf;
//+
Transfinite Surface {11};

// Block 19

//+
Transfinite Curve {38, 8} = N3 Using Progression 1;
//+
Transfinite Curve {49,53} = N7 Using Progression pf;
//+
Transfinite Surface {15};


// Block 20

//+
Transfinite Curve {39, 9} = N1 Using Progression 1;
//+
Transfinite Curve {53} = N7 Using Progression pf;
Transfinite Curve {10} = N7 Using Progression 1/pf;
//+
Transfinite Surface {19};

// Beam transfinite
// Block 21
Transfinite Curve {26, 24} = N6 Using Progression 1;
Transfinite Curve {25, 28} = N5 Using Progression 1;
Transfinite Surface {21};
//
Transfinite Curve {23, 27} = N4 Using Progression pf_db;
//+
Transfinite Curve {22, 28} = N5 Using Progression 1;
//+
Transfinite Surface {20};

// PHYSICAL CURVES
Physical Curve("INLET", 54) = {16, 15, 14};
Physical Curve("OUTLET", 55) = {5, 6, 7, 8, 9};
Physical Curve("walls", 56) = {1, 2, 3, 4, 13, 12, 11, 10};
Physical Curve("CYLINDER", 57) = {17, 18, 19, 20, 21, 22};  
Physical Curve("BEAM_INTERFACE", 59) = {27, 26, 25, 24, 23};  

Physical Surface("FLUID", 60) = {1, 2, 3, 4, 5, 6, 7, 8, 13, 12, 16, 17, 18, 14, 9, 10, 11, 15, 19};
Physical Surface("BEAM", 61) = {20, 21};
Recombine Surface {1:21};
 Mesh 2;
 Save "TF_1MESH_quads.msh";
 
