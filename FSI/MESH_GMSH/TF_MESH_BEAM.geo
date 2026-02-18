// Mesh for TF FSI2 benchmark test- > SOLID DOMAIN
h_ref = 0.5;

L = 2.5; // m
H = 0.41; // m
c = 0.2; // m is the x-position of cylinder center
r = 0.05; // m radius of the cylinder
l = 0.35; // m length of the beam from the cylinder
h = 0.02; // m height of the beam
l_beam_from0 = 0.6;


// Calculate x-position of beam attachment to ensure points are exactly on cylinder
x_attach = c + Sqrt(r*r - (h/2)*(h/2)); // Using Pythagorean theorem

// Beam attachment points - EXACTLY on the cylinder
Point(1) = {x_attach, c - h/2, 0, h_ref/20}; // Bottom of beam at cylinder
Point(2) = {x_attach, c + h/2, 0, h_ref/20}; // Top of beam at cylinder
Point(3) = {0.3, c + h/2, 0, h_ref/20};
Point(4) = {0.3, c - h/2, 0, h_ref/20}; 

Point(5) = {l_beam_from0, c + h/2, 0, h_ref/20}; // Top right of beam
Point(6) = {l_beam_from0, c - h/2, 0, h_ref/20}; // Bottom right of beam

// Lines
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,5};
Line(4) = {5,6};
Line(5) = {6,4};
Line(6) = {4,1};
Line(7) = {3,4};


//+
Curve Loop(1) = {6, 1, 2, 7};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {5, -7, 3, 4};
//+
Plane Surface(2) = {2};
// TRANSFINTE
N_mult = 1.2;
N4 = 6*N_mult;
N5 = 3*N_mult;
N6  = 10*N_mult; 
pf_bd = 1/1.3;





// BLOCK 10 
//+
Transfinite Curve {1, 7} = N5 Using Progression 1;
//+
Transfinite Curve {6} = N4 Using Progression pf_bd;
Transfinite Curve {2} = N4 Using Progression 1/pf_bd;

//+
Transfinite Surface {1};

// BLOCK 14

// BLOCK 10 
//+
Transfinite Curve {7,4} = N5 Using Progression 1;
//+
Transfinite Curve {3, 5} = N6 Using Progression 1;
//+
Transfinite Surface {2};

// BLOCK 14
//+
Physical Curve("BEAM_INTERFACE", 8) = {2, 3, 4, 5, 6};
//+
Physical Curve("BEAM_LEFT", 9) = {1};
//+
Physical Surface("BEAM", 10) = {1, 2};
//+
Recombine Surface {1, 2};



Mesh 2; 
Save "TF_MESH_BEAM.msh";
