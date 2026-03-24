// Definition of the Geometry for Fluid Dynamics Simulation
// to run it in the terminal: gmsh Beam.geo -2

////////////////////////////////////////////////////////////
// Constants
h_ref = 0.5;
L = 10;
thickness = 0.005*L;

// Number of elements along each direction
nx1 = 20;  // Left of beam
nx2 = 40;  // Along beam
nx3 = 20;  // Right of beam
ny1 = 20;  // Below/Above beam (symmetric)
ny2 = 5;   // Along beam thickness

////////////////////////////////////////////////////////////
// Points Definition
////////////////////////////////////////////////////////////

// Points for beam
Point(5) = {0, -thickness/2, 0, h_ref};                        // Beam bottom left
Point(6) = {L, -thickness/2, 0, h_ref};                        // Beam bottom right
Point(7) = {L, thickness/2, 0, h_ref};                // Beam top right
Point(8) = {0, thickness/2, 0, h_ref};                // Beam top left



//+
Line(1) = {8, 5};
//+
Line(2) = {5, 6};
//+
Line(3) = {6, 7};
//+
Line(4) = {7, 8};



Curve Loop(1) = {2, 3, 4, 1};

//+
Plane Surface(1) = {1};
//+
Physical Curve("beam_left", 1) = {1};
//+
Physical Curve("beam_bottom", 2) = {2};
//+
Physical Curve("beam_right", 3) = {3};
//+
Physical Curve("beam_top", 4) = {4};


// Physical Surfaces
Physical Surface("beam", 5) = {1};

//+
Transfinite Surface {1} = {8, 5, 6, 7};

//
Transfinite Curve {1, 3} = 10 Using Progression 1;
//+
Transfinite Curve {2, 4} = 40 Using Progression 1;

Recombine Surface{1}; 
Mesh.Algorithm = 6;

Mesh 2;
Save "Beam.msh";
