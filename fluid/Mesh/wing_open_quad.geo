// Dragonfly wing in open domain with Quad mesh 
////////////////////////////////////////////////////////////
SetFactory("OpenCASCADE");

L = 0.01;
lc = 1e-4;

L_channel = 36*L;
H = 21*L;
dx = 10*L;
dy = 11*L;

Rectangle(1) = {0, 0, 0, L_channel, H};

////////////////////////////////////////////////////////////
// Wing geometry
////////////////////////////////////////////////////////////

thickness = 0.005*L;

// Bottom layer points
Point(6)  = {0, -thickness/2, 0, lc};
Point(7)  = {0.09*L, -thickness/2, 0, lc};
Point(8)  = {0.14*L, 0.055*L - thickness/2, 0, lc};
Point(9)  = {0.19*L, -thickness/2, 0, lc};
Point(10) = {0.24*L, 0.055*L - thickness/2, 0, lc};
Point(11) = {0.29*L, -thickness/2, 0, lc};
Point(12) = {0.34*L, -thickness/2, 0, lc};
Point(13) = {0.5*L, 0.02*L - thickness/2, 0, lc};
Point(14) = {0.6*L, 0.06*L - thickness/2, 0, lc};
Point(15) = {0.7*L, 0.08*L - thickness/2, 0, lc};
Point(16) = {0.75*L, 0.085*L - thickness/2, 0, lc};
Point(17) = {0.8*L, 0.08*L - thickness/2, 0, lc};
Point(18) = {0.85*L, 0.07*L - thickness/2, 0, lc};
Point(19) = {0.9*L, 0.055*L - thickness/2, 0, lc};
Point(20) = {0.95*L, 0.04*L - thickness/2, 0, lc};
Point(21) = {1*L, 0.025*L - thickness/2, 0, lc};

// Top layer points
Point(22) = {0, thickness/2, 0, lc};
Point(23) = {0.09*L, thickness/2, 0, lc};
Point(24) = {0.14*L, 0.055*L + thickness/2, 0, lc};
Point(25) = {0.19*L, thickness/2, 0, lc};
Point(26) = {0.24*L, 0.055*L + thickness/2, 0, lc};
Point(27) = {0.29*L, thickness/2, 0, lc};
Point(28) = {0.34*L, thickness/2, 0, lc};
Point(29) = {0.5*L, 0.02*L + thickness/2, 0, lc};
Point(30) = {0.6*L, 0.06*L + thickness/2, 0, lc};
Point(31) = {0.7*L, 0.08*L + thickness/2, 0, lc};
Point(32) = {0.75*L, 0.085*L + thickness/2, 0, lc};
Point(33) = {0.8*L, 0.08*L + thickness/2, 0, lc};
Point(34) = {0.85*L, 0.07*L + thickness/2, 0, lc};
Point(35) = {0.9*L, 0.055*L + thickness/2, 0, lc};
Point(36) = {0.95*L, 0.04*L + thickness/2, 0, lc};
Point(37) = {1*L, 0.025*L + thickness/2, 0, lc};

// Bottom contour
Line(6)  = {6, 7};
Line(7)  = {7, 8};
Line(8)  = {8, 9};
Line(9)  = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 15};
Line(15) = {15, 16};
Line(16) = {16, 17};
Line(17) = {17, 18};
Line(18) = {18, 19};
Line(19) = {19, 20};
Line(20) = {20, 21};

// Top contour
Line(21) = {22, 23};
Line(22) = {23, 24};
Line(23) = {24, 25};
Line(24) = {25, 26};
Line(25) = {26, 27};
Line(26) = {27, 28};
Line(27) = {28, 29};
Line(28) = {29, 30};
Line(29) = {30, 31};
Line(30) = {31, 32};
Line(31) = {32, 33};
Line(32) = {33, 34};
Line(33) = {34, 35};
Line(34) = {35, 36};
Line(35) = {36, 37};

Line(36) = {6, 22};
Line(37) = {21, 37};

Line Loop(10) = {
  6:20,
  37,
  -35:-21,
  -36
};

Plane Surface(2) = {10};

Translate {dx, dy, 0} {
  Surface{2};
}

wing_boundary[] = Boundary{ Surface{2}; };

////////////////////////////////////////////////////////////
// Boolean
////////////////////////////////////////////////////////////

fluid_surface[] =
BooleanDifference{ Surface{1}; Delete; }
                 { Surface{2}; Delete; };

////////////////////////////////////////////////////////////
// Physical groups
////////////////////////////////////////////////////////////

Physical Surface("Fluid", 1) = {fluid_surface[0]};
Physical Curve("Inlet", 2) = {38};
Physical Curve("Outlet", 3) = {39};
Physical Curve("Walls", 4) = {37, 40};
Physical Curve("Wing", 5) = {wing_boundary[]};

////////////////////////////////////////////////////////////
// Mesh controls
////////////////////////////////////////////////////////////

res_min = thickness;
lc_max =  L;

Field[1] = Distance;
Field[1].CurvesList = wing_boundary[];
Field[1].Sampling = 500;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = res_min;
Field[2].SizeMax = lc_max;
Field[2].DistMin = 0.2;
Field[2].DistMax = 1;

Background Field = 2;

Mesh.Algorithm = 8;                 // Frontal-Delaunay for quads
Mesh.RecombinationAlgorithm = 2;  // Simple full-quad recombination
Mesh.RecombineAll = 1;            // Recombine all triangles
Mesh.SubdivisionAlgorithm = 1;    // All quadrangles


Mesh 2;
Mesh.ElementOrder = 2;
SetOrder 2;

Mesh.Optimize = 1;
OptimizeMesh "Netgen";

Mesh.MshFileVersion = 2.2;     
Save "wing_open_quad.msh";