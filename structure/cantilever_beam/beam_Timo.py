"""
1D Cantilever Beam using Timoshenko beam theory

Governing equations: 
- Shear force equilibrium: dV/dx = 0
- Moment equilibrium: dM/dx = V
- Constitutive relations: M = EI * dtheta/dx, V = kGA * (dw/dx - theta)

Boundary conditions:
At x=0 (fixed end): w(0) = 0, theta(0) = 0
At x=L (free end): M(L) = 0, V(L) = F

Analytical solution: Max deflection = -PL^3/3EI
"""

import getfem as gf 
import numpy as np
import csv

E = 70e9  # Young's Modulus in Pa
H = 0.005 # Height in m
B = 0.005 # Base in m
A = B * H  # Cross-sectional area in m²
I = B*(H**3)/12  # Second moment of inertia in m⁴
L = 0.1 # Beam length in m
nu = 0.3    # Poisson ratio
k = 5/6     # Shear correction factor
G = E/(2*(1+nu))    # Shear modulus

F = -100   # Force at the right Boundary in N

############## Analytical solution #########################################################
# Calculate analytical solution for comparison
analytical_tip_deflection = abs(F * L**3) / (3 * E * I)
print(f"Expected analytical tip deflection: {analytical_tip_deflection:.6e} meters")

############## Mesh #########################################################
X = np.linspace(0, L, 41)
mesh = gf.Mesh('cartesian', X)

# # Create a 1D mesh embedded in 2D (e.g., for Timoshenko beam)
# mesh = gf.Mesh('empty', 2)  # 1D mesh

# pts = list(zip(X, np.zeros_like(X)))  # shape (2, N+1)
# print(pts)


# point_indices = [mesh.add_point(pts[i]) for i in range(len(pts))]

# # Add 1D convexes between consecutive points
# GT = gf.GeoTrans('GT_PK(1,1)')
# for i in range(len(point_indices) - 1):
#     mesh.add_convex(GT, np.array([pts[i], pts[i+1]]).T)

############## Boundary Identification #########################################################

fleft = mesh.outer_faces_with_direction(v=[-1.0], angle=0.01) #returns faces not shared by convex with outward vector v
fright = mesh.outer_faces_with_direction(v=[+1.0], angle=0.01)
NEUMANN_BOUNDARY = 1
DIRICHLET_BOUNDARY = 2
mesh.set_region(NEUMANN_BOUNDARY, fright) #flag faces
mesh.set_region(DIRICHLET_BOUNDARY, fleft)

############## Finite Element / Integration #################################

mfu = gf.MeshFem(mesh, 2)  # Finite element for the elastic displacement
mfu.set_fem(gf.Fem("FEM_PK(1,3)"))

mim = gf.MeshIm(mesh, gf.Integ("IM_GAUSS1D(5)")) # Integration method  

############## Initialising the model #############################

# empty real model
md = gf.Model('real')

# declare that the unknowns of the system on the finite element method "mfu"
md.add_fem_variable("u", mfu)        # Displacement [w, theta]

# Adding a constant scalar value
md.add_initialized_data("kGA", k*G*A)
md.add_initialized_data("EI", E*I)

############## Weak Form for Timoshenko Beam ####################################
# Bending energy: EI * dtheta/dx * d(delta_theta)/dx
bending_term = "EI * Grad_u(2) * Grad_Test_u(2)"

# Shear energy: kGA * (dw/dx - theta) * (d(delta_w)/dx - delta_theta)
shear_term = "kGA * (Grad_u(1) - u(2)) * (Grad_Test_u(1) - Test_u(2))"

# Add the complete bilinear form
md.add_linear_term(mim, bending_term + " + " + shear_term)

############## Applied Forces ################################################

# Apply transverse force at the tip in global coordinates
# Need to transform to local normal direction   
md.add_initialized_data('F_global', [F, 0])

# Transform global force to local coordinates at the boundary
# F_local_normal = F_global · normal_vector
md.add_source_term_brick(mim, 'u', 'F_global', NEUMANN_BOUNDARY)

############## Boundary Conditions ########################################################

# Fix the left side
md.add_Dirichlet_condition_with_multipliers(mim, "u", 2, DIRICHLET_BOUNDARY)

############### Solve and Export #############################################

# solve the linear system
md.solve()

# main unknown
U = md.variable("u")
print(U)

u_transverse = U[0::2]

max_u = np.max(np.abs(u_transverse))
print(f"Maximum transverse displacement: {max_u} meters")

# export computed solution using vtk format (you can load it with Paraview)
mfu.export_to_vtk("Beam_Timo.vtk", U, "Displacement")
with open('Beam_Timo.csv', 'w', newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ',')
    my_writer.writerow(u_transverse)
