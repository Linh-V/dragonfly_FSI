import getfem as gf
import numpy as np
import os
from regions_check import verify_regions

# geometric paramters:
L = 10
thickness = 0.005*L

# Material properties
E = 70e5                                      # Young Modulus  ---> 70 Gpa
nu = 0.33                                       # Poisson ratio
clambda = E*nu/((1+nu)*(1-nu))                  # First Lame coefficient (N/cm^2)
cmu = E/(2*(1+nu))                              # Second Lame coefficient (N/cm^2)

# Force
F = - 0.01 /(thickness*0.1)                   # Force at the right Boundary (N)



# IMPORT OF THE DRAGONFLY GMSH 

m = gf.Mesh('Import','gmsh','dragonfly2_quads.msh')
#verify_regions(m,'quadmeshcheck')
# # triangles
# LEFT = 1
# RIGHT = 17
# quads
LEFT = 31
RIGHT = 46
# selection of finite element
mu = gf.MeshFem(m, 2)
mu.set_fem(gf.Fem('FEM_Q2_INCOMPLETE(2)'))
mim = gf.MeshIm(m, gf.Integ('IM_QUAD(17)'))  # Standard integration OK

# MODEL
md = gf.Model("real")
md.add_fem_variable("u",mu)
md.add_initialized_data('cmu', [cmu])
md.add_initialized_data('clambda', [clambda])




md.add_macro("eps(u)", "0.5*(Grad_u + Grad_u')")
md.add_macro("sigma(u)", "2*cmu*eps(u) + clambda*Id(2)*Div_u")
md.add_linear_term(mim, "sigma(u):eps(Test_u)")
#md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'clambda', 'cmu')
md.add_initialized_data('ForceData',[0,F])

md.add_source_term_brick(mim, 'u', 'ForceData', RIGHT)
# Check deflection/length ratio


# BOUNDARY CONDITIONS:

md.add_Dirichlet_condition_with_multipliers(mim,'u', mu, LEFT)

md.solve()

U = md.variable("u")
# Separate x and y displacements
u_x = U[0::2]  # x-displacements
u_y = U[1::2]  # y-displacements
if max(abs(u_y))/L > 0.1:
    print("WARNING: Large deflection - linear theory may not be valid")

print(f"Maximum displacement: {max(abs(u_y))} (cm)")

FolderName = "LinearElasticityResults"
os.system(f'mkdir {FolderName}')
SaveFile = os.path.join(FolderName,'2d_dragonfly_quads.vtk')
mu.export_to_vtk(SaveFile,  mu, U, 'Displacements')


# Regions for b.c seems correct,
# THe results seems mesh indipendent, test with difference types of mesh and different dimension
# the physics seems correct becuase it works fine with the beam
# what else could be different? 
# types of element instead of qk?
# maybe the rotation? 
