import getfem as gf 
from FSI.Functions import verify_regions
import numpy as np
import os
from datetime import datetime

gf.util_trace_level(1)
gf.util_warning_level(1)
π = np.pi

######################################
# Static Fluid-Structure Interaction
# Two-mesh monolithic ALE formulation
######################################

output_dir = "FSI/FSI_I_BENCHMARK/FSI_Benchmark_I_Results_2meshes"
os.makedirs(output_dir, exist_ok=True)

# Open log file
log_file = open(f"{output_dir}/results_log.txt", "w")

def log(msg=""):
    """Print to console and write to log file."""
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

log(f"FSI-1 Benchmark — Two Mesh Monolithic ALE")
log(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("")

##################
## PROBLEM DATA ##
##################

# Geometry parameters
L = 2.5
H = 0.41
c_x = 0.2
c_y = 0.2
r = 0.05
L_beam = 0.35
W_beam = 0.02

# Structural properties
rho_solid = 1000.0
nu_solid = 0.4
mu_solid = 5e+5
E = 2 * mu_solid * (1 + nu_solid)
lambda_solid = E * nu_solid / ((1 + nu_solid) * (1 - 2 * nu_solid))

# Fluid properties
ν_fluid = 1e-3
rho_fluid = 1000.0

# Boundaries values
U_mean = 0.2

log("=" * 60)
log("Problem Parameters")
log("=" * 60)
log(f"  Channel: L = {L}, H = {H}")
log(f"  Cylinder: center = ({c_x}, {c_y}), r = {r}")
log(f"  Beam: L = {L_beam}, W = {W_beam}")
log(f"  Fluid: rho = {rho_fluid}, nu = {ν_fluid}")
log(f"  Solid: rho = {rho_solid}, E = {E}, nu = {nu_solid}")
log(f"         mu_s = {mu_solid}, lambda_s = {lambda_solid}")
log(f"  Inlet: U_mean = {U_mean}")
log("")

#############
## MESH ##
#############

Mesh_fluid = gf.Mesh('Import', 'gmsh', 'FSI/FSI_I_BENCHMARK/FSI_I_BENCHMARK/MESH_GMSH/TF_MESH_FLUID.msh')
Mesh_solid = gf.Mesh('Import', 'gmsh', 'FSI/FSI_I_BENCHMARK/FSI_I_BENCHMARK/MESH_GMSH/TF_MESH_BEAM.msh')

#############
## REGIONS ##
#############

INLET = 201
OUTLET = 202
WALLS = 203
CYLINDER = 204
BEAM_LEFT = 205
BEAM_INTERFACE = 206
FLUID = 207
BEAM = 208

# Fluid regions
Mesh_fluid.region_merge(FLUID, 28)
for i in range(29, 48):
    if i != 45:
        Mesh_fluid.region_merge(FLUID, i)

Mesh_fluid.region_merge(WALLS, 1)
for i in [2, 3, 4, 10, 11, 12, 13]:
    Mesh_fluid.region_merge(WALLS, i)

Mesh_fluid.region_merge(CYLINDER, 17)
for i in range(18, 22):
    Mesh_fluid.region_merge(CYLINDER, i)

Mesh_fluid.region_merge(BEAM_INTERFACE, 23)
for i in range(24, 28):
    Mesh_fluid.region_merge(BEAM_INTERFACE, i)

Mesh_fluid.region_merge(OUTLET, 5)
for i in range(6, 10):
    Mesh_fluid.region_merge(OUTLET, i)

Mesh_fluid.region_merge(INLET, 14)
for i in range(15, 17):
    Mesh_fluid.region_merge(INLET, i)

# Solid regions
Mesh_solid.region_merge(BEAM_INTERFACE, 2)
for i in range(3, 7):
    Mesh_solid.region_merge(BEAM_INTERFACE, i)

Mesh_solid.region_merge(BEAM_LEFT, 1)
Mesh_solid.region_merge(BEAM, 11)
Mesh_solid.region_merge(BEAM, 7)

log("=" * 60)
log("Mesh Information")
log("=" * 60)
log(f"  Fluid mesh:")
log(f"    Points:   {Mesh_fluid.nbpts()}")
log(f"    Convexes: {Mesh_fluid.nbcvs()}")
log(f"  Solid mesh:")
log(f"    Points:   {Mesh_solid.nbpts()}")
log(f"    Convexes: {Mesh_solid.nbcvs()}")
log("")

########################
## INTEGRATION METHOD ##
########################

mim_fluid = gf.MeshIm(Mesh_fluid, gf.Integ('IM_QUAD(9)'))
mim_solid = gf.MeshIm(Mesh_solid, gf.Integ('IM_QUAD(9)'))

#########################
## FEM ELEMENTS ##
#########################

mfu_fluid = gf.MeshFem(Mesh_fluid, 2)
mfu_fluid.set_fem(gf.Fem('FEM_QK(2,2)'))

mfv_fluid = gf.MeshFem(Mesh_fluid, 2)
mfv_fluid.set_fem(gf.Fem('FEM_QK(2,2)'))

mfp_fluid = gf.MeshFem(Mesh_fluid, 1)
mfp_fluid.set_fem(gf.Fem('FEM_QK(2,1)'))

mfu_solid = gf.MeshFem(Mesh_solid, 2)
mfu_solid.set_fem(gf.Fem('FEM_QK(2,2)'))

###########
## MODEL ##
###########

md = gf.Model("real")

###################
## FEM VARIABLES ##
###################

md.add_fem_variable("u_f", mfu_fluid)
md.add_fem_variable("v_f", mfv_fluid)
md.add_fem_variable("p_f", mfp_fluid)
md.add_fem_variable("u_s", mfu_solid)

# Lagrange Multipliers
md.add_filtered_fem_variable("mult", mfu_fluid, BEAM_INTERFACE)

###################################
## INTERPOLATION TRANSFORMATIONS ##
###################################

md.add_interpolate_transformation_from_expression("fluid_to_solid", Mesh_solid, Mesh_fluid, "X")
md.add_interpolate_transformation_from_expression("solid_to_fluid", Mesh_fluid, Mesh_solid, "X")

###########################
## INITIALIZED CONSTANTS ##
###########################

md.add_initialized_data("rho_f", rho_fluid)
md.add_initialized_data("nu_f", ν_fluid)
md.add_initialized_data("lambda_solid", lambda_solid)
md.add_initialized_data("mu_s", mu_solid)
md.add_initialized_data("rho_s", rho_solid)
md.add_initialized_data("H", H)
md.add_initialized_data("U_mean", U_mean)
md.add_initialized_data("p_out", 0.0)

#################################################
## WEAK FORMULATION (Thomas Wick Problem 7.28) ##
#################################################

########### definition of macros for the weak formulation ###########
# COMMON MACROS
md.add_macro("F(u)", "Id(2)+Grad(u)")
md.add_macro("J(u)", "Det(F(u))")

# FLUID and SOLID STRESS TENSORS
md.add_macro('sigma_f_vu(v,u)', "rho_f*nu_f*(Grad(v)*Inv(F(u)) + (Inv(F(u)))'*(Grad(v))')")
md.add_macro('sigma_f_p(p)', "-p*Id(2)")
md.add_macro("E(u)", "0.5*((F(u))'*F(u) - Id(2))")
md.add_macro('Sigma_s(u)', "2*mu_s*E(u) + lambda_solid*Trace(E(u))*Id(2)")

# CORRECTIVE TERM
md.add_macro("g_f(v,u)", "-rho_f*nu_f*( Inv(F(u))'*(Grad(v))' )")

# MESH MOTION
md.add_macro('Mesh_def(u)', "Grad(u)")

######################
## FLUID EQUATIONS  ##
######################

md.add_nonlinear_term(mim_fluid, "rho_f*J(u_f)*(Grad_v_f.(Inv(F(u_f))*v_f)).Test_v_f", FLUID)
md.add_nonlinear_term(mim_fluid, "( J(u_f)*( sigma_f_p(p_f)+sigma_f_vu(v_f,u_f) )*(Inv(F(u_f)))' ):Grad_Test_v_f", FLUID)
md.add_nonlinear_term(mim_fluid, "-(g_f(v_f,u_f)*Normal).Test_v_f", OUTLET)

md.add_nonlinear_term(mim_fluid, "J(u_f)*Trace( (Grad(v_f))*Inv(F(u_f)) ).Test_p_f", FLUID)

md.add_nonlinear_term(mim_fluid, "Mesh_def(u_f):Grad_Test_u_f", FLUID)

#####################
## SOLID EQUATIONS ##
#####################

md.add_nonlinear_term(mim_solid, "( F(u_s)*Sigma_s(u_s) ): Grad_Test_u_s", BEAM)

#########################
## COUPLING CONDITIONS ##
#########################

# Kinematic coupling
md.add_nonlinear_term(mim_fluid, "(u_f - Interpolate(u_s, solid_to_fluid)).Test_mult", BEAM_INTERFACE)
md.add_nonlinear_term(mim_fluid, "mult.Test_u_f", BEAM_INTERFACE)

# Dynamic Coupling
md.add_nonlinear_term(mim_solid,
    f"-(J(Interpolate(u_f,fluid_to_solid))*(sigma_f_p(Interpolate(p_f,fluid_to_solid)) + " \
    "sigma_f_vu(Interpolate(v_f,fluid_to_solid),Interpolate(u_f,fluid_to_solid)) )*Inv(F(Interpolate(u_f,fluid_to_solid)))'*Normal).Test_u_s",
    BEAM_INTERFACE)
md.add_nonlinear_term(mim_fluid,
    "(F(Interpolate(u_s, solid_to_fluid))*Sigma_s(Interpolate(u_s, solid_to_fluid))*Normal).Test_v_f",
    BEAM_INTERFACE)

#########################
## BOUNDARY CONDITIONS ##
#########################

V_inlet_expr = "[4*1.5*U_mean*X(2)*(H-X(2))/(H*H), 0]"
V_inlet = md.interpolation(V_inlet_expr, mfv_fluid)
md.add_initialized_fem_data('V_inlet', mfv_fluid, V_inlet)
md.add_nonlinear_term(mim_fluid, "1e9*(v_f - V_inlet).Test_v_f", INLET)

md.add_nonlinear_term(mim_fluid, "1e9*v_f.Test_v_f", WALLS)
md.add_nonlinear_term(mim_fluid, "1e9*v_f.Test_v_f", CYLINDER)
md.add_nonlinear_term(mim_fluid, "1e10*v_f.Test_v_f", BEAM_INTERFACE)

md.add_nonlinear_term(mim_fluid, "1e9*u_f.Test_u_f", WALLS)
md.add_nonlinear_term(mim_fluid, "1e9*u_f.Test_u_f", CYLINDER)
md.add_nonlinear_term(mim_fluid, "1e9*u_f.Test_u_f", INLET)
md.add_nonlinear_term(mim_fluid, "1e9*u_f.Test_u_f", OUTLET)

md.add_nonlinear_term(mim_solid, "1e10*u_s.Test_u_s", BEAM_LEFT)

####################
## MODEL SOLUTION ##
####################

# Count DOFs
n_uf = mfu_fluid.nbdof()
n_vf = mfv_fluid.nbdof()
n_pf = mfp_fluid.nbdof()
n_us = mfu_solid.nbdof()
n_mult = len(md.variable("mult"))
total_dofs = n_uf + n_vf + n_pf + n_us + n_mult

log("=" * 60)
log("Degrees of Freedom")
log("=" * 60)
log(f"  Fluid mesh displacement (u_f): {n_uf}")
log(f"  Fluid velocity          (v_f): {n_vf}")
log(f"  Pressure                (p_f): {n_pf}")
log(f"  Solid displacement      (u_s): {n_us}")
log(f"  Lagrange multiplier    (mult): {n_mult}")
log(f"  ─────────────────────────────────")
log(f"  Total DOFs:                     {total_dofs}")
log("")

log("=" * 60)
log("FEM Information")
log("=" * 60)
log(f"  Velocity/Displacement FEM: FEM_QK(2,2) (Q2)")
log(f"  Pressure FEM:              FEM_QK(2,1) (Q1)")
log(f"  Integration:               IM_QUAD(9)")
log("")

log("=" * 60)
log("Starting FSI-1 Static Analysis (Two Mesh)")
log("=" * 60)

md.solve("very noisy",
         "lsolver", "superlu")

log("")
log("Solve completed successfully!")
log("=" * 60)

# Extract solutions
u_f = md.variable("u_f")
v_f = md.variable("v_f")
p = md.variable("p_f")
u_s = md.variable("u_s")

######################
## POST-PROCESSING ##
######################

# Force computation model
md_force = gf.Model("real")
md_force.add_fem_data("u_force", mfu_fluid)
md_force.add_fem_data("v_force", mfv_fluid)
md_force.add_fem_data("p_force", mfp_fluid)
md_force.add_initialized_data("nu_f", ν_fluid)
md_force.add_initialized_data("rho_f", rho_fluid)

md_force.add_macro("F(u)", "Id(2)+Grad(u)")
md_force.add_macro("J(u)", "Det(F(u))")
md_force.add_macro("sigma_f_vu(v,u)",
    "rho_f*nu_f*(Grad(v)*Inv(F(u)) + (Inv(F(u)))'*(Grad(v))')")
md_force.add_macro("sigma_f_p(p)", "-p*Id(2)")

md_force.set_variable("u_force", u_f.copy())
md_force.set_variable("v_force", v_f.copy())
md_force.set_variable("p_force", p.copy())

# Cylinder traction
traction_cyl = gf.asm_generic(mim_fluid, 0,
    "(J(u_force)*(sigma_f_p(p_force) + sigma_f_vu(v_force, u_force))"
    "*Inv(F(u_force))'*Normal)",
    CYLINDER, md_force)

# Beam interface traction
traction_beam = gf.asm_generic(mim_fluid, 0,
    "(J(u_force)*(sigma_f_p(p_force) + sigma_f_vu(v_force, u_force))"
    "*Inv(F(u_force))'*Normal)",
    BEAM_INTERFACE, md_force)

F_D = -(traction_cyl[0] + traction_beam[0])
F_L = -(traction_cyl[1] + traction_beam[1])

# Displacement at point A
A = np.array([0.6, 0.2])
result = gf.compute_interpolate_on(mfu_solid, u_s, A)
u_s_at_A_x = float(result[0])
u_s_at_A_y = float(result[1])

# Pressure difference
p_front_point = np.array([[0.15], [0.2]])
p_back_point = np.array([[0.25], [0.2]])
p_front = gf.compute_interpolate_on(mfp_fluid, p, p_front_point)[0]
p_back = gf.compute_interpolate_on(mfp_fluid, p, p_back_point)[0]
p_diff = p_front - p_back

log("")
log("=" * 60)
log("FSI-1 Benchmark Results")
log("=" * 60)
log(f"Displacement at A = ({A[0]}, {A[1]}):")
log(f"  u_x(A) = {u_s_at_A_x:.10e}")
log(f"  u_y(A) = {u_s_at_A_y:.10e}")
log("")
log(f"Forces (Wick eq. 254-255):")
log(f"  F_D (drag) = {F_D:.10e}")
log(f"  F_L (lift) = {F_L:.10e}")
log("")
log(f"  Cylinder traction:  [{traction_cyl[0]:.10e}, {traction_cyl[1]:.10e}]")
log(f"  Interface traction: [{traction_beam[0]:.10e}, {traction_beam[1]:.10e}]")
log("")
log(f"Pressure difference:")
log(f"  p(0.15, 0.2)                 = {p_front:.10e}")
log(f"  p(0.25, 0.2)                 = {p_back:.10e}")
log(f"  p(0.15,0.2) - p(0.25,0.2)   = {p_diff:.10e}")
log("")
log("=" * 60)
log("Reference Values (Wick Table 9, finest mesh)")
log("=" * 60)
log(f"  u_x(A) = 2.2703e-05   (computed: {u_s_at_A_x:.4e})")
log(f"  u_y(A) = 8.1809e-04   (computed: {u_s_at_A_y:.4e})")
log(f"  F_D    = 15.3776       (computed: {F_D:.4f})")
log(f"  F_L    = 0.74111       (computed: {F_L:.5f})")
log("")

#######################
## EXPORTING RESULTS ##
#######################

mfu_fluid.export_to_vtu(f"{output_dir}/fluid_static.vtu",
                        mfu_fluid, u_f, "MeshDisplacement",
                        mfv_fluid, v_f, "Velocity",
                        mfp_fluid, p, "Pressure")

mfu_solid.export_to_vtu(f"{output_dir}/solid_static.vtu",
                        mfu_solid, u_s, "Displacement")

log(f"✓ Fluid results exported to {output_dir}/fluid_static.vtu")
log(f"✓ Solid results exported to {output_dir}/solid_static.vtu")
log(f"✓ Log saved to {output_dir}/results_log.txt")
log("")
log("=" * 60)
log("Analysis complete!")
log("=" * 60)

log_file.close()