import getfem as gf 
from FSI.Functions import verify_regions
import numpy as np
import os
import sys
from datetime import datetime

gf.util_trace_level(1)
gf.util_warning_level(1)
π = np.pi

######################################
# Static Fluid-Structure Interaction
# Single-mesh monolithic ALE formulation
######################################

output_dir = "FSI/FSI_Benchmark_I_Results_1mesh"
os.makedirs(output_dir, exist_ok=True)

# Open log file
log_file = open(f"{output_dir}/results_log.txt", "w")

def log(msg=""):
    """Print to console and write to log file."""
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

log(f"FSI-1 Benchmark — Single Mesh Monolithic ALE")
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

# Fluid properties
ν_fluid = 0.001
rho_fluid = 1000.0

# Structural properties
rho_solid = 1000.0
nu_solid = 0.4
mu_solid = 5e+5
E = 2 * mu_solid * (1 + nu_solid)
lambda_solid = E * nu_solid / ((1 + nu_solid) * (1 - 2 * nu_solid))

# Inlet velocity parameters
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

Mesh = gf.Mesh('Import', 'gmsh', 'FSI/FSI_I_BENCHMARK/MESH_GMSH/TF_1MESH_quads.msh')

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

Mesh.region_merge(FLUID, 28)
Mesh.region_merge(FLUID, 29)
Mesh.region_merge(FLUID, 30)
Mesh.region_merge(FLUID, 31)
Mesh.region_merge(FLUID, 32)
Mesh.region_merge(FLUID, 33)
Mesh.region_merge(FLUID, 34)
Mesh.region_merge(FLUID, 35)
Mesh.region_merge(FLUID, 36)
Mesh.region_merge(FLUID, 37)
Mesh.region_merge(FLUID, 38)
Mesh.region_merge(FLUID, 39)
Mesh.region_merge(FLUID, 40)
Mesh.region_merge(FLUID, 41)
Mesh.region_merge(FLUID, 42)
Mesh.region_merge(FLUID, 43)
Mesh.region_merge(FLUID, 44)
Mesh.region_merge(FLUID, 46)
Mesh.region_merge(FLUID, 47)

Mesh.region_merge(BEAM, 45)
Mesh.region_merge(BEAM, 51)

Mesh.region_merge(WALLS, 1)
Mesh.region_merge(WALLS, 10)
Mesh.region_merge(WALLS, 11)
Mesh.region_merge(WALLS, 12)
Mesh.region_merge(WALLS, 13)
Mesh.region_merge(WALLS, 2)
Mesh.region_merge(WALLS, 3)
Mesh.region_merge(WALLS, 4)

Mesh.region_merge(CYLINDER, 17)
Mesh.region_merge(CYLINDER, 18)
Mesh.region_merge(CYLINDER, 19)
Mesh.region_merge(CYLINDER, 20)
Mesh.region_merge(CYLINDER, 21)

Mesh.region_merge(BEAM_INTERFACE, 23)
Mesh.region_merge(BEAM_INTERFACE, 24)
Mesh.region_merge(BEAM_INTERFACE, 25)
Mesh.region_merge(BEAM_INTERFACE, 26)
Mesh.region_merge(BEAM_INTERFACE, 27)

Mesh.region_merge(BEAM_LEFT, 22)

Mesh.region_merge(OUTLET, 5)
Mesh.region_merge(OUTLET, 6)
Mesh.region_merge(OUTLET, 7)
Mesh.region_merge(OUTLET, 8)
Mesh.region_merge(OUTLET, 9)

Mesh.region_merge(INLET, 14)
Mesh.region_merge(INLET, 15)
Mesh.region_merge(INLET, 16)

BEAM_INTERFACE_FLUID = 209
BEAM_INTERFACE_SOLID = 210

# Get the (cvid, face) pairs stored in each volume region
fluid_region = Mesh.region(FLUID)
beam_region = Mesh.region(BEAM)

# Extract unique convex IDs
fluid_cv_list = np.unique(fluid_region[0])
beam_cv_list = np.unique(beam_region[0])

# Get outer faces of each subdomain
fluid_outer = Mesh.outer_faces(fluid_cv_list)
beam_outer = Mesh.outer_faces(beam_cv_list)

# Create one-sided interface regions
Mesh.set_region(BEAM_INTERFACE_FLUID, fluid_outer)
Mesh.region_intersect(BEAM_INTERFACE_FLUID, BEAM_INTERFACE)

Mesh.set_region(BEAM_INTERFACE_SOLID, beam_outer)
Mesh.region_intersect(BEAM_INTERFACE_SOLID, BEAM_INTERFACE)

log("=" * 60)
log("Mesh Information")
log("=" * 60)
log(f"  Total points:       {Mesh.nbpts()}")
log(f"  Total convexes:     {Mesh.nbcvs()}")
log(f"  Fluid convexes:     {len(fluid_cv_list)}")
log(f"  Beam convexes:      {len(beam_cv_list)}")
log(f"  Fluid outer faces:  {fluid_outer.shape[1]}")
log(f"  Beam outer faces:   {beam_outer.shape[1]}")
log("")

########################
## INTEGRATION METHOD ##
########################

mim = gf.MeshIm(Mesh, gf.Integ("IM_QUAD(5)"))

#########################
## FEM ELEMENTS ##
#########################

mfu_fluid = gf.MeshFem(Mesh, 2)
mfu_fluid.set_fem(gf.Fem('FEM_QK(2,2)'))

mfv_fluid = gf.MeshFem(Mesh, 2)
mfv_fluid.set_fem(gf.Fem('FEM_QK(2,2)'))

mfp_fluid = gf.MeshFem(Mesh, 1)
mfp_fluid.set_fem(gf.Fem('FEM_QK(2,1)'))

mfu_solid = gf.MeshFem(Mesh, 2)
mfu_solid.set_fem(gf.Fem('FEM_QK(2,2)'))

###########
## MODEL ##
###########

md = gf.Model("real")

###################
## FEM VARIABLES ##
###################

md.add_filtered_fem_variable("u_f", mfu_fluid, FLUID)
md.add_filtered_fem_variable("v_f", mfv_fluid, FLUID)
md.add_filtered_fem_variable("p_f", mfp_fluid, FLUID)
md.add_filtered_fem_variable("u_s", mfu_solid, BEAM)

# Lagrange Multiplier for displacement coupling
md.add_filtered_fem_variable("mult", mfu_fluid, BEAM_INTERFACE_FLUID)

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

#################################################
## WEAK FORMULATION (Thomas Wick Problem 7.28) ##
#################################################

########### MACROS ###########

# COMMON
md.add_macro("F(u)", "Id(2)+Grad(u)")
md.add_macro("J(u)", "Det(F(u))")

# FLUID STRESS TENSORS
md.add_macro('sigma_f_vu(v,u)', "rho_f*nu_f*(Grad(v)*Inv(F(u)) + (Inv(F(u)))'*(Grad(v))')")
md.add_macro('sigma_f_p(p)', "-p*Id(2)")

# SOLID STRESS TENSORS
md.add_macro("E(u)", "0.5*((F(u))'*F(u) - Id(2))")
md.add_macro('Sigma_s(u)', "2*mu_s*E(u) + lambda_solid*Trace(E(u))*Id(2)")
md.add_macro("PK1(u)", "F(u)*Sigma_s(u)")

# CORRECTIVE TERM
md.add_macro("g_f(v,u)", "-rho_f*nu_f*( Inv(F(u))'*(Grad(v))' )")

# MESH MOTION
md.add_macro('Mesh_def(u)', "Grad(u)")

######################
## FLUID EQUATIONS  ##
######################

md.add_nonlinear_term(mim,
    "rho_f*J(u_f)*(Grad(v_f).(Inv(F(u_f))*v_f)).Test_v_f",
    FLUID)

md.add_nonlinear_term(mim,
    "(J(u_f)*(sigma_f_p(p_f) + sigma_f_vu(v_f, u_f))*(Inv(F(u_f)))'):Grad_Test_v_f",
    FLUID)

md.add_nonlinear_term(mim,
    "-(g_f(v_f, u_f)*Normal).Test_v_f",
    OUTLET)

md.add_nonlinear_term(mim,
    "J(u_f)*Trace(Grad(v_f)*Inv(F(u_f)))*Test_p_f",
    FLUID)

md.add_nonlinear_term(mim,
    "Mesh_def(u_f):Grad_Test_u_f",
    FLUID)

#####################
## SOLID EQUATIONS ##
#####################

md.add_nonlinear_term(mim,
    "(PK1(u_s)):Grad_Test_u_s",
    BEAM)

#########################
## COUPLING CONDITIONS ##
#########################

# Kinematic coupling
md.add_nonlinear_term(mim,
    "(u_f - u_s).Test_mult",
    BEAM_INTERFACE_FLUID)
md.add_nonlinear_term(mim,
    "mult.Test_u_f",
    BEAM_INTERFACE_FLUID)

# Verify normals
n_fluid_side = gf.asm_generic(mim, 0, "Normal", BEAM_INTERFACE_FLUID)
n_solid_side = gf.asm_generic(mim, 0, "Normal", BEAM_INTERFACE_SOLID)

log("=" * 60)
log("Normal Verification")
log("=" * 60)
log(f"  Fluid-side normal integral: {n_fluid_side}")
log(f"  Solid-side normal integral: {n_solid_side}")
log(f"  Sum (should be ~0):         {n_fluid_side + n_solid_side}")
log("")

# Dynamic coupling: fluid traction on solid
md.add_nonlinear_term(mim,
    "((J(u_f)*(sigma_f_p(p_f) + sigma_f_vu(v_f, u_f))*(Inv(F(u_f)))')*Normal).Test_u_s",
    BEAM_INTERFACE_FLUID)

# # Dynamic coupling: solid traction on fluid
# md.add_nonlinear_term(mim,
#     "-((PK1(u_s))*Normal).Test_v_f",
#     BEAM_INTERFACE_SOLID)

# #########################
# ## BOUNDARY CONDITIONS (pen.) ##
# remark: when b.c conditions are imposed through penalization
# it seems like superlu works better than mumps
# #########################

# V_inlet_expr = "[4*1.5*U_mean*X(2)*(H-X(2))/(H*H), 0]"
# V_inlet = md.interpolation(V_inlet_expr, mfv_fluid)
# md.add_initialized_fem_data('V_inlet', mfv_fluid, V_inlet)
# md.add_nonlinear_term(mim, "1e12*(v_f - V_inlet).Test_v_f", INLET)

# md.add_nonlinear_term(mim, "1e12*v_f.Test_v_f", WALLS)
# md.add_nonlinear_term(mim, "1e12*v_f.Test_v_f", CYLINDER)
# md.add_nonlinear_term(mim, "1e12*v_f.Test_v_f", BEAM_INTERFACE_FLUID)

# md.add_nonlinear_term(mim, "1e12*u_f.Test_u_f", WALLS)
# md.add_nonlinear_term(mim, "1e12*u_f.Test_u_f", CYLINDER)
# md.add_nonlinear_term(mim, "1e12*u_f.Test_u_f", INLET)
# md.add_nonlinear_term(mim, "1e12*u_f.Test_u_f", OUTLET)

# md.add_nonlinear_term(mim, "1e12*u_s.Test_u_s", BEAM_LEFT)

#########################
## BOUNDARY CONDITIONS (mult.) ##
# remark: when b.c conditions are imposed with multipliers
# is necessary to use mumps (superlu doens't work)
#########################

# INLET: Parabolic velocity profile
V_inlet_expr = "[4*1.5*U_mean*X(2)*(H-X(2))/(H*H), 0]"
V_inlet = md.interpolation(V_inlet_expr, mfv_fluid)
md.add_initialized_fem_data('V_inlet', mfv_fluid, V_inlet)

# --- Velocity Dirichlet BCs (Lagrange multipliers) ---
md.add_Dirichlet_condition_with_multipliers(mim, "v_f", mfv_fluid, INLET, "V_inlet")
md.add_Dirichlet_condition_with_multipliers(mim, "v_f", mfv_fluid, WALLS)
md.add_Dirichlet_condition_with_multipliers(mim, "v_f", mfv_fluid, CYLINDER)
md.add_Dirichlet_condition_with_multipliers(mim, "v_f", mfv_fluid, BEAM_INTERFACE_FLUID)

# --- Mesh displacement Dirichlet BCs (Lagrange multipliers) ---
md.add_Dirichlet_condition_with_multipliers(mim, "u_f", mfu_fluid, WALLS)
md.add_Dirichlet_condition_with_multipliers(mim, "u_f", mfu_fluid, CYLINDER)
md.add_Dirichlet_condition_with_multipliers(mim, "u_f", mfu_fluid, INLET)
md.add_Dirichlet_condition_with_multipliers(mim, "u_f", mfu_fluid, OUTLET)

# --- Solid: Fixed left boundary ---
md.add_Dirichlet_condition_with_multipliers(mim, "u_s", mfu_solid, BEAM_LEFT)

####################
## MODEL SOLUTION ##
####################

# Count filtered DOFs
n_uf = len(md.variable("u_f"))
n_vf = len(md.variable("v_f"))
n_pf = len(md.variable("p_f"))
n_us = len(md.variable("u_s"))
n_mult = len(md.variable("mult"))
total_dofs = n_uf + n_vf + n_pf + n_us + n_mult

log("=" * 60)
log("Degrees of Freedom (Filtered)")
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
log(f"  Integration:               IM_QUAD(5)")
log("")



md.solve("very noisy",
         "max_iter", 100,
         "max_res", 1e-8,
         "lsolver", "mumps",
         "lsearch", "simplest")



log("")
log("=" * 60)
log("Solve completed successfully!")
log("=" * 60)

# Extract solutions
u_f = md.variable("u_f")
v_f = md.variable("v_f")
p_f = md.variable("p_f")
u_s = md.variable("u_s")

######################
## POST-PROCESSING ##
######################

# Use model interpolation for proper DOF mapping
u_f_full = md.interpolation("u_f", mfu_fluid)
v_f_full = md.interpolation("v_f", mfv_fluid)
p_f_full = md.interpolation("p_f", mfp_fluid)
u_s_full = md.interpolation("u_s", mfu_solid)

# ---- Force computation (Wick eq. 254-255) ----
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

md_force.set_variable("u_force", u_f_full)
md_force.set_variable("v_force", v_f_full)
md_force.set_variable("p_force", p_f_full)

# Cylinder traction
traction_cyl = gf.asm_generic(mim, 0,
    "(J(u_force)"
    "*(sigma_f_p(p_force) + sigma_f_vu(v_force, u_force))"
    "*Inv(F(u_force))'*Normal)",
    CYLINDER, md_force)

# Beam interface traction — FLUID SIDE
traction_beam = gf.asm_generic(mim, 0,
    "(J(u_force)"
    "*(sigma_f_p(p_force) + sigma_f_vu(v_force, u_force))"
    "*Inv(F(u_force))'*Normal)",
    BEAM_INTERFACE_FLUID, md_force)

F_D = -(traction_cyl[0] + traction_beam[0])
F_L = -(traction_cyl[1] + traction_beam[1])

# Displacement at point A
A = np.array([0.6, 0.2])
result = gf.compute_interpolate_on(mfu_solid, u_s_full, A)
u_s_at_A_x = float(result[0])
u_s_at_A_y = float(result[1])

# Pressure difference
p_front_point = np.array([[0.15], [0.2]])
p_back_point = np.array([[0.25], [0.2]])
p_front = gf.compute_interpolate_on(mfp_fluid, p_f_full, p_front_point)[0]
p_back = gf.compute_interpolate_on(mfp_fluid, p_f_full, p_back_point)[0]
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
log(f"  u_x(A) = (2.25 +-0.02)e-05   (computed: {u_s_at_A_x:.4e})")
log(f"  u_y(A) = (8.20+-0.05)e-04     (computed: {u_s_at_A_y:.4e})")
log(f"  F_D    = 15.3776       (computed: {F_D:.4f})")
log(f"  F_L    = 0.74111       (computed: {F_L:.5f})")
log("")

#########################
## EXPORT RESULTS ##
#########################

mfv_fluid.export_to_vtu(f"{output_dir}/fluid_results.vtu",
                        mfu_fluid, u_f_full, "MeshDisplacement",
                        mfv_fluid, v_f_full, "Velocity",
                        mfp_fluid, p_f_full, "Pressure")

mfu_solid.export_to_vtu(f"{output_dir}/solid_results.vtu",
                        mfu_solid, u_s_full, "Displacement")

log(f"✓ Fluid results exported to {output_dir}/fluid_results.vtu")
log(f"✓ Solid results exported to {output_dir}/solid_results.vtu")
log(f"✓ Log saved to {output_dir}/results_log.txt")
log("")
log("=" * 60)
log("Analysis complete!")
log("=" * 60)

log_file.close()