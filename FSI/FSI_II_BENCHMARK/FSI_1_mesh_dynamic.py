import getfem as gf
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

gf.util_trace_level(1)
gf.util_warning_level(1)
π = np.pi

##########################################
#   Dynamic Fluid-Structure Interaction
#   Single-mesh monolithic ALE formulation
#   Biharmonic mesh motion
#   FSI-2 Benchmark (Turek & Hron)
##########################################

output_dir = "FSI/FSI_II_BENCHMARK/FSI_Benchmark_II_Results_biharmonic"
os.makedirs(output_dir, exist_ok=True)

# Open log file
log_file = open(f"{output_dir}/results_log.txt", "w")

def log(msg=""):
    """Print to console and write to log file."""
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

log(f"FSI-2 Benchmark — Single Mesh Dynamic (Biharmonic Mesh Motion)")
log(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("")

##################
#  PROBLEM DATA
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

# ===== FSI-2: Structural properties (changed from FSI-1) =====
rho_solid = 10000.0         # FSI-2: 10x denser than fluid (was 1000 in FSI-1)
nu_solid = 0.4
mu_solid = 0.5e6            # Same as FSI-1
E = 2 * mu_solid * (1 + nu_solid)
lambda_solid = E * nu_solid / ((1 + nu_solid) * (1 - 2 * nu_solid))

# ===== FSI-2: Inlet velocity (changed from FSI-1) =====
U_mean = 1.0                # FSI-2: 5x higher than FSI-1 (was 0.2)

# ===== FSI-2: Real transient time stepping =====
dt = 0.001                  # Small time step for accuracy
theta = 0.5 + dt            # Slightly implicit (Crank-Nicolson + stabilization)
num_steps = 15000           # 15 seconds total
T = num_steps * dt

# Biharmonic mesh stiffness parameter
alpha_mesh = 1.0            # Scaling parameter for biharmonic operator

# Ramp duration
t_ramp = 2.0               # Smooth ramp-up over 2 seconds

log("=" * 60)
log("Problem Parameters (FSI-2 Benchmark)")
log("=" * 60)
log(f"  Channel:   L = {L}, H = {H}")
log(f"  Cylinder:  center = ({c_x}, {c_y}), r = {r}")
log(f"  Beam:      L = {L_beam}, W = {W_beam}")
log(f"  Fluid:     rho = {rho_fluid}, nu = {ν_fluid}")
log(f"  Solid:     rho = {rho_solid}, E = {E}, nu = {nu_solid}")
log(f"             mu_s = {mu_solid}, lambda_s = {lambda_solid}")
log(f"  Inlet:     U_mean = {U_mean}")
log(f"  Re = {2 * rho_fluid * U_mean * H / (3 * rho_fluid * ν_fluid):.1f}")
log("")
log("  Time stepping (real transient):")
log(f"    dt = {dt}, num_steps = {num_steps}, T = {T}")
log(f"    theta = {theta}")
log(f"    Ramp duration = {t_ramp} s")
log("")
log("  Mesh motion: BIHARMONIC (split into two 2nd-order equations)")
log(f"    alpha_mesh = {alpha_mesh}")
log("")

#############
#   MESH
#############

Mesh = gf.Mesh('Import', 'gmsh', 'FSI/MESH_GMSH/TF_1MESH_quads.msh')

#############
#  REGIONS
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

# One-sided interface regions
BEAM_INTERFACE_FLUID = 209
BEAM_INTERFACE_SOLID = 210

fluid_region = Mesh.region(FLUID)
beam_region = Mesh.region(BEAM)

fluid_cv_list = np.unique(fluid_region[0])
beam_cv_list = np.unique(beam_region[0])

fluid_outer = Mesh.outer_faces(fluid_cv_list)
beam_outer = Mesh.outer_faces(beam_cv_list)

Mesh.set_region(BEAM_INTERFACE_FLUID, fluid_outer)
Mesh.region_intersect(BEAM_INTERFACE_FLUID, BEAM_INTERFACE)

Mesh.set_region(BEAM_INTERFACE_SOLID, beam_outer)
Mesh.region_intersect(BEAM_INTERFACE_SOLID, BEAM_INTERFACE)

log("=" * 60)
log("Mesh Information")
log("=" * 60)
log(f"  Total points:      {Mesh.nbpts()}")
log(f"  Total convexes:    {Mesh.nbcvs()}")
log(f"  Fluid convexes:    {len(fluid_cv_list)}")
log(f"  Beam convexes:     {len(beam_cv_list)}")
log(f"  Fluid outer faces: {fluid_outer.shape[1]}")
log(f"  Beam outer faces:  {beam_outer.shape[1]}")
log("")

########################
#  INTEGRATION METHOD
########################

mim = gf.MeshIm(Mesh, gf.Integ("IM_QUAD(5)"))

#########################
#    FEM ELEMENTS
#########################

mfu_fluid = gf.MeshFem(Mesh, 2)
mfu_fluid.set_fem(gf.Fem('FEM_QK(2,2)'))

mfv_fluid = gf.MeshFem(Mesh, 2)
mfv_fluid.set_fem(gf.Fem('FEM_QK(2,2)'))

mfp_fluid = gf.MeshFem(Mesh, 1)
mfp_fluid.set_fem(gf.Fem('FEM_QK(2,1)'))

mfu_solid = gf.MeshFem(Mesh, 2)
mfu_solid.set_fem(gf.Fem('FEM_QK(2,2)'))

mfv_solid = gf.MeshFem(Mesh, 2)
mfv_solid.set_fem(gf.Fem('FEM_QK(2,2)'))

# ===== NEW: Auxiliary MeshFem for biharmonic variable w_f =====
mfw_fluid = gf.MeshFem(Mesh, 2)
mfw_fluid.set_fem(gf.Fem('FEM_QK(2,2)'))

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

###########
#  MODEL
###########

md = gf.Model("real")

###################
#  FEM VARIABLES
###################

# Current time step unknowns
md.add_filtered_fem_variable("u_f", mfu_fluid, FLUID)
md.add_filtered_fem_variable("v_f", mfv_fluid, FLUID)
md.add_filtered_fem_variable("p_f", mfp_fluid, FLUID)
md.add_filtered_fem_variable("u_s", mfu_solid, BEAM)
md.add_filtered_fem_variable("v_s", mfv_solid, BEAM)

# ===== NEW: Biharmonic auxiliary variable w_f (filtered to fluid) =====
md.add_filtered_fem_variable("w_f", mfw_fluid, FLUID)

# Previous time step data (full MeshFem, set via interpolation)
md.add_fem_data("u_f_n", mfu_fluid)
md.add_fem_data("v_f_n", mfv_fluid)
md.add_fem_data("u_s_n", mfu_solid)
md.add_fem_data("v_s_n", mfv_solid)

# ===== NEW: Previous time step data for w_f =====
md.add_fem_data("w_f_n", mfw_fluid)

# Lagrange multipliers for interface coupling
md.add_filtered_fem_variable("mult_u", mfu_fluid, BEAM_INTERFACE_FLUID)
md.add_filtered_fem_variable("mult_v", mfv_fluid, BEAM_INTERFACE_FLUID)

###########################
#  INITIALIZED CONSTANTS
###########################

md.add_initialized_data("rho_f", rho_fluid)
md.add_initialized_data("nu_f", ν_fluid)
md.add_initialized_data("lambda_solid", lambda_solid)
md.add_initialized_data("mu_s", mu_solid)
md.add_initialized_data("rho_s", rho_solid)
md.add_initialized_data("H", H)
md.add_initialized_data("U_mean", U_mean)
md.add_initialized_data("dt", dt)
md.add_initialized_data("theta0", theta)
md.add_initialized_data("theta1", 1.0 - theta)
md.add_initialized_data("alpha_mesh", alpha_mesh)  # NEW: biharmonic parameter

#################################################
#            WEAK FORMULATION
#################################################

########### MACROS ###########

# COMMON
md.add_macro("F(u)", "Id(2)+Grad(u)")
md.add_macro("J(u)", "Det(F(u))")

# FLUID STRESS TENSORS
md.add_macro('sigma_f_vu(v,u)',
    "rho_f*nu_f*(Grad(v)*Inv(F(u)) + (Inv(F(u)))'*(Grad(v))')")
md.add_macro('sigma_f_p(p)', "-p*Id(2)")

# SOLID STRESS TENSORS
md.add_macro("E(u)", "0.5*((F(u))'*F(u) - Id(2))")
md.add_macro('Sigma_s(u)', "2*mu_s*E(u) + lambda_solid*Trace(E(u))*Id(2)")
md.add_macro("PK1(u)", "F(u)*Sigma_s(u)")

# CORRECTIVE TERM
md.add_macro("g_f(v,u)", "-rho_f*nu_f*( Inv(F(u))'*(Grad(v))' )")

######################
#  FLUID EQUATIONS
######################

# ==========================================
# A_T: TIME TERMS
# ==========================================
# Fluid temporal: (1/k) * J^{n,theta} * rho_f * (v_f - v_f_n) · psi^v
md.add_nonlinear_term(mim,
    "theta0*(rho_f/dt)*J(u_f)*(v_f - v_f_n).Test_v_f",
    FLUID)
md.add_nonlinear_term(mim,
    "theta1*(rho_f/dt)*J(u_f_n)*(v_f - v_f_n).Test_v_f",
    FLUID)

# ALE correction
md.add_nonlinear_term(mim,
    "-(rho_f/dt)*J(u_f)*(Grad(v_f).Inv(F(u_f))*(u_f - u_f_n)).Test_v_f",
    FLUID)

# ==========================================
# A_P: PRESSURE — FULLY IMPLICIT
# ==========================================
md.add_nonlinear_term(mim,
    "(J(u_f)*sigma_f_p(p_f)*(Inv(F(u_f)))'):Grad_Test_v_f",
    FLUID)

# ==========================================
# A_I: INCOMPRESSIBILITY — FULLY IMPLICIT
# ==========================================
md.add_nonlinear_term(mim,
    "J(u_f)*Trace(Grad(v_f)*Inv(F(u_f)))*Test_p_f",
    FLUID)

# ==========================================
# theta * A_E(U^n): TERMS AT TIME n
# ==========================================
# Convection at n
md.add_nonlinear_term(mim,
    "theta0*rho_f*J(u_f)*(Grad(v_f).(Inv(F(u_f))*v_f)).Test_v_f",
    FLUID)

# Viscous stress at n
md.add_nonlinear_term(mim,
    "theta0*(J(u_f)*sigma_f_vu(v_f, u_f)*(Inv(F(u_f)))'):Grad_Test_v_f",
    FLUID)

# Do-nothing at n
md.add_nonlinear_term(mim,
    "-theta0*(g_f(v_f, u_f)*Normal).Test_v_f",
    OUTLET)

# ==========================================
# -(1-theta) * A_E(U^{n-1}): TERMS AT TIME n-1 (source)
# ==========================================
# Convection at n-1
md.add_source_term(mim,
    "-theta1*rho_f*J(u_f_n)*(Grad(v_f_n).(Inv(F(u_f_n))*v_f_n)).Test_v_f",
    FLUID)

# Viscous stress at n-1
md.add_source_term(mim,
    "-theta1*(J(u_f_n)*sigma_f_vu(v_f_n, u_f_n)*(Inv(F(u_f_n)))'):Grad_Test_v_f",
    FLUID)

# Do-nothing at n-1
md.add_source_term(mim,
    "theta1*(g_f(v_f_n, u_f_n)*Normal).Test_v_f",
    OUTLET)

# ==========================================
# BIHARMONIC MESH MOTION (replaces Laplacian)
# ==========================================
# The biharmonic equation alpha * laplace^2(u_f) = 0
# is split into two second-order equations:
#
#   Eq 1 (tested with Test_u_f):
#       alpha * w_f · Test_u_f - alpha * Grad(u_f) : Grad(Test_u_f) = 0
#
#   Eq 2 (tested with Test_w_f):
#       alpha * Grad(w_f) : Grad(Test_w_f) = 0
#
# With theta-scheme time discretization:

# --- Equation 1: w_f definition (tested with Test_u_f) ---
# theta * terms at n
md.add_nonlinear_term(mim,
    "theta0*alpha_mesh*w_f.Test_u_f",
    FLUID)
md.add_nonlinear_term(mim,
    "-theta0*alpha_mesh*Grad(u_f):Grad_Test_u_f",
    FLUID)

# (1-theta) * terms at n-1 (source / RHS)
md.add_source_term(mim,
    "-theta1*alpha_mesh*w_f_n.Test_u_f",
    FLUID)
md.add_source_term(mim,
    "theta1*alpha_mesh*Grad(u_f_n):Grad_Test_u_f",
    FLUID)

# --- Equation 2: Laplace(w_f) = 0 (tested with Test_w_f) ---
# theta * terms at n
md.add_nonlinear_term(mim,
    "theta0*alpha_mesh*Grad(w_f):Grad_Test_w_f",
    FLUID)

# (1-theta) * terms at n-1 (source / RHS)
md.add_source_term(mim,
    "-theta1*alpha_mesh*Grad(w_f_n):Grad_Test_w_f",
    FLUID)

#####################
#  SOLID EQUATIONS
#####################

# ==========================================
# A_T: TIME TERMS
# ==========================================
# Solid inertia
md.add_nonlinear_term(mim,
    "(rho_s/dt)*(v_s - v_s_n).Test_v_s",
    BEAM)

# Kinematic relation
md.add_nonlinear_term(mim,
    "(rho_s/dt)*(u_s - u_s_n).Test_u_s",
    BEAM)

# ==========================================
# theta * A_E(U^n): TERMS AT TIME n
# ==========================================
# Kinematic relation at n
md.add_nonlinear_term(mim,
    "-theta0*rho_s*v_s.Test_u_s",
    BEAM)

# Solid stress at n
md.add_nonlinear_term(mim,
    "theta0*(PK1(u_s)):Grad_Test_v_s",
    BEAM)

# ==========================================
# -(1-theta) * A_E(U^{n-1}): source terms
# ==========================================
# Kinematic relation at n-1
md.add_source_term(mim,
    "(theta1*rho_s*v_s_n).Test_u_s",
    BEAM)

# Solid stress at n-1
md.add_source_term(mim,
    "-theta1*(PK1(u_s_n)):Grad_Test_v_s",
    BEAM)

#########################
#  COUPLING CONDITIONS
#########################

# Kinematic coupling: u_f = u_s on interface
md.add_nonlinear_term(mim,
    "(u_f - u_s).Test_mult_u",
    BEAM_INTERFACE_FLUID)
md.add_nonlinear_term(mim,
    "mult_u.Test_u_f",
    BEAM_INTERFACE_FLUID)

# Kinematic coupling: v_f = v_s on interface
md.add_nonlinear_term(mim,
    "(v_f - v_s).Test_mult_v",
    BEAM_INTERFACE_FLUID)
md.add_nonlinear_term(mim,
    "mult_v.Test_v_f",
    BEAM_INTERFACE_FLUID)

# Dynamic coupling: fluid traction on solid
md.add_nonlinear_term(mim,
    "((J(u_f)*(sigma_f_p(p_f) + sigma_f_vu(v_f, u_f))"
    "*(Inv(F(u_f)))')*Normal).Test_v_s",
    BEAM_INTERFACE_FLUID)

# Dynamic coupling: solid traction on fluid
md.add_nonlinear_term(mim,
    "-((PK1(u_s))*Normal).Test_v_f",
    BEAM_INTERFACE_SOLID)

#########################
#  BOUNDARY CONDITIONS
#########################

# INLET: Parabolic velocity profile
V_inlet = md.interpolation("[0,0]", mfv_fluid)
md.add_initialized_fem_data('V_inlet', mfv_fluid, V_inlet)

md.add_Dirichlet_condition_with_multipliers(mim, "v_f", mfv_fluid, INLET, "V_inlet")
md.add_Dirichlet_condition_with_multipliers(mim, "v_f", mfv_fluid, WALLS)
md.add_Dirichlet_condition_with_multipliers(mim, "v_f", mfv_fluid, CYLINDER)

# Mesh displacement BCs (u_f = 0 on all external boundaries)
md.add_Dirichlet_condition_with_multipliers(mim, "u_f", mfu_fluid, WALLS)
md.add_Dirichlet_condition_with_multipliers(mim, "u_f", mfu_fluid, CYLINDER)
md.add_Dirichlet_condition_with_multipliers(mim, "u_f", mfu_fluid, INLET)
md.add_Dirichlet_condition_with_multipliers(mim, "u_f", mfu_fluid, OUTLET)

# ===== NEW: Biharmonic auxiliary variable BCs =====
# w_f = 0 on all external fluid boundaries
# This enforces d(grad(u_f))/dn = 0 in effect (smoothness at boundary)
md.add_Dirichlet_condition_with_multipliers(mim, "w_f", mfw_fluid, WALLS)
md.add_Dirichlet_condition_with_multipliers(mim, "w_f", mfw_fluid, CYLINDER)
md.add_Dirichlet_condition_with_multipliers(mim, "w_f", mfw_fluid, INLET)
md.add_Dirichlet_condition_with_multipliers(mim, "w_f", mfw_fluid, OUTLET)

# Solid BCs
md.add_Dirichlet_condition_with_multipliers(mim, "u_s", mfu_solid, BEAM_LEFT)
md.add_Dirichlet_condition_with_multipliers(mim, "v_s", mfv_solid, BEAM_LEFT)

#########################
#  INITIAL CONDITIONS
#########################

u_f_init = md.interpolation("[0,0]", mfu_fluid)
v_f_init = md.interpolation("[0,0]", mfv_fluid)
u_s_init = md.interpolation("[0,0]", mfu_solid)
v_s_init = md.interpolation("[0,0]", mfv_solid)
w_f_init = md.interpolation("[0,0]", mfw_fluid)  # NEW

md.set_variable("u_f_n", u_f_init)
md.set_variable("v_f_n", v_f_init)
md.set_variable("u_s_n", u_s_init)
md.set_variable("v_s_n", v_s_init)
md.set_variable("w_f_n", w_f_init)  # NEW

####################
#   DOF SUMMARY
####################

n_uf = len(md.variable("u_f"))
n_vf = len(md.variable("v_f"))
n_pf = len(md.variable("p_f"))
n_us = len(md.variable("u_s"))
n_vs = len(md.variable("v_s"))
n_wf = len(md.variable("w_f"))       # NEW
n_mult_u = len(md.variable("mult_u"))
n_mult_v = len(md.variable("mult_v"))
total_dofs = n_uf + n_vf + n_pf + n_us + n_vs + n_wf + n_mult_u + n_mult_v

log("=" * 60)
log("Degrees of Freedom (Filtered)")
log("=" * 60)
log(f"  Fluid mesh displacement (u_f):  {n_uf}")
log(f"  Fluid velocity          (v_f):  {n_vf}")
log(f"  Pressure                (p_f):  {n_pf}")
log(f"  Biharmonic auxiliary    (w_f):  {n_wf}")   # NEW
log(f"  Solid displacement      (u_s):  {n_us}")
log(f"  Solid velocity          (v_s):  {n_vs}")
log(f"  Multiplier u          (mu_u):   {n_mult_u}")
log(f"  Multiplier v          (mu_v):   {n_mult_v}")
log(f"  ─────────────────────────────────")
log(f"  Total DOFs:                      {total_dofs}")
log("")

log("=" * 60)
log("FEM Information")
log("=" * 60)
log(f"  Velocity/Displacement FEM: FEM_QK(2,2) (Q2)")
log(f"  Pressure FEM:              FEM_QK(2,1) (Q1)")
log(f"  Biharmonic w_f FEM:        FEM_QK(2,2) (Q2)")   # NEW
log(f"  Integration:               IM_QUAD(5)")
log("")

#########################
#  TRACKING & HISTORY
#########################

A = np.array([0.6, 0.2])

time_history = []
ux_history = []
uy_history = []
drag_history = []
lift_history = []
p_diff_history = []

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

p_front_point = np.array([[0.15], [0.2]])
p_back_point  = np.array([[0.25], [0.2]])

####################
#   TIME STEPPING
####################

log("=" * 60)
log("Starting FSI-2 Dynamic Analysis (Biharmonic Mesh Motion)")
log("=" * 60)

export_every = 100        # Export VTU every 100 steps (0.1 s)
log_every = 10            # Log to console every 10 steps
save_history_every = 1    # Save history arrays every step

progress = tqdm(desc="FSI-2 time stepping", total=num_steps)

for step in range(num_steps):
    progress.update(1)

    t = (step + 1) * dt

    # ---- Update inlet BC with smooth ramp ----
    if t < t_ramp:
        ramp = 0.5 * (1.0 - np.cos(π * t / t_ramp))
    else:
        ramp = 1.0

    V_inlet_expr = f"{ramp}*[4*1.5*U_mean*X(2)*(H-X(2))/(H*H), 0]"
    V_inlet = md.interpolation(V_inlet_expr, mfv_fluid)
    md.set_variable('V_inlet', V_inlet)

    # ---- Solve ----
    try:
        nbit, converged = md.solve("noisy",
                                    "max_iter", 200,
                                    "max_res", 1e-8,
                                    "lsolver", "mumps",
                                    "lsearch", "simplest")
    except Exception as e:
        log(f"  WARNING at step {step+1} (t={t:.4f}): {e}")
        log(f"  Trying with relaxed tolerance...")
        try:
            nbit, converged = md.solve("noisy",
                                        "max_iter", 500,
                                        "max_res", 1e-6,
                                        "lsolver", "mumps",
                                        "lsearch", "systematic")
        except Exception as e2:
            log(f"  FATAL at step {step+1} (t={t:.4f}): {e2}")
            break

    # ---- Extract solution (filtered) ----
    u_f_filt = md.variable("u_f")
    v_f_filt = md.variable("v_f")
    p_f_filt = md.variable("p_f")
    u_s_filt = md.variable("u_s")
    v_s_filt = md.variable("v_s")
    w_f_filt = md.variable("w_f")   # NEW

    # ---- Interpolate to full MeshFem ----
    u_f_full = md.interpolation("u_f", mfu_fluid)
    v_f_full = md.interpolation("v_f", mfv_fluid)
    p_f_full = md.interpolation("p_f", mfp_fluid)
    u_s_full = md.interpolation("u_s", mfu_solid)
    v_s_full = md.interpolation("v_s", mfv_solid)
    w_f_full = md.interpolation("w_f", mfw_fluid)   # NEW

    # ---- Update previous time step data ----
    md.set_variable("u_f_n", u_f_full)
    md.set_variable("v_f_n", v_f_full)
    md.set_variable("u_s_n", u_s_full)
    md.set_variable("v_s_n", v_s_full)
    md.set_variable("w_f_n", w_f_full)   # NEW

    # ---- Displacement at point A ----
    result = gf.compute_interpolate_on(mfu_solid, u_s_full, A)
    u_Ax = float(result[0])
    u_Ay = float(result[1])

    # ---- Drag and lift ----
    md_force.set_variable("u_force", u_f_full)
    md_force.set_variable("v_force", v_f_full)
    md_force.set_variable("p_force", p_f_full)

    traction_cyl = gf.asm_generic(mim, 0,
        "(J(u_force)"
        "*(sigma_f_p(p_force) + sigma_f_vu(v_force, u_force))"
        "*Inv(F(u_force))'*Normal)",
        CYLINDER, md_force)

    traction_beam = gf.asm_generic(mim, 0,
        "(J(u_force)"
        "*(sigma_f_p(p_force) + sigma_f_vu(v_force, u_force))"
        "*Inv(F(u_force))'*Normal)",
        BEAM_INTERFACE_FLUID, md_force)

    F_D = -(traction_cyl[0] + traction_beam[0])
    F_L = -(traction_cyl[1] + traction_beam[1])

    # ---- Pressure difference ----
    p_front = gf.compute_interpolate_on(mfp_fluid, p_f_full, p_front_point)[0]
    p_back  = gf.compute_interpolate_on(mfp_fluid, p_f_full, p_back_point)[0]
    p_diff = p_front - p_back

    # ---- Store history ----
    time_history.append(t)
    ux_history.append(u_Ax)
    uy_history.append(u_Ay)
    drag_history.append(F_D)
    lift_history.append(F_L)
    p_diff_history.append(p_diff)

    # ---- Log step results (periodic) ----
    if step % log_every == 0 or step == num_steps - 1:
        log(f"")
        log(f"Step {step+1}/{num_steps}, t = {t:.4f} s (ramp = {ramp:.4f})")
        log(f"  Newton iters: {nbit}")
        log(f"  u_x(A) = {u_Ax:.8e},  u_y(A) = {u_Ay:.8e}")
        log(f"  F_D = {F_D:.6f},  F_L = {F_L:.6f},  dP = {p_diff:.6f}")
        log(f"  max|u_s| = {np.max(np.abs(u_s_filt)):.6e}")
        log(f"  max|v_f| = {np.max(np.abs(v_f_filt)):.6e}")
        log(f"  max|w_f| = {np.max(np.abs(w_f_filt)):.6e}")

    # ---- Save histories periodically ----
    if step % 100 == 0 or step == num_steps - 1:
        np.savetxt(f"{output_dir}/displacement_history.txt",
                   np.column_stack([time_history, ux_history, uy_history]),
                   header="Time u_x(A) u_y(A)",
                   fmt='%.10e')

        np.savetxt(f"{output_dir}/force_history.txt",
                   np.column_stack([time_history, drag_history,
                                    lift_history, p_diff_history]),
                   header="Time F_D F_L Pressure_Diff",
                   fmt='%.10e')

    # ---- Export VTU ----
    if step % export_every == 0 or step == num_steps - 1:
        mfv_fluid.export_to_vtu(
            f"{output_dir}/fluid_{step:06d}.vtu",
            mfu_fluid, u_f_full, "MeshDisplacement",
            mfv_fluid, v_f_full, "Velocity",
            mfp_fluid, p_f_full, "Pressure")

        mfu_solid.export_to_vtu(
            f"{output_dir}/solid_{step:06d}.vtu",
            mfu_solid, u_s_full, "Displacement",
            mfv_solid, v_s_full, "Velocity")

progress.close()

# =========================================================================
#                        FINAL OUTPUT
# =========================================================================

# Compute oscillation statistics from last 5 seconds of data
t_analysis_start = T - 5.0
analysis_mask = np.array(time_history) >= t_analysis_start

if np.any(analysis_mask):
    ux_osc = np.array(ux_history)[analysis_mask]
    uy_osc = np.array(uy_history)[analysis_mask]
    drag_osc = np.array(drag_history)[analysis_mask]
    lift_osc = np.array(lift_history)[analysis_mask]
    t_osc = np.array(time_history)[analysis_mask]

    ux_mean = 0.5 * (np.max(ux_osc) + np.min(ux_osc))
    ux_amp  = 0.5 * (np.max(ux_osc) - np.min(ux_osc))
    uy_mean = 0.5 * (np.max(uy_osc) + np.min(uy_osc))
    uy_amp  = 0.5 * (np.max(uy_osc) - np.min(uy_osc))
    drag_mean = 0.5 * (np.max(drag_osc) + np.min(drag_osc))
    drag_amp  = 0.5 * (np.max(drag_osc) - np.min(drag_osc))
    lift_mean = 0.5 * (np.max(lift_osc) + np.min(lift_osc))
    lift_amp  = 0.5 * (np.max(lift_osc) - np.min(lift_osc))

    # Estimate frequency from u_y zero-crossings
    uy_centered = uy_osc - uy_mean
    crossings = np.where(np.diff(np.sign(uy_centered)))[0]
    if len(crossings) >= 2:
        periods = np.diff(t_osc[crossings[::2]])  # full periods
        if len(periods) > 0:
            freq_uy = 1.0 / np.mean(periods)
        else:
            freq_uy = float('nan')
    else:
        freq_uy = float('nan')

    log("")
    log("=" * 60)
    log("FSI-2 Benchmark Final Results (Oscillatory Regime)")
    log("=" * 60)
    log(f"Analysis window: t = [{t_analysis_start:.1f}, {T:.1f}] s")
    log("")
    log(f"Displacement at A = ({A[0]}, {A[1]}):")
    log(f"  u_x(A) = {ux_mean:.6e} ± {ux_amp:.6e}")
    log(f"  u_y(A) = {uy_mean:.6e} ± {uy_amp:.6e}")
    log("")
    log(f"Forces:")
    log(f"  F_D (drag) = {drag_mean:.4f} ± {drag_amp:.4f}")
    log(f"  F_L (lift) = {lift_mean:.4f} ± {lift_amp:.4f}")
    log("")
    log(f"Frequency (from u_y): {freq_uy:.4f} Hz")
    log("")
    log("=" * 60)
    log("Reference Values (Turek & Hron, FSI-2)")
    log("=" * 60)
    log(f"  u_x(A) = -14.58 ± 12.44 × 10^-3")
    log(f"           (computed: {ux_mean:.4e} ± {ux_amp:.4e})")
    log(f"  u_y(A) = 1.23 ± 80.6 × 10^-3")
    log(f"           (computed: {uy_mean:.4e} ± {uy_amp:.4e})")
    log(f"  F_D    = 457.3 ± 22.66")
    log(f"           (computed: {drag_mean:.2f} ± {drag_amp:.2f})")
    log(f"  F_L    = 2.22 ± 149.78")
    log(f"           (computed: {lift_mean:.2f} ± {lift_amp:.2f})")
    log(f"  f(u_y) ≈ 3.8 Hz  (computed: {freq_uy:.2f} Hz)")
    log("")

# ---- Final export ----
mfv_fluid.export_to_vtu(f"{output_dir}/fluid_final.vtu",
    mfu_fluid, u_f_full, "MeshDisplacement",
    mfv_fluid, v_f_full, "Velocity",
    mfp_fluid, p_f_full, "Pressure")

mfu_solid.export_to_vtu(f"{output_dir}/solid_final.vtu",
    mfu_solid, u_s_full, "Displacement",
    mfv_solid, v_s_full, "Velocity")

log(f"✓ Fluid results exported to {output_dir}/fluid_final.vtu")
log(f"✓ Solid results exported to {output_dir}/solid_final.vtu")
log(f"✓ Histories saved to {output_dir}/")
log(f"✓ Log saved to {output_dir}/results_log.txt")
log("")
log("=" * 60)
log("Analysis complete!")
log("=" * 60)

log_file.close()