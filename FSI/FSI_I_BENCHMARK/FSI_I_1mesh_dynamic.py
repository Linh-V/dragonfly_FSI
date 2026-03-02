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
#   Global variables (turtleFSI style)
#   FSI-2 Benchmark (Turek & Hron)
##########################################

output_dir = "FSI/FSI_I_BENCHMARK/FSI_Benchmark_I_Results_biharmonic"
os.makedirs(output_dir, exist_ok=True)

# Open log file
log_file = open(f"{output_dir}/results_log.txt", "w")

def log(msg=""):
    """Print to console and write to log file."""
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

log(f"FSI-2 Benchmark — Single Mesh Dynamic (Global Variables, Biharmonic)")
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

# FSI-2: Structural properties
rho_solid = 1000.0
nu_solid = 0.4
mu_solid = 0.5e6
E = 2 * mu_solid * (1 + nu_solid)
lambda_solid = E * nu_solid / ((1 + nu_solid) * (1 - 2 * nu_solid))

# FSI-2: Inlet velocity
U_mean = 0.2

# Time stepping
dt = 0.1
dt_2 = 1
theta = 1
num_steps = 50
T = num_steps * dt

# Biharmonic mesh stiffness
alpha_mesh = 0.01

# Penalty for kinematic relation (turtleFSI uses 1e7)
delta = 1.0e7

# Ramp duration
t_ramp = 1.0

log("=" * 60)
log("Problem Parameters (FSI-I Benchmark)")
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
log("  Time stepping:")
log(f"    dt = {dt}, num_steps = {num_steps}, T = {T}")
log(f"    theta = {theta}")
log(f"    Ramp duration = {t_ramp} s")
log("")
log("  Mesh motion: BIHARMONIC")
log(f"    alpha_mesh = {alpha_mesh}")
log(f"    delta (kinematic penalty) = {delta}")
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

# One-sided interface regions (for force computation only)
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

mim = gf.MeshIm(Mesh, gf.Integ("IM_QUAD(9)"))

#########################
#    FEM ELEMENTS
#########################

# All defined on the WHOLE mesh (global variables)
mfu = gf.MeshFem(Mesh, 2)
mfu.set_fem(gf.Fem('FEM_QK(2,2)'))

mfv = gf.MeshFem(Mesh, 2)
mfv.set_fem(gf.Fem('FEM_QK(2,2)'))

mfp = gf.MeshFem(Mesh, 1)
mfp.set_fem(gf.Fem('FEM_QK(2,1)'))

mfw = gf.MeshFem(Mesh, 2)
mfw.set_fem(gf.Fem('FEM_QK(2,2)'))

###########
#  MODEL
###########

md = gf.Model("real")

###################
#  FEM VARIABLES
###################

# Global variables (NOT filtered, except pressure)
md.add_fem_variable("u", mfu)                      # displacement everywhere
md.add_fem_variable("v", mfv)                      # velocity everywhere
md.add_filtered_fem_variable("p", mfp, FLUID)      # pressure (fluid only)
md.add_filtered_fem_variable("w", mfw, FLUID)             # biharmonic auxiliary

# Previous time step data
md.add_fem_data("u_n", mfu)
md.add_fem_data("v_n", mfv)

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
md.add_initialized_data("alpha_mesh", alpha_mesh)
md.add_initialized_data("delta", delta)

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
md.add_macro("E_GL(u)", "0.5*((F(u))'*F(u) - Id(2))")
md.add_macro('Sigma_s(u)', "2*mu_s*E_GL(u) + lambda_solid*Trace(E_GL(u))*Id(2)")
md.add_macro("PK1(u)", "F(u)*Sigma_s(u)")

# CORRECTIVE TERM (do-nothing)
md.add_macro("g_f(v,u)", "-rho_f*nu_f*( Inv(F(u))'*(Grad(v))' )")

######################
#  FLUID EQUATIONS
#  Integrated over FLUID, tested by Test_v, Test_p
######################

# ==========================================
# A_T: TIME TERMS
# ==========================================
# Temporal: (1/k) * J^{n,theta} * rho_f * (v - v_n) · Test_v
md.add_nonlinear_term(mim,
    "theta0*(rho_f/dt)*J(u)*(v - v_n).Test_v",
    FLUID)
md.add_nonlinear_term(mim,
    "theta1*(rho_f/dt)*J(u_n)*(v - v_n).Test_v",
    FLUID)

# ALE correction
md.add_nonlinear_term(mim,
    "-(rho_f/dt)*J(u)*(Grad(v)*Inv(F(u))*(u - u_n)).Test_v",
    FLUID)

# ==========================================
# A_P: PRESSURE — FULLY IMPLICIT
# ==========================================
md.add_nonlinear_term(mim,
    "(J(u)*sigma_f_p(p)*(Inv(F(u)))'):Grad_Test_v",
    FLUID)

# ==========================================
# A_I: INCOMPRESSIBILITY — FULLY IMPLICIT
# ==========================================
md.add_nonlinear_term(mim,
    "J(u)*Trace(Grad(v)*Inv(F(u)))*Test_p",
    FLUID)

# ==========================================
# theta * A_E(U^n): TERMS AT TIME n
# ==========================================
# Convection at n
md.add_nonlinear_term(mim,
    "theta0*rho_f*J(u)*(Grad(v)*(Inv(F(u))*v)).Test_v",
    FLUID)

# Viscous stress at n
md.add_nonlinear_term(mim,
    "theta0*(J(u)*sigma_f_vu(v, u)*(Inv(F(u)))'):Grad_Test_v",
    FLUID)

# Do-nothing at n
md.add_nonlinear_term(mim,
    "-theta0*(g_f(v, u)*Normal).Test_v",
    OUTLET)

# ==========================================
# -(1-theta) * A_E(U^{n-1}): SOURCE TERMS
# ==========================================
# Convection at n-1
md.add_source_term(mim,
    "-theta1*rho_f*J(u_n)*(Grad(v_n)*(Inv(F(u_n))*v_n)).Test_v",
    FLUID)

# Viscous stress at n-1
md.add_source_term(mim,
    "-theta1*(J(u_n)*sigma_f_vu(v_n, u_n)*(Inv(F(u_n)))'):Grad_Test_v",
    FLUID)

# Do-nothing at n-1
md.add_source_term(mim,
    "theta1*(g_f(v_n, u_n)*Normal).Test_v",
    OUTLET)

# ==========================================
# BIHARMONIC MESH MOTION (turtleFSI convention)
# alpha * laplace^2(u) = 0 in FLUID
#
# Split:
#   Eq 1:  alpha*w - alpha*laplace(u) = 0   tested by Test_w
#   Eq 2:  alpha*laplace(w) = 0              tested by Test_u
#
# NO theta scheme — spatial constraint at each time step
# ==========================================

# --- Eq 1: w-definition (tested by Test_w) ---
#   Weak: alpha*w·Test_w - alpha*Grad(u):Grad(Test_w) = 0
md.add_nonlinear_term(mim,
    "alpha_mesh*w.Test_w",
    FLUID)
md.add_nonlinear_term(mim,
    "-alpha_mesh*Grad(u):Grad_Test_w",
    FLUID)

# --- Eq 2: mesh constraint (tested by Test_u) ---
#   Weak: alpha*Grad(w):Grad(Test_u) = 0
md.add_nonlinear_term(mim,
    "alpha_mesh*Grad(w):Grad_Test_u",
    FLUID)

#####################
#  SOLID EQUATIONS
#  Integrated over BEAM, tested by Test_v, Test_u
#####################

# ==========================================
# MOMENTUM (tested by Test_v)
# ==========================================
# Inertia
md.add_nonlinear_term(mim,
    "(rho_s/dt)*(v - v_n).Test_v",
    BEAM)

# Solid stress at n
md.add_nonlinear_term(mim,
    "theta0*(PK1(u)):Grad_Test_v",
    BEAM)

# Solid stress at n-1 (source)
md.add_source_term(mim,
    "-theta1*(PK1(u_n)):Grad_Test_v",
    BEAM)

# ==========================================
# KINEMATIC RELATION: du/dt = v
# (tested by Test_u, with PENALTY delta)
# This is the KEY ingredient from turtleFSI
# ==========================================

# (delta * rho_s / dt) * (u - u_n) · Test_u
md.add_nonlinear_term(mim,
    "delta*rho_s*(1/dt)*(u - u_n).Test_u",
    BEAM)

# -delta * rho_s * theta * v · Test_u
md.add_nonlinear_term(mim,
    "-delta*rho_s*theta0*v.Test_u",
    BEAM)

# delta * rho_s * (1-theta) * v_n · Test_u (source)
md.add_source_term(mim,
    "delta*rho_s*theta1*v_n.Test_u",
    BEAM)

# ==========================================
# NO INTERFACE COUPLING TERMS NEEDED
# Coupling is AUTOMATIC through shared
# global variables and shared test functions
# ==========================================

#########################
#  BOUNDARY CONDITIONS
#########################

# ---- Velocity BCs ----
V_inlet = md.interpolation("[0,0]", mfv)
md.add_initialized_fem_data('V_inlet', mfv, V_inlet)

md.add_Dirichlet_condition_with_multipliers(mim, "v", mfv, INLET, "V_inlet")
md.add_Dirichlet_condition_with_multipliers(mim, "v", mfv, WALLS)
md.add_Dirichlet_condition_with_multipliers(mim, "v", mfv, CYLINDER)

# ---- Displacement BCs (u = 0 on all external fluid boundaries) ----
md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, WALLS)
md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, CYLINDER)
md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, INLET)
md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, OUTLET)

# ---- Biharmonic auxiliary BCs (w = 0 on all external fluid boundaries) ----
md.add_Dirichlet_condition_with_multipliers(mim, "w", mfw, WALLS)
md.add_Dirichlet_condition_with_multipliers(mim, "w", mfw, CYLINDER)
md.add_Dirichlet_condition_with_multipliers(mim, "w", mfw, INLET)
md.add_Dirichlet_condition_with_multipliers(mim, "w", mfw, OUTLET)

# ---- Solid: fixed left boundary ----
md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, BEAM_LEFT)
md.add_Dirichlet_condition_with_multipliers(mim, "v", mfv, BEAM_LEFT)

#########################
#  INITIAL CONDITIONS
#########################

u_init = md.interpolation("[0,0]", mfu)
v_init = md.interpolation("[0,0]", mfv)

md.set_variable("u_n", u_init)
md.set_variable("v_n", v_init)

####################
#   DOF SUMMARY
####################

n_u = len(md.variable("u"))
n_v = len(md.variable("v"))
n_p = len(md.variable("p"))
n_w = len(md.variable("w"))
total_dofs = n_u + n_v + n_p + n_w

log("=" * 60)
log("Degrees of Freedom")
log("=" * 60)
log(f"  Displacement  (u):  {n_u}")
log(f"  Velocity      (v):  {n_v}")
log(f"  Pressure      (p):  {n_p}")
log(f"  Biharmonic    (w):  {n_w}")
log(f"  ─────────────────────────────────")
log(f"  Total DOFs:          {total_dofs}")
log("")

log("=" * 60)
log("FEM Information")
log("=" * 60)
log(f"  Velocity/Displacement FEM: FEM_QK(2,2) (Q2)")
log(f"  Pressure FEM:              FEM_QK(2,1) (Q1)")
log(f"  Biharmonic w FEM:          FEM_QK(2,2) (Q2)")
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
md_force.add_fem_data("u_force", mfu)
md_force.add_fem_data("v_force", mfv)
md_force.add_fem_data("p_force", mfp)
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
log("Starting FSI-2 Dynamic Analysis (Global Variables, Biharmonic)")
log("=" * 60)

export_every = 10
log_every = 10
progress = tqdm(desc="FSI-2 time stepping", total=num_steps)



for step in range(num_steps):
    progress.update(1)

    t = (step + 1) * dt

    # ---- Update inlet BC with smooth ramp ----
    if t < t_ramp:
        ramp = 0.5 * (1.0 - np.cos(π * t / t_ramp))
    else:
        ramp = 1.0
    if t == t_ramp:
        md.set_variable('dt', dt_2)
        

    V_inlet_expr = f"{ramp}*[4*1.5*U_mean*X(2)*(H-X(2))/(H*H), 0]"
    V_inlet = md.interpolation(V_inlet_expr, mfv)
    md.set_variable('V_inlet', V_inlet)

    # ---- Solve ----
    nbit, converged = md.solve("noisy",
                               "max_iter", 300,
                               "max_res", 1e-8,
                               "lsolver", "mumps",
                               "lsearch", "simplest")

    # ---- Extract solution ----
    u_sol = md.variable("u")
    v_sol = md.variable("v")
    p_sol = md.variable("p")
    w_sol = md.variable("w")

    # ---- Interpolate p to full MeshFem (filtered → full) ----
    p_full = md.interpolation("p", mfp)
    w_full = md.interpolation("w", mfw)

    # ---- Update previous time step data ----
    md.set_variable("u_n", u_sol.copy())
    md.set_variable("v_n", v_sol.copy())

    # ---- Displacement at point A ----
    result = gf.compute_interpolate_on(mfu, u_sol, A)
    u_Ax = float(result[0])
    u_Ay = float(result[1])

    # ---- Drag and lift ----
    md_force.set_variable("u_force", u_sol)
    md_force.set_variable("v_force", v_sol)
    md_force.set_variable("p_force", p_full)

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
    p_front = gf.compute_interpolate_on(mfp, p_full, p_front_point)[0]
    p_back  = gf.compute_interpolate_on(mfp, p_full, p_back_point)[0]
    p_diff = p_front - p_back

    # ---- Store history ----
    time_history.append(t)
    ux_history.append(u_Ax)
    uy_history.append(u_Ay)
    drag_history.append(F_D)
    lift_history.append(F_L)
    p_diff_history.append(p_diff)

    # ---- Log step results ----
    if step % log_every == 0 or step == num_steps - 1:
        log(f"")
        log(f"Step {step+1}/{num_steps}, t = {t:.4f} s (ramp = {ramp:.4f})")
        log(f"  Newton iters: {nbit}, converged: {converged}")
        log(f"  u_x(A) = {u_Ax:.8e},  u_y(A) = {u_Ay:.8e}")
        log(f"  F_D = {F_D:.6f},  F_L = {F_L:.6f},  dP = {p_diff:.6f}")
        log(f"  max|u| = {np.max(np.abs(u_sol)):.6e}")
        log(f"  max|v| = {np.max(np.abs(v_sol)):.6e}")
        log(f"  max|w| = {np.max(np.abs(w_sol)):.6e}")

   
    if step % export_every == 0 or step == num_steps - 1:

         # ---- Save histories periodically ----
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
        
        mfv.export_to_vtu(
        f"{output_dir}/fluid_and_solid_{step:06d}.vtu",
        mfu, u_sol, "Displacement",
        mfv, v_sol, "Velocity",
        mfp, p_full, "Pressure",
        mfw, w_full, "w_biharmonic")

        # ---- Save  restart files ----
        restart_data = {
        'model_state': md.from_variables(),
        'u_n': md.variable("u_n").copy(),
        'v_n': md.variable("v_n").copy(),
        'step': step,
        't': t,
        'time_history': time_history,
        'ux_history': ux_history,
        'uy_history': uy_history,
        'drag_history': drag_history,
        'lift_history': lift_history,
        'p_diff_history': p_diff_history,
        }
        np.save(f"{output_dir}/restart_{step:06d}.npy", restart_data, allow_pickle=True)


  

progress.close()

# =========================================================================
#                        FINAL OUTPUT
# =========================================================================

# Compute oscillation statistics from last 5 seconds
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
        periods = np.diff(t_osc[crossings[::2]])
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
mfv.export_to_vtu(f"{output_dir}/fluid_final.vtu",
    mfu, u_sol, "Displacement",
    mfv, v_sol, "Velocity",
    mfp, p_full, "Pressure")

mfu.export_to_vtu(f"{output_dir}/solid_final.vtu",
    mfu, u_sol, "Displacement",
    mfv, v_sol, "Velocity")

log(f"✓ Results exported to {output_dir}/")
log(f"✓ Log saved to {output_dir}/results_log.txt")
log("")
log("=" * 60)
log("Analysis complete!")
log("=" * 60)

log_file.close()