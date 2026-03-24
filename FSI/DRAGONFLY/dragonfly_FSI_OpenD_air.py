"""
Dragonfly Wing FSI Benchmark — MPI-Parallelized with Robust Restart
OPEN-DOMAIN VERSION:
  • Uniform (flat) inlet velocity profile  (same cosine ramp)
  • Slip walls  (symmetry / free-slip)
  • Wing chord  L_w = 0.45 m
  • Dragonfly-realistic stiff wing material  (E ~ 3 GPa, ν = 0.3)

  SCALING: all weak-form equations multiplied by dt
           → removes 1/dt factors from Jacobian diagonal
           → blocks scale as O(rho_f), O(rho_s), O(E*dt/L^2)
           → much better conditioned for large dt

Mesh: Dragonfly_FSI_openD.msh
Launch: mpirun --allow-run-as-root -n 4 python FSI/DRAGONFLY/dragonfly_FSI_OpenD_air.py

Physical groups (from .msh):
  Surface 1 → FLUID
  Surface 2 → WING_STRUCTURE
  Curve   3 → INLET
  Curve   4 → OUTLET
  Curve   5 → WALLS
  Curve   6 → WING_FSI_INTERFACE  (full wing perimeter)
  Curve   7 → WING_LEFT           (leading-edge clamp, line 53)
"""

import getfem as gf
import numpy as np
import os
import sys
from datetime import datetime
from FSI.Functions import verify_regions

# ==============================================================
#  MPI INITIALIZATION
# ==============================================================
try:
    from mpi4py import MPI
    comm   = MPI.COMM_WORLD
    rank   = comm.Get_rank()
    nprocs = comm.Get_size()
    HAS_MPI = True
except ImportError:
    comm    = None
    rank    = 0
    nprocs  = 1
    HAS_MPI = False

is_master = (rank == 0)

if is_master:
    from tqdm import tqdm
    gf.util_trace_level(1)
    gf.util_warning_level(1)
else:
    gf.util_trace_level(0)
    gf.util_warning_level(0)

π = np.pi

##########################################
#   Dragonfly Wing FSI — OPEN DOMAIN
#   PARALLEL + RESTART
##########################################

output_dir = "FSI/DRAGONFLY/Dragonfly_FSI_OpenDomain_Results_Re3000_air_ref"

if is_master:
    os.makedirs(output_dir, exist_ok=True)

if HAS_MPI:
    comm.Barrier()

# =====================================================================
#  RESTART CONFIGURATION
# =====================================================================
#RESTART_FROM = None
RESTART_FROM = "FSI/DRAGONFLY/Dragonfly_FSI_OpenDomain_Results_Re3000_air_ref/restart_009100.npy"

##################
#  PROBLEM DATA
##################

# ---- Domain geometry (must match the .msh) ----
r_cyl  = 0.05
l_beam = 0.35
L_w    = 2 * r_cyl + l_beam        # total wing chord = 0.45 m

L_ch = L_w * 14                    # domain length
H_ch = L_w * 8                     # domain height
dx_LE = 3 * L_w                    # leading-edge x-offset from inlet
dy_LE = 4 * L_w                    # wing centre-line y

# ---- Air at 20°C ----
rho_fluid = 1.2           # kg/m³
ν_fluid   = 1.5e-5        # m²/s (kinematic viscosity)
# ---- Water at 20°C ----
# rho_fluid = 1000         # kg/m³
# ν_fluid   = 1e-6         # m²/s (kinematic viscosity)

# ---- Solid material — dragonfly forewing (stiff vein composite) ----
rho_solid    = 1200.0               # density      [kg/m³]  (chitin)
nu_solid     = 0.3                  # Poisson ratio
E            = 3.0e9                # Young's modulus [Pa]  (3 GPa)
mu_solid     = E / (2.0 * (1.0 + nu_solid))
lambda_solid = E * nu_solid / ((1.0 + nu_solid) * (1.0 - 2.0 * nu_solid))

# ---- Inflow ----
Re     = 3000.0
U_mean = Re * ν_fluid / L_w
print(U_mean)

# ---- Time integration ----
dx_min = 0.001 # From ref mesh
cfl = 0.5
dt   = 0.1     # dt which aims for this cfl
dt2  = cfl*dx_min/(U_mean)                     # time step after ramp
theta     = 1                 # theta during ramp
theta2 = 0.501                    # theta after ramp
T         = 200*L_w/U_mean        # 200 times the convetive time
num_steps = int(T/dt2)

# ---- Inlet ramp ----
t_ramp = 2.0

# ---- ALE mesh smoothing ----
alpha_mesh = 0.01

# =====================================================================
#  FSI PENALTY — rescaled because equations are multiplied by dt
#
#  Original (unscaled):  delta * rho_s / dt * u  ~  E / L^2
#  After   *dt:          delta * rho_s      * u  ~  E * dt / L^2
#
#  So delta is unchanged in value — the dt factor moves into the
#  stress term instead.  We just need:
#
#    delta * rho_s  ~  E * dt / L^2
#    delta = E * dt / (rho_s * L_w^2)
# =====================================================================
delta = (E * dt) / (rho_solid * L_w**2)

if is_master:
    print("=" * 55)
    print("JACOBIAN BLOCK SCALES  (after multiplying by dt)")
    print("=" * 55)
    print(f"  Fluid inertia:      rho_f          = {rho_fluid:.3e}")
    print(f"  Fluid convection:   rho_f*U*dt     = {rho_fluid*U_mean*dt:.3e}")
    print(f"  Fluid viscous:      rho_f*nu_f*dt  = {rho_fluid*ν_fluid*dt:.3e}")
    print(f"  Structural inertia: rho_s          = {rho_solid:.3e}")
    print(f"  Structural stress:  E*dt/L^2       = {E*dt/L_w**2:.3e}")
    print(f"  FSI penalty:        delta*rho_s    = {delta*rho_solid:.3e}")
    print(f"  Penalty/Stress:     {delta*rho_solid / (E*dt/L_w**2):.3f}  (want 1)")
    print("=" * 55)

#############
#   MESH
#############

Mesh = gf.Mesh('Import', 'gmsh', 'FSI/MESH_GMSH/Dragonfly_FSI_openD_almost_fine_paper.msh')

#############
#  REGION TAGS
#############

INLET          = 3
OUTLET         = 4
WALLS          = 5
WING_FSI       = 6
WING_LEFT      = 7
FLUID          = 1
WING_STRUCTURE = 2

WING_FSI_FLUID = 209
WING_FSI_SOLID = 210

fluid_region  = Mesh.region(FLUID)
wing_region   = Mesh.region(WING_STRUCTURE)
fluid_cv_list = np.unique(fluid_region[0])
wing_cv_list  = np.unique(wing_region[0])

fluid_outer = Mesh.outer_faces(fluid_cv_list)
wing_outer  = Mesh.outer_faces(wing_cv_list)

Mesh.set_region(WING_FSI_FLUID, fluid_outer)
Mesh.region_intersect(WING_FSI_FLUID, WING_FSI)

Mesh.set_region(WING_FSI_SOLID, wing_outer)
Mesh.region_intersect(WING_FSI_SOLID, WING_FSI)

Mesh.region_subtract(WING_FSI_FLUID, WING_LEFT)
Mesh.region_subtract(WING_FSI_SOLID, WING_LEFT)

########################
#  INTEGRATION METHOD
########################

mim = gf.MeshIm(Mesh, gf.Integ("IM_QUAD(9)"))

#########################
#    FEM ELEMENTS
#########################

mfu = gf.MeshFem(Mesh, 2)
mfu.set_fem(gf.Fem('FEM_QK(2,2)'))

mfv = gf.MeshFem(Mesh, 2)
mfv.set_fem(gf.Fem('FEM_QK(2,2)'))

mfp = gf.MeshFem(Mesh, 1)
mfp.set_fem(gf.Fem('FEM_QK(2,1)'))

# mfw = gf.MeshFem(Mesh, 2)
# mfw.set_fem(gf.Fem('FEM_QK(2,2)'))

###########
#  MODEL
###########

md = gf.Model("real")

md.add_fem_variable("u", mfu)
md.add_fem_variable("v", mfv)
md.add_filtered_fem_variable("p", mfp, FLUID)
#md.add_filtered_fem_variable("w", mfw, FLUID)

md.add_fem_data("u_n", mfu)
md.add_fem_data("v_n", mfv)

md.add_initialized_data("rho_f",        rho_fluid)
md.add_initialized_data("nu_f",         ν_fluid)
md.add_initialized_data("lambda_solid", lambda_solid)
md.add_initialized_data("mu_s",         mu_solid)
md.add_initialized_data("rho_s",        rho_solid)
md.add_initialized_data("H",            H_ch)
md.add_initialized_data("U_mean",       U_mean)
md.add_initialized_data("dt",           dt)
md.add_initialized_data("theta0",       theta)
md.add_initialized_data("theta1",       1.0 - theta)
md.add_initialized_data("alpha_mesh",   alpha_mesh)
md.add_initialized_data("delta",        delta)

#################################################
#  WEAK FORMULATION — MULTIPLIED BY dt
#
#  Original:   (rho_f/dt) * J(u) * (v-v_n) · φ  +  rho_f * convection  +  viscous  +  pressure = 0
#  × dt:        rho_f     * J(u) * (v-v_n) · φ  +  rho_f*dt * convection + dt*viscous + dt*pressure = 0
#
#  Block scales AFTER ×dt (all dt-independent or proportional):
#    Fluid inertia:      rho_f          ~ O(1.2)
#    Fluid convection:   rho_f*U*dt     ~ O(0.012)
#    Fluid viscous:      rho_f*nu*dt/L² ~ O(1e-6)  (small at Re=3000, correct)
#    Fluid pressure:     dt/L           ~ O(0.22)
#    Struct inertia:     rho_s          ~ O(1200)
#    Struct stress:      E*dt/L²        ~ O(1.48e9) ← still large vs fluid
#    FSI penalty:        delta*rho_s    ~ O(1.48e9) ← matches struct stress ✓
#
#  The fluid/solid scale gap (rho_f vs rho_s) is physical (m* = 1000)
#  and cannot be removed without full non-dimensionalization.
#  But at least 1/dt no longer amplifies the imbalance.
#################################################

md.add_macro("F(u)",    "Id(2)+Grad(u)")
md.add_macro("J(u)",    "Det(F(u))")
md.add_macro("sigma_f_vu(v,u)",
    "rho_f*nu_f*(Grad(v)*Inv(F(u)) + (Inv(F(u)))'*(Grad(v))')")
md.add_macro("sigma_f_p(p)",  "-p*Id(2)")
md.add_macro("E_GL(u)",
    "0.5*((F(u))'*F(u) - Id(2))")
md.add_macro("Sigma_s(u)",
    "2*mu_s*E_GL(u) + lambda_solid*Trace(E_GL(u))*Id(2)")
md.add_macro("PK1(u)",  "F(u)*Sigma_s(u)")
md.add_macro("g_f(v,u)",
    "-rho_f*nu_f*( Inv(F(u))'*(Grad(v))' )")

# =====================================================================
#  FLUID MOMENTUM  (× dt)
#
#  Before ×dt:   (rho_f/dt)*J(u)*(v-v_n)
#  After  ×dt:    rho_f    *J(u)*(v-v_n)
# =====================================================================

# ---- Inertia: rho_f * J(u) * (v - v_n)  [was rho_f/dt * ...] ----
md.add_nonlinear_term(mim,
    "theta0*rho_f*J(u)*(v - v_n).Test_v", FLUID)
md.add_nonlinear_term(mim,
    "theta1*rho_f*J(u_n)*(v - v_n).Test_v", FLUID)

# ---- ALE correction: -rho_f * J(u) * Grad(v)*F^{-1}*(u-u_n)  [was -rho_f/dt * ...] ----
md.add_nonlinear_term(mim,
    "-rho_f*J(u)*(Grad(v)*Inv(F(u))*(u - u_n)).Test_v", FLUID)

# ---- Pressure:  dt * J(u) * sigma_f_p * F^{-T} : Grad_Test_v ----
md.add_nonlinear_term(mim,
    "dt*(J(u)*sigma_f_p(p)*(Inv(F(u)))'):Grad_Test_v", FLUID)

# ---- Continuity: dt * J(u) * tr(Grad(v)*F^{-1}) = 0 ----
md.add_nonlinear_term(mim,
    "dt*J(u)*Trace(Grad(v)*Inv(F(u)))*Test_p", FLUID)

# ---- Convection: dt * rho_f * J(u) * Grad(v)*F^{-1}*v ----
md.add_nonlinear_term(mim,
    "theta0*dt*rho_f*J(u)*(Grad(v)*(Inv(F(u))*v)).Test_v", FLUID)

# ---- Viscous: dt * J(u) * sigma_f_vu * F^{-T} : Grad_Test_v ----
md.add_nonlinear_term(mim,
    "theta0*dt*(J(u)*sigma_f_vu(v, u)*(Inv(F(u)))'):Grad_Test_v", FLUID)

# ---- Outflow natural BC: dt * g_f · Normal ----
md.add_nonlinear_term(mim,
    "-theta0*dt*(g_f(v, u)*Normal).Test_v", OUTLET)

# ---- Previous-step source terms (×dt) ----
md.add_source_term(mim,
    "-theta1*dt*rho_f*J(u_n)*(Grad(v_n)*(Inv(F(u_n))*v_n)).Test_v", FLUID)
md.add_source_term(mim,
    "-theta1*dt*(J(u_n)*sigma_f_vu(v_n, u_n)*(Inv(F(u_n)))'):Grad_Test_v", FLUID)
md.add_source_term(mim,
    "theta1*dt*(g_f(v_n, u_n)*Normal).Test_v", OUTLET)

# =====================================================================
#  BIHARMONIC ALE MESH SMOOTHING  (×dt)
#  Original form has no 1/dt, so ×dt just scales the whole system
#  — keep as is for now (alpha_mesh already a tuning parameter)
# =====================================================================
# md.add_nonlinear_term(mim, "alpha_mesh*w.Test_w",             FLUID)
# md.add_nonlinear_term(mim, "-alpha_mesh*Grad(u):Grad_Test_w", FLUID)
# md.add_nonlinear_term(mim, "alpha_mesh*Grad(w):Grad_Test_u",  FLUID)
# =====================================================================
#  LAPLACIAN ALE MESH SMOOTHING
#  Replaces the 3-term biharmonic system with a single equation:
#
#  α * ∇u : ∇φ = 0   in fluid domain
#
#  Alpha variants:
#    constant:       "alpha_mesh*Grad(u):Grad_Test_u"
#    volume_change:  "(alpha_mesh/J(u))*Grad(u):Grad_Test_u"
# =====================================================================


md.add_nonlinear_term(mim,
        "alpha_mesh*Grad(u):Grad_Test_u", FLUID)

# =====================================================================
#  STRUCTURAL DYNAMICS  (×dt)
#
#  Before ×dt:   (rho_s/dt)*(v-v_n)  +  theta0*PK1 : Grad_φ
#  After  ×dt:    rho_s    *(v-v_n)  +  theta0*dt*PK1 : Grad_φ
# =====================================================================

# ---- Inertia: rho_s*(v-v_n)  [was rho_s/dt * ...] ----
md.add_nonlinear_term(mim,
    "rho_s*(v - v_n).Test_v", WING_STRUCTURE)

# ---- Stress: dt * theta0 * PK1 : Grad_Test_v ----
md.add_nonlinear_term(mim,
    "theta0*dt*(PK1(u)):Grad_Test_v", WING_STRUCTURE)
md.add_source_term(mim,
    "-theta1*dt*(PK1(u_n)):Grad_Test_v", WING_STRUCTURE)

# =====================================================================
#  ALE KINEMATIC COUPLING  (×dt)
#
#  Before ×dt:   delta*rho_s/dt*(u-u_n)  -  delta*rho_s*theta0*v
#  After  ×dt:   delta*rho_s   *(u-u_n)  -  delta*rho_s*theta0*dt*v
# =====================================================================

# ---- Kinematic: delta*rho_s*(u-u_n)  [was delta*rho_s/dt * ...] ----
md.add_nonlinear_term(mim,
    "delta*rho_s*(u - u_n).Test_u", WING_STRUCTURE)

# ---- Coupling: -delta*rho_s*theta0*dt*v ----
md.add_nonlinear_term(mim,
    "-delta*rho_s*theta0*dt*v.Test_u", WING_STRUCTURE)
md.add_source_term(mim,
    "delta*rho_s*theta1*dt*v_n.Test_u", WING_STRUCTURE)

# =============================================================
#  BOUNDARY CONDITIONS  (unchanged)
# =============================================================

# ---- Inlet: uniform velocity (updated every step) ----
V_inlet = md.interpolation("[0,0]", mfv)
md.add_initialized_fem_data('V_inlet', mfv, V_inlet)
md.add_Dirichlet_condition_with_multipliers(mim, "v", mfv, INLET, "V_inlet")

# ---- Walls: FREE-SLIP — only v·n = 0 ----
md.add_normal_Dirichlet_condition_with_multipliers(mim, "v", 1, WALLS)

# ---- Leading-edge clamp: full zero on v, u, w ----
md.add_Dirichlet_condition_with_multipliers(mim, "v", mfv, WING_LEFT)
md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, WING_LEFT)


# ---- Walls/Inlet/Outlet: ALE mesh fixed, biharmonic helper zero ----
md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, WALLS)
md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, INLET)
md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, OUTLET)

# md.add_Dirichlet_condition_with_multipliers(mim, "w", mfw, WING_LEFT)
# md.add_Dirichlet_condition_with_multipliers(mim, "w", mfw, WALLS)
# md.add_Dirichlet_condition_with_multipliers(mim, "w", mfw, INLET)
# md.add_Dirichlet_condition_with_multipliers(mim, "w", mfw, OUTLET)

# =====================================================================
#  RESTART LOADING
# =====================================================================

restart_data_loaded = None
if RESTART_FROM is not None:
    if not os.path.exists(RESTART_FROM):
        if is_master:
            print(f"ERROR: Restart file not found: {RESTART_FROM}")
        if HAS_MPI:
            comm.Barrier()
        sys.exit(1)
    restart_data_loaded = np.load(RESTART_FROM, allow_pickle=True).item()
    if is_master:
        print(f"Loaded restart file: {RESTART_FROM}")
        print(f"  Keys: {list(restart_data_loaded.keys())}")

is_restart = (restart_data_loaded is not None)

# ---- Log file ----
log_file = None
if is_master:
    log_mode = "a" if is_restart else "w"
    log_file = open(f"{output_dir}/results_log.txt", log_mode)

def log(msg=""):
    if is_master:
        print(msg)
        if log_file is not None:
            log_file.write(msg + "\n")
            log_file.flush()

if is_restart:
    has_individual  = ('u_n' in restart_data_loaded and
                       'v_n' in restart_data_loaded)
    has_model_state = 'model_state' in restart_data_loaded

    log("")
    log("=" * 60)
    log("RESTART LOADING")
    log("=" * 60)
    log(f"  File: {RESTART_FROM}")
    log(f"  Step: {restart_data_loaded['step']}")
    log(f"  Time: {restart_data_loaded['t']:.6f} s")
    log(f"  Has individual u_n/v_n: {has_individual}")
    log(f"  Has model_state:        {has_model_state}")
    log(f"  MPI processes:          {nprocs}")

    if has_individual:
        saved_u_n = restart_data_loaded['u_n']
        saved_v_n = restart_data_loaded['v_n']

        expected_u_size = len(md.variable("u"))
        expected_v_size = len(md.variable("v"))

        log(f"  Expected u size: {expected_u_size}, saved: {len(saved_u_n)}")
        log(f"  Expected v size: {expected_v_size}, saved: {len(saved_v_n)}")

        if (len(saved_u_n) != expected_u_size or
                len(saved_v_n) != expected_v_size):
            log("  ERROR: Size mismatch — mesh or FEM order changed!")
            if HAS_MPI:
                comm.Barrier()
            sys.exit(1)

        md.set_variable("u_n", saved_u_n)
        md.set_variable("v_n", saved_v_n)
        md.set_variable("u",   saved_u_n.copy())
        md.set_variable("v",   saved_v_n.copy())

        log(f"  ✓ Restored u_n, v_n on all {nprocs} process(es)")
        log(f"  max|u_n| = {np.max(np.abs(saved_u_n)):.6e}")
        log(f"  max|v_n| = {np.max(np.abs(saved_v_n)):.6e}")

    elif has_model_state:
        log("  WARNING: Using model_state fallback")
        try:
            md.to_variables(restart_data_loaded['model_state'])
            log(f"  ✓ Restored model_state on all {nprocs} process(es)")
        except Exception as e:
            log(f"  ✗ model_state FAILED: {e}")
            is_restart = False
    else:
        log("  ✗ No usable restart data!")
        is_restart = False

    if is_restart:
        start_step = restart_data_loaded['step'] + 1

        if is_master:
            time_history   = list(restart_data_loaded.get('time_history',   []))
            ux_history     = list(restart_data_loaded.get('ux_history',     []))
            uy_history     = list(restart_data_loaded.get('uy_history',     []))
            drag_history   = list(restart_data_loaded.get('drag_history',   []))
            lift_history   = list(restart_data_loaded.get('lift_history',   []))
            p_front_history = list(restart_data_loaded.get('p_front_history',[]))
            p_back_history  = list(restart_data_loaded.get('p_back_history',[]))
            velocity_vx_history = list(restart_data_loaded.get('velocity_vx_history', []))
            velocity_vy_history = list(restart_data_loaded.get('velocity_vy_history', []))
    
            log(f"  Histories restored: {len(time_history)} entries")

        log(f"  Resuming from step {start_step}")
        log("=" * 60)
        log("")

if not is_restart:
    u_init = md.interpolation("[0,0]", mfu)
    v_init = md.interpolation("[0,0]", mfv)
    md.set_variable("u_n", u_init)
    md.set_variable("v_n", v_init)

    start_step = 0

    if is_master:
        time_history    = []
        ux_history      = []
        uy_history      = []
        drag_history    = []
        lift_history    = []
        p_front_history = []
        p_back_history  = []
        velocity_vx_history = []
        velocity_vy_history = []

    log(f"Dragonfly Wing FSI (Open Domain, ×dt scaled) — Fresh Start  [{nprocs} procs]")
    log(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("")

if HAS_MPI:
    comm.Barrier()

# =====================================================================
#  LOGGING
# =====================================================================

log("=" * 60)
log("Problem Parameters — Open-Domain Dragonfly Wing FSI (×dt scaled)")
log("=" * 60)
log(f"  Domain:      L = {L_ch:.4f} m,  H = {H_ch:.4f} m")
log(f"  Wing chord:  L_w = {L_w:.4f} m")
log(f"  LE offset:   dx_LE = {dx_LE:.4f} m,  dy_LE = {dy_LE:.4f} m")
log("")
log(f"  Fluid:  rho = {rho_fluid} kg/m³,  nu (kinematic) = {ν_fluid} m²/s")
log(f"          mu  (dynamic)  = {rho_fluid*ν_fluid:.4e} Pa·s")
log(f"          Re  = {Re:.1f}  (U = {U_mean:.6f} m/s, L = {L_w} m)")
log("")
log(f"  Solid (dragonfly forewing):")
log(f"    rho = {rho_solid} kg/m³")
log(f"    E   = {E:.3e} Pa  ({E/1e9:.2f} GPa)")
log(f"    nu  = {nu_solid}")
log(f"    mu_s     = {mu_solid:.4e} Pa")
log(f"    lambda_s = {lambda_solid:.4e} Pa")
log("")
log(f"  Inlet:  UNIFORM U = {U_mean:.6f} m/s (cosine ramp, t_ramp = {t_ramp} s)")
log(f"  Walls:  FREE-SLIP  (v·n = 0,  v·t free,  u = 0,  w = 0)")
log("")
log(f"   num_steps = {num_steps},  T = {T} s")
log(f"  during ramp: theta  = {theta}, dt = {dt}, iteration ramp = {2/dt}")
log(f"  after ramp: theta2 = {theta2}, dt = cfl * d_xmin /(Uinf) ={dt2}, cfl = {cfl}, dx_min = {dx_min}")
log(f"  delta  = {delta:.4e}  (= E*dt/(rho_s*L_w^2))")
log(f"  MPI processes: {nprocs}")
log("")
log("Jacobian block scales (after ×dt):")
log(f"  rho_f            = {rho_fluid:.3e}  (fluid inertia)")
log(f"  rho_s            = {rho_solid:.3e}  (struct inertia)")
log(f"  E*dt/L^2         = {E*dt/L_w**2:.3e}  (struct stress)")
log(f"  delta*rho_s      = {delta*rho_solid:.3e}  (FSI penalty)")
log(f"  rho_f*U*dt       = {rho_fluid*U_mean*dt:.3e}  (convection)")

n_u = len(md.variable("u"))
n_v = len(md.variable("v"))
n_p = len(md.variable("p"))
#n_w = len(md.variable("w"))
log(f"  Total DOFs: {n_u +  n_v +  n_p }") # +n_w
log("")

#########################
#  TRACKING SETUP
#########################

# tip displacement
x_TE = dx_LE + 1.00 * L_w
y_TE = dy_LE + 0.025 * L_w
A    = np.array([x_TE, y_TE])

# pressure probe
p_front_point = np.array([[dx_LE - 0.02], [dy_LE]])
p_back_point  = np.array([[x_TE  + 0.02], [y_TE ]])

# velocity probe (Nastro's paper)
velocity_point = np.array([[dx_LE + 1.3 * L_w],
                        [dy_LE + 0.17 * L_w]])


# ---- Force sub-model (lightweight, all procs) ----
md_force = gf.Model("real")
md_force.add_fem_data("u_force", mfu)
md_force.add_fem_data("v_force", mfv)
md_force.add_fem_data("p_force", mfp)
md_force.add_initialized_data("nu_f",  ν_fluid)
md_force.add_initialized_data("rho_f", rho_fluid)

md_force.add_macro("F(u)",  "Id(2)+Grad(u)")
md_force.add_macro("J(u)",  "Det(F(u))")
md_force.add_macro("sigma_f_vu(v,u)",
    "rho_f*nu_f*(Grad(v)*Inv(F(u)) + (Inv(F(u)))'*(Grad(v))')")
md_force.add_macro("sigma_f_p(p)", "-p*Id(2)")

# NOTE: force assembly uses ORIGINAL (unscaled) dimensional stress
# so F_D, F_L are correct dimensional forces in [N/m]

####################
#   TIME STEPPING
####################

remaining_steps = num_steps - start_step

if remaining_steps <= 0:
    log(f"Nothing to do: start_step={start_step} >= num_steps={num_steps}")
    if is_master and log_file:
        log_file.close()
    if HAS_MPI:
        comm.Barrier()
    sys.exit(0)

log("=" * 60)
if is_restart:
    log(f"Resuming from step {start_step},  {remaining_steps} steps left")
else:
    log("Starting Open-Domain Dragonfly Wing FSI Dynamic Analysis (×dt scaled)")
log("=" * 60)

export_every = 100
log_every    = 10

if is_master:
    progress = tqdm(desc=f"Dragonfly FSI open [{nprocs} procs]",
                    total=remaining_steps)

# =====================================================================
#  TIME INITIALIZATION
# =====================================================================
if is_restart:
    t = restart_data_loaded['t']  # Load from restart
else:
    t = 0.0  # Fresh start

for step in range(start_step, num_steps):
    if is_master:
        progress.update(1)

    t = t + dt

    # ---- Cosine ramp — UNIFORM profile ----
    if t < t_ramp:
        ramp = 0.5 * (1.0 - np.cos(π * t / t_ramp))
    else:
        
        ramp = 1.0
        md.set_variable('theta0', np.array([theta2]))
        md.set_variable('theta1', np.array([1.0 - theta2]))
        
        if dt != dt2:  # Only update once
            dt = dt2
            delta = (E * dt) / (rho_solid * L_w**2)  # Recompute!
            md.set_variable('dt', np.array([dt]))
            md.set_variable('delta', np.array([delta]))
            if is_master:
                log(f">>> Switched to dt = {dt}, delta = {delta:.4e}")
    U_ramp = ramp * U_mean

    V_inlet_expr = f"[{U_ramp}, 0]"
    V_inlet = md.interpolation(V_inlet_expr, mfv)
    md.set_variable('V_inlet', V_inlet)

    # ---- Nonlinear solve ----
    if step < 50:
        res_tol  = 1e-5
        max_iter = 500
    else:
        res_tol  = 1e-8
        max_iter = 300

    nbit, converged = md.solve(
        "noisy",
        "max_iter", max_iter,
        "max_res",  res_tol,
        "lsolver",  "mumps",
        "lsearch",  "simplest"
    )
   

    # ---- Extract solution ----
    u_sol  = md.variable("u")
    v_sol  = md.variable("v")
    p_full = md.interpolation("p", mfp)
    #w_full = md.interpolation("w", mfw)

    # ---- Advance time step ----
    md.set_variable("u_n", u_sol.copy())
    md.set_variable("v_n", v_sol.copy())

    # ---- Aerodynamic forces (all procs) ----
    # Uses UNSCALED stress → correct dimensional forces
    md_force.set_variable("u_force", u_sol)
    md_force.set_variable("v_force", v_sol)
    md_force.set_variable("p_force", p_full)

    traction_wing = gf.asm_generic(
        mim, 0,
        "(J(u_force)"
        " * (sigma_f_p(p_force) + sigma_f_vu(v_force, u_force))"
        " * Inv(F(u_force))' * Normal)",
        WING_FSI_FLUID,
        md_force
    )

    F_D = -traction_wing[0]
    F_L = -traction_wing[1]

    # ---- Post-processing (master only) ----
    if is_master:

        # displacement
        result = gf.compute_interpolate_on(mfu, u_sol, A)
        u_Ax = float(result[0])
        u_Ay = float(result[1])

        # pressure probe
        p_front = float(gf.compute_interpolate_on(mfp, p_full, p_front_point)[0])
        p_back  = float(gf.compute_interpolate_on(mfp, p_full, p_back_point)[0])

        # velocity probe
        velocity_result = gf.compute_interpolate_on(mfv, v_sol, velocity_point)
        velocity_vx = float(velocity_result[0])   # longitudinal component
        velocity_vy = float(velocity_result[1])  
        
        time_history.append(t)
        ux_history.append(u_Ax)
        uy_history.append(u_Ay)
        drag_history.append(F_D)
        lift_history.append(F_L)
        p_front_history.append(p_front)
        p_back_history.append(p_back)
        velocity_vx_history.append(velocity_vx)
        velocity_vy_history.append(velocity_vy)

        if step % log_every == 0 or step == num_steps - 1:
            log("")
            log(f"Step {step+1}/{num_steps},  t = {t:.4f} s  "
                f"(ramp = {ramp:.4f},  U_inlet = {U_ramp:.6f} m/s)")
            log(f"  Newton iters: {nbit},  converged: {converged}")
            log(f"  Trailing-edge tip: "
                f"u_x = {u_Ax:.8e} m,  u_y = {u_Ay:.8e} m")
            log(f"  Aero forces:  F_D = {F_D:.6f} N,  F_L = {F_L:.6f} N")
            log(f"  p_front = {p_front:.6f} Pa,  "
                f"p_back  = {p_back:.6f} Pa,  "
                f"ΔP = {p_front - p_back:.6f} Pa")
            log(f"  max|u| = {np.max(np.abs(u_sol)):.6e}")
            log(f"  max|v| = {np.max(np.abs(v_sol)):.6e}")
            #log(f"  max|w| = {np.max(np.abs(w_full)):.6e}")

        if step % export_every == 0 or step == num_steps - 1:
            np.savetxt(
                f"{output_dir}/displacement_history.txt",
                np.column_stack([time_history, ux_history, uy_history]),
                header="Time  u_x_TE  u_y_TE",
                fmt='%.10e'
            )

            # SAFE VERSION — slices time to match velocity length
            if len(velocity_vx_history) > 0:
                N_vel    = len(velocity_vx_history)
                time_vel = time_history[-N_vel:]      # last N entries of time
                np.savetxt(
                    f"{output_dir}/velocity_history.txt",
                    np.column_stack([time_vel,
                                    velocity_vx_history,
                                    velocity_vy_history]),
                    header="Time  v_x_wake  v_y_wake",
                    fmt='%.10e'
                )
            np.savetxt(
                f"{output_dir}/force_history.txt",
                np.column_stack([time_history, drag_history,
                                 lift_history, p_front_history, p_back_history]),
                header="Time  F_D  F_L p_front p_back",
                fmt='%.10e'
            )



            mfv.export_to_vtu(
                f"{output_dir}/dragonfly_FSI_{step:06d}.vtu",
                mfu, u_sol,  "Displacement",
                mfv, v_sol,  "Velocity",
                mfp, p_full, "Pressure",
                #mfw, w_full, "w_biharmonic"
            )

            restart_dict = {
                'u_n':             md.variable("u_n").copy(),
                'v_n':             md.variable("v_n").copy(),
                'u':               u_sol.copy(),
                'v':               v_sol.copy(),
                'p':               p_full.copy(),
                #'w':               w_full.copy(),
                'model_state':     md.from_variables(),
                'step':            step,
                't':               t,
                'time_history':    time_history,
                'ux_history':      ux_history,
                'uy_history':      uy_history,
                'drag_history':    drag_history,
                'lift_history':    lift_history,
                'p_front_history': p_front_history,
                'p_back_history':  p_back_history,
                'velocity_vx_history': velocity_vx_history,
                'velocity_vy_history': velocity_vy_history,
            }
            restart_path = f"{output_dir}/restart_{step:06d}.npy"
            np.save(restart_path, restart_dict, allow_pickle=True)
            log(f"  >> Saved checkpoint: {restart_path}")

    if HAS_MPI:
        comm.Barrier()

if is_master:
    progress.close()

if HAS_MPI:
    comm.Barrier()