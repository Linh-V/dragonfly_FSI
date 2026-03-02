import getfem as gf
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

gf.util_trace_level(1)
gf.util_warning_level(1)
π = np.pi

##########################################
#   FSI-2 Benchmark WITH ROBUST RESTART
##########################################

output_dir = "FSI/FSI_II_BENCHMARK/FSI_Benchmark_II_Results_biharmonic_mesh_ref"
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
#  RESTART CONFIGURATION
# =====================================================================
RESTART_FROM = "FSI/FSI_II_BENCHMARK/FSI_Benchmark_II_Results_biharmonic_mesh_ref/restart_009300.npy"
# Set to None to start fresh:
# RESTART_FROM = None
# =====================================================================

##################
#  PROBLEM DATA
##################

L = 2.5
H = 0.41
c_x = 0.2
c_y = 0.2
r = 0.05
L_beam = 0.35
W_beam = 0.02

ν_fluid = 0.001
rho_fluid = 1000.0

rho_solid = 10000.0
nu_solid = 0.4
mu_solid = 0.5e6
E = 2 * mu_solid * (1 + nu_solid)
lambda_solid = E * nu_solid / ((1 + nu_solid) * (1 - 2 * nu_solid))

U_mean = 1.0

dt = 0.001
theta = 0.5 + dt
num_steps = 15000
T = num_steps * dt

alpha_mesh = 0.01
delta = 1.0e7
t_ramp = 2.0

#############
#   MESH
#############

Mesh = gf.Mesh('Import', 'gmsh', 'FSI/MESH_GMSH/TF_1MESH_quads_ref.msh')

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

for r_id in [28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47]:
    Mesh.region_merge(FLUID, r_id)

for r_id in [45, 51]:
    Mesh.region_merge(BEAM, r_id)

for r_id in [1,10,11,12,13,2,3,4]:
    Mesh.region_merge(WALLS, r_id)

for r_id in [17,18,19,20,21]:
    Mesh.region_merge(CYLINDER, r_id)

for r_id in [23,24,25,26,27]:
    Mesh.region_merge(BEAM_INTERFACE, r_id)

Mesh.region_merge(BEAM_LEFT, 22)

for r_id in [5,6,7,8,9]:
    Mesh.region_merge(OUTLET, r_id)

for r_id in [14,15,16]:
    Mesh.region_merge(INLET, r_id)

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

mfw = gf.MeshFem(Mesh, 2)
mfw.set_fem(gf.Fem('FEM_QK(2,2)'))

###########
#  MODEL
###########

md = gf.Model("real")

md.add_fem_variable("u", mfu)
md.add_fem_variable("v", mfv)
md.add_filtered_fem_variable("p", mfp, FLUID)
md.add_filtered_fem_variable("w", mfw, FLUID)

md.add_fem_data("u_n", mfu)
md.add_fem_data("v_n", mfv)

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
#            WEAK FORMULATION (same as before)
#################################################

md.add_macro("F(u)", "Id(2)+Grad(u)")
md.add_macro("J(u)", "Det(F(u))")
md.add_macro('sigma_f_vu(v,u)',
    "rho_f*nu_f*(Grad(v)*Inv(F(u)) + (Inv(F(u)))'*(Grad(v))')")
md.add_macro('sigma_f_p(p)', "-p*Id(2)")
md.add_macro("E_GL(u)", "0.5*((F(u))'*F(u) - Id(2))")
md.add_macro('Sigma_s(u)', "2*mu_s*E_GL(u) + lambda_solid*Trace(E_GL(u))*Id(2)")
md.add_macro("PK1(u)", "F(u)*Sigma_s(u)")
md.add_macro("g_f(v,u)", "-rho_f*nu_f*( Inv(F(u))'*(Grad(v))' )")

md.add_nonlinear_term(mim, "theta0*(rho_f/dt)*J(u)*(v - v_n).Test_v", FLUID)
md.add_nonlinear_term(mim, "theta1*(rho_f/dt)*J(u_n)*(v - v_n).Test_v", FLUID)
md.add_nonlinear_term(mim, "-(rho_f/dt)*J(u)*(Grad(v)*Inv(F(u))*(u - u_n)).Test_v", FLUID)
md.add_nonlinear_term(mim, "(J(u)*sigma_f_p(p)*(Inv(F(u)))'):Grad_Test_v", FLUID)
md.add_nonlinear_term(mim, "J(u)*Trace(Grad(v)*Inv(F(u)))*Test_p", FLUID)
md.add_nonlinear_term(mim, "theta0*rho_f*J(u)*(Grad(v)*(Inv(F(u))*v)).Test_v", FLUID)
md.add_nonlinear_term(mim, "theta0*(J(u)*sigma_f_vu(v, u)*(Inv(F(u)))'):Grad_Test_v", FLUID)
md.add_nonlinear_term(mim, "-theta0*(g_f(v, u)*Normal).Test_v", OUTLET)

md.add_source_term(mim, "-theta1*rho_f*J(u_n)*(Grad(v_n)*(Inv(F(u_n))*v_n)).Test_v", FLUID)
md.add_source_term(mim, "-theta1*(J(u_n)*sigma_f_vu(v_n, u_n)*(Inv(F(u_n)))'):Grad_Test_v", FLUID)
md.add_source_term(mim, "theta1*(g_f(v_n, u_n)*Normal).Test_v", OUTLET)

md.add_nonlinear_term(mim, "alpha_mesh*w.Test_w", FLUID)
md.add_nonlinear_term(mim, "-alpha_mesh*Grad(u):Grad_Test_w", FLUID)
md.add_nonlinear_term(mim, "alpha_mesh*Grad(w):Grad_Test_u", FLUID)

md.add_nonlinear_term(mim, "(rho_s/dt)*(v - v_n).Test_v", BEAM)
md.add_nonlinear_term(mim, "theta0*(PK1(u)):Grad_Test_v", BEAM)
md.add_source_term(mim, "-theta1*(PK1(u_n)):Grad_Test_v", BEAM)

md.add_nonlinear_term(mim, "delta*rho_s*(1/dt)*(u - u_n).Test_u", BEAM)
md.add_nonlinear_term(mim, "-delta*rho_s*theta0*v.Test_u", BEAM)
md.add_source_term(mim, "delta*rho_s*theta1*v_n.Test_u", BEAM)

#########################
#  BOUNDARY CONDITIONS
#########################

V_inlet = md.interpolation("[0,0]", mfv)
md.add_initialized_fem_data('V_inlet', mfv, V_inlet)

md.add_Dirichlet_condition_with_multipliers(mim, "v", mfv, INLET, "V_inlet")
md.add_Dirichlet_condition_with_multipliers(mim, "v", mfv, WALLS)
md.add_Dirichlet_condition_with_multipliers(mim, "v", mfv, CYLINDER)

md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, WALLS)
md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, CYLINDER)
md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, INLET)
md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, OUTLET)

md.add_Dirichlet_condition_with_multipliers(mim, "w", mfw, WALLS)
md.add_Dirichlet_condition_with_multipliers(mim, "w", mfw, CYLINDER)
md.add_Dirichlet_condition_with_multipliers(mim, "w", mfw, INLET)
md.add_Dirichlet_condition_with_multipliers(mim, "w", mfw, OUTLET)

md.add_Dirichlet_condition_with_multipliers(mim, "u", mfu, BEAM_LEFT)
md.add_Dirichlet_condition_with_multipliers(mim, "v", mfv, BEAM_LEFT)

# =====================================================================
#  INITIAL CONDITIONS / RESTART LOADING
# =====================================================================

restart_data_loaded = None
if RESTART_FROM is not None:
    if not os.path.exists(RESTART_FROM):
        raise FileNotFoundError(f"Restart file not found: {RESTART_FROM}")
    restart_data_loaded = np.load(RESTART_FROM, allow_pickle=True).item()
    print(f"Loaded restart file: {RESTART_FROM}")
    print(f"  Keys in file: {list(restart_data_loaded.keys())}")

is_restart = (restart_data_loaded is not None)

log_mode = "a" if is_restart else "w"
log_file = open(f"{output_dir}/results_log.txt", log_mode)

def log(msg=""):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

if is_restart:
    # ==============================================================
    #  FIX: Restore variables INDIVIDUALLY instead of model_state
    #  This avoids issues with Lagrange multiplier ordering
    # ==============================================================

    # Check what keys exist in the restart file
    has_model_state = 'model_state' in restart_data_loaded
    has_individual = 'u_n' in restart_data_loaded and 'v_n' in restart_data_loaded

    log("")
    log("=" * 60)
    log("RESTART LOADING")
    log("=" * 60)
    log(f"  File: {RESTART_FROM}")
    log(f"  Step: {restart_data_loaded['step']}")
    log(f"  Time: {restart_data_loaded['t']:.6f} s")
    log(f"  Has model_state: {has_model_state}")
    log(f"  Has individual u_n/v_n: {has_individual}")

    if has_individual:
        # ---- ROBUST METHOD: restore u_n and v_n individually ----
        saved_u_n = restart_data_loaded['u_n']
        saved_v_n = restart_data_loaded['v_n']

        # Validate sizes match
        expected_u_size = len(md.variable("u"))
        expected_v_size = len(md.variable("v"))

        log(f"  Expected u size: {expected_u_size}, saved: {len(saved_u_n)}")
        log(f"  Expected v size: {expected_v_size}, saved: {len(saved_v_n)}")

        if len(saved_u_n) != expected_u_size:
            raise ValueError(f"u_n size mismatch: saved {len(saved_u_n)} vs expected {expected_u_size}")
        if len(saved_v_n) != expected_v_size:
            raise ValueError(f"v_n size mismatch: saved {len(saved_v_n)} vs expected {expected_v_size}")

        # Set u_n and v_n (the "previous step" data)
        md.set_variable("u_n", saved_u_n)
        md.set_variable("v_n", saved_v_n)

        # Set u and v as initial guess for Newton (same as u_n, v_n)
        # This gives Newton a good starting point
        md.set_variable("u", saved_u_n.copy())
        md.set_variable("v", saved_v_n.copy())

        log(f"  ✓ Restored u_n, v_n individually")
        log(f"  ✓ Set u, v initial guess from u_n, v_n")
        log(f"  max|u_n| = {np.max(np.abs(saved_u_n)):.6e}")
        log(f"  max|v_n| = {np.max(np.abs(saved_v_n)):.6e}")

    elif has_model_state:
        # ---- FALLBACK: use model_state (less robust) ----
        log("  WARNING: Using model_state (may include stale multipliers)")
        try:
            md.to_variables(restart_data_loaded['model_state'])
            log(f"  ✓ Restored model_state")
            log(f"  max|u_n| = {np.max(np.abs(md.variable('u_n'))):.6e}")
            log(f"  max|v_n| = {np.max(np.abs(md.variable('v_n'))):.6e}")
        except Exception as e:
            log(f"  ✗ model_state restore FAILED: {e}")
            log(f"  Falling back to fresh start!")
            is_restart = False
    else:
        log("  ✗ No usable restart data found!")
        is_restart = False

    if is_restart:
        start_step = restart_data_loaded['step'] + 1

        # Restore histories
        time_history   = list(restart_data_loaded.get('time_history', []))
        ux_history     = list(restart_data_loaded.get('ux_history', []))
        uy_history     = list(restart_data_loaded.get('uy_history', []))
        drag_history   = list(restart_data_loaded.get('drag_history', []))
        lift_history   = list(restart_data_loaded.get('lift_history', []))
        p_diff_history = list(restart_data_loaded.get('p_diff_history', []))

        log(f"  Histories restored: {len(time_history)} entries")
        log(f"  Resuming from step {start_step}")
        log("=" * 60)
        log("")

if not is_restart:
    # ---- FRESH START ----
    u_init = md.interpolation("[0,0]", mfu)
    v_init = md.interpolation("[0,0]", mfv)
    md.set_variable("u_n", u_init)
    md.set_variable("v_n", v_init)

    start_step = 0

    time_history   = []
    ux_history     = []
    uy_history     = []
    drag_history   = []
    lift_history   = []
    p_diff_history = []

    log(f"FSI-2 Benchmark — Fresh Start")
    log(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("")

# =====================================================================
#  LOGGING
# =====================================================================

log("=" * 60)
log("Problem Parameters (FSI-2 Benchmark)")
log("=" * 60)
log(f"  Channel:   L = {L}, H = {H}")
log(f"  Fluid:     rho = {rho_fluid}, nu = {ν_fluid}")
log(f"  Solid:     rho = {rho_solid}, E = {E}, nu = {nu_solid}")
log(f"  dt = {dt}, num_steps = {num_steps}, T = {T}")
log(f"  theta = {theta}")
log("")

n_u = len(md.variable("u"))
n_v = len(md.variable("v"))
n_p = len(md.variable("p"))
n_w = len(md.variable("w"))
total_dofs = n_u + n_v + n_p + n_w

log(f"  Total DOFs: {total_dofs}")
log("")

#########################
#  TRACKING SETUP
#########################

A = np.array([0.6, 0.2])

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

remaining_steps = num_steps - start_step

if remaining_steps <= 0:
    log(f"Nothing to do: start_step={start_step} >= num_steps={num_steps}")
    log_file.close()
    exit(0)

log("=" * 60)
if is_restart:
    log(f"Resuming from step {start_step}, {remaining_steps} steps remaining")
else:
    log("Starting FSI-2 Dynamic Analysis")
log("=" * 60)

export_every = 100
log_every = 10

progress = tqdm(desc="FSI-2 time stepping", total=remaining_steps)

for step in range(start_step, num_steps):
    progress.update(1)

    t = (step + 1) * dt

    # ---- Inlet ramp ----
    if t < t_ramp:
        ramp = 0.5 * (1.0 - np.cos(π * t / t_ramp))
    else:
        ramp = 1.0

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

    p_full = md.interpolation("p", mfp)
    w_full = md.interpolation("w", mfw)

    # ---- Update previous time step ----
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

    # ---- Log ----
    if step % log_every == 0 or step == num_steps - 1:
        log(f"")
        log(f"Step {step+1}/{num_steps}, t = {t:.4f} s (ramp = {ramp:.4f})")
        log(f"  Newton iters: {nbit}, converged: {converged}")
        log(f"  u_x(A) = {u_Ax:.8e},  u_y(A) = {u_Ay:.8e}")
        log(f"  F_D = {F_D:.6f},  F_L = {F_L:.6f},  dP = {p_diff:.6f}")
        log(f"  max|u| = {np.max(np.abs(u_sol)):.6e}")
        log(f"  max|v| = {np.max(np.abs(v_sol)):.6e}")
        log(f"  max|w| = {np.max(np.abs(w_sol)):.6e}")

    # ---- Export & Save ----
    if step % export_every == 0 or step == num_steps - 1:

        np.savetxt(f"{output_dir}/displacement_history.txt",
                   np.column_stack([time_history, ux_history, uy_history]),
                   header="Time u_x(A) u_y(A)", fmt='%.10e')

        np.savetxt(f"{output_dir}/force_history.txt",
                   np.column_stack([time_history, drag_history,
                                    lift_history, p_diff_history]),
                   header="Time F_D F_L Pressure_Diff", fmt='%.10e')

        mfv.export_to_vtu(
            f"{output_dir}/fluid_and_solid_{step:06d}.vtu",
            mfu, u_sol, "Displacement",
            mfv, v_sol, "Velocity",
            mfp, p_full, "Pressure",
            mfw, w_full, "w_biharmonic")

        # ============================================================
        #  FIX: Save variables INDIVIDUALLY for robust restart
        #  Also save model_state as backup
        # ============================================================
        restart_dict = {
            # Individual variables (ROBUST — always works)
            'u_n': md.variable("u_n").copy(),
            'v_n': md.variable("v_n").copy(),
            'u':   u_sol.copy(),
            'v':   v_sol.copy(),
            'p':   p_full.copy(),
            'w':   w_full.copy(),

            # Model state (backup — may not work across versions)
            'model_state': md.from_variables(),

            # Metadata
            'step': step,
            't':    t,

            # Histories
            'time_history':   time_history,
            'ux_history':     ux_history,
            'uy_history':     uy_history,
            'drag_history':   drag_history,
            'lift_history':   lift_history,
            'p_diff_history': p_diff_history,
        }

        restart_path = f"{output_dir}/restart_{step:06d}.npy"
        np.save(restart_path, restart_dict, allow_pickle=True)
        log(f"  >> Saved restart: {restart_path}")

progress.close()

# =========================================================================
#                        FINAL OUTPUT
# =========================================================================

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

    uy_centered = uy_osc - uy_mean
    crossings = np.where(np.diff(np.sign(uy_centered)))[0]
    if len(crossings) >= 2:
        periods = np.diff(t_osc[crossings[::2]])
        freq_uy = 1.0 / np.mean(periods) if len(periods) > 0 else float('nan')
    else:
        freq_uy = float('nan')

    log("")
    log("=" * 60)
    log("FSI-2 Benchmark Final Results")
    log("=" * 60)
    log(f"  u_x(A) = {ux_mean:.6e} ± {ux_amp:.6e}")
    log(f"  u_y(A) = {uy_mean:.6e} ± {uy_amp:.6e}")
    log(f"  F_D = {drag_mean:.4f} ± {drag_amp:.4f}")
    log(f"  F_L = {lift_mean:.4f} ± {lift_amp:.4f}")
    log(f"  f(u_y) = {freq_uy:.4f} Hz")

mfv.export_to_vtu(f"{output_dir}/fluid_final.vtu",
    mfu, u_sol, "Displacement",
    mfv, v_sol, "Velocity",
    mfp, p_full, "Pressure")

log("Analysis complete!")
log_file.close()