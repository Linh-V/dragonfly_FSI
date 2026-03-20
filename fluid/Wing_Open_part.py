from datetime import datetime
import getfem as gf 
import numpy as np
import os
import sys

###########################
# Dragongon wing in open domain
###########################
"""
The goal is to solve the full Navier-Stokes equations for a dragonfly profile 
immersed in a fluid in an open domain. 
The aim is to replicate the phenomenon observed for 0° AoA in Nastro & Chiarini::
https://arxiv.org/pdf/2502.11309

Parrallelized over multiple processes with a restart possibility

Launch: mpirun --allow-run-as-root -n 4 python Wing_Open_part.py
"""

if __name__ == "__main__":

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

    #output directory to save results
    output_dir = "result_wing/results_wing_open_Re_100"

    if is_master:
        os.makedirs(output_dir, exist_ok=True)

    if HAS_MPI:
        comm.Barrier()

    def log(msg=""):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    # =====================================================================
    #  RESTART CONFIGURATION
    # =====================================================================
    # RESTART_FROM = None
    RESTART_FROM = "/home/l_vu/research_project/fluid/result_wing/results_wing_open_Re_100/restart.npy"  # Path to the restart file (if any)
    # =====================================================================

    ##################
    ## PROBLEM DATA ##
    ##################

    # Geometry parameters
    L_chord = 0.01 # Chord length of the wing    

    L = 36*L_chord      # Channel length
    H = 21*L_chord      # Channel height
    dx = 10*L_chord     # Wing bottom tip x-coordinate
    dy = 11*L_chord     # Wing bottom tip y-coordinate
    t = 0.005*L_chord   # Wing thickness

    ## Using non dimensional parameters
    Re = 100           # Simulated Reynolds number
    U_inlet = 1 # Inlet velocity (m/s)

    # Fluid properties
    rho = 1.0    # Density (kg/m³)
    mu = (U_inlet*L_chord) / Re   # Kinematic viscosity (Pa·s)

    # Time parameters:
    T = 2    # Total simulation time (s)
    dt = 5e-6  # Time step
    num_steps = int(T / dt)

    ##########
    ## MESH ##
    ##########

    Mesh= gf.Mesh('Import', 'gmsh','Mesh/wing_open_quad.msh')

    # h = min(Mesh.convex_radius())
    # print( f"Minimum mesh size h ={h}, and CFL = "f"{1}, thus dt should be less than {1*h/(1.5*U_inlet)}" )
    
    ############
    # REGIONS ##
    ############

    FLUID = 1
    INLET = 2
    OUTLET = 3
    WALLS = 4
    WING = 5

    ########################
    ## INTEGRATION METHOD ##    
    ########################

    # check with fenics
    mim = gf.MeshIm(Mesh, gf.Integ('IM_QUAD(9)'))

    ##################
    ## FEM ELEMENTS ##
    ##################

    # Velocity: P2 elements (quadratic)
    mf_v = gf.MeshFem(Mesh, 2)
    mf_v.set_fem(gf.Fem('FEM_QK(2,2)'))

    # Pressure: P1 elements (linear)
    mf_p = gf.MeshFem(Mesh, 1)
    mf_p.set_fem(gf.Fem('FEM_QK(2,1)'))


    ####################
    ## SOLVER SETUP   ##
    ####################

    # Storage for results
    time_history = []
    cd_history = []
    cl_history = []
    p_diff_history = []
    div_history = []

    p_front_point = np.array([[10*L_chord], [11*L_chord+t/2]])  # Point in front of the wing
    p_back_point = np.array([[11*L_chord], [11*L_chord+t/2]])   # Point behind the wing

    ###################
    ##     Models    ##
    ###################
    """
    We implement a projection method with three models:
    1. Tentative velocity
    2. Pressure correction
    3. Velocity correction

    for the tentative velocity we use an Adams-Bashforth scheme for the convective term
    and Crank-Nicolson for the diffusive term.

    remark:
    all the terms in the weak form with the corresponding fem variable
    have to be written in the source term (thus as a rhs term)
    Time derivative

    """

    #################################
    ## Model 1: Tentative velocity ##
    #################################
    # Solve: rho/dt*(u* - u^n) + rho*(1.5*u^n - 0.5*u^{n-1})·∇u*
    #        + mu*∇²u* = 0
    
    md1 = gf.Model("real")
    

    md1.add_fem_variable("u", mf_v) # u i s the variable
    md1.add_fem_data("u_n", mf_v) # u is the previus step
    md1.add_fem_data("u_n1", mf_v) # u is the previus to the previus step
    md1.add_fem_data("p_n", mf_p) # p is the half previus step p(n-1/2)
    
    # data
    md1.add_initialized_data("rho", rho)
    md1.add_initialized_data("mu", mu)
    md1.add_initialized_data("dt", dt)
    md1.add_initialized_data("H", H)

    
    md1.add_linear_term(mim, '(rho/dt)*u.Test_u', FLUID)
    md1.add_source_term(mim, '(rho/dt)*u_n.Test_u', FLUID)

    # Convection (Adams-Bashforth): (1.5*u_n - 0.5*u_n1)*0.5*Grad(u+un)
    md1.add_linear_term(mim,
        '0.5*rho*((Grad_u).(1.5*u_n - 0.5*u_n1)).Test_u', FLUID)
    md1.add_source_term(mim,
        '-0.5*rho*((Grad_u_n).(1.5*u_n - 0.5*u_n1)).Test_u', FLUID) ############
    
    # Crank-Nicolson diffusion: 0.5*(mu*∇²(u+u_n))f
    md1.add_linear_term(mim, ' 0.5*mu*(Grad_u):Grad_Test_u', FLUID) 
    md1.add_source_term(mim, '- 0.5*mu*(Grad_u_n):Grad_Test_u', FLUID) 

    # Pressure from previous step
    md1.add_source_term(mim, 'p_n*Div_Test_u', FLUID) 
    

    # Boundary conditions

    """V_inlet is zero at the first iteration 
    then it is a uniform profile with a ramp function during 2s up to the maximum velocity U_max"""

    inlet_dofs = mf_v.basic_dof_on_region(INLET)
    V_inlet= md1.interpolation("[0,0]", mf_v)
    md1.add_initialized_fem_data('V_inlet', mf_v, V_inlet)

    V_noslip = md1.interpolation( "[0,0]" , mf_v)
    md1.add_initialized_fem_data('V_noslip', mf_v, V_noslip)
    
    md1.add_Dirichlet_condition_with_multipliers(mim, "u", mf_v, INLET, "V_inlet")
    # md1.add_Dirichlet_condition_with_multipliers(mim, "u", mf_v, WING, "V_noslip")
    md1.add_Dirichlet_condition_with_simplification("u", WING)


    # Slip-wall with u.n = 0
    gamma = 1e6 # Penalty parameter
    md1.add_nonlinear_term(mim, f"{gamma} * (u.Normal) * (Test_u.Normal)", WALLS)

    ##################################
    ## Model 2: Pressure correction ##
    ##################################
    # Solve: ∇²φ = (rho/dt)*∇·u*
    
    md2 = gf.Model("real")

    md2.add_fem_variable("phi", mf_p)
    md2.add_fem_data("u_star", mf_v)
    
    # problem  data
    md2.add_initialized_data("rho", rho)
    md2.add_initialized_data("dt", dt)
    
    # Poisson equation weak form 
    md2.add_linear_term(mim, 'Grad_phi.Grad_Test_phi', FLUID)
    md2.add_source_term(mim, '-(rho/dt)*Div_u_star*Test_phi', FLUID) # the minus comes from the integration by parts of the ∇²φ

    # BC: φ = 0 at outlet
    # md2.add_Dirichlet_condition_with_multipliers(mim, "phi", 1, OUTLET)
    md2.add_Dirichlet_condition_with_simplification("phi", OUTLET)


    
    ################################
    # Model 3: Velocity correction #
    ################################
    # u^{n+1} = u* - (dt/rho)*∇φ
    
    # Mass matrix M = ∫ ρ * u · v dx weak form
    md3 = gf.Model("real")
    md3.add_fem_variable("u_new", mf_v)
    md3.add_fem_data("phi", mf_p)
    md3.add_fem_data("u_star", mf_v)

    md3.add_initialized_data("rho", rho)
    md3.add_initialized_data("dt", dt)
    md3.add_initialized_data("mu", mu)
    md3.add_initialized_data("H", H)
    
    md3.add_linear_term(mim, 'rho*u_new.Test_u_new', FLUID)
    md3.add_source_term(mim, 'rho*u_star.Test_u_new - dt*Grad_phi.Test_u_new', FLUID) 

    ##################################
    # Model to compute drag and lift #
    ##################################

    md_force = gf.Model("real")
    md_force.add_fem_data("p_new", mf_p)
    md_force.add_fem_data("u_new", mf_v)
    md_force.add_initialized_data("mu", mu)

    #########################
    ## Flow Initialization ##
    #########################

    # =====================================================================
    #  RESTART LOADING (ALL processes must load and set the same data)
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

    # Log file — master only
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
        has_individual = 'u_n' in restart_data_loaded
        has_model_state = 'model_state1' in restart_data_loaded

        log("")
        log("=" * 60)
        log("RESTART LOADING")
        log("=" * 60)
        log(f"  File: {RESTART_FROM}")
        log(f"  Step: {restart_data_loaded['step']}")
        log(f"  Time: {restart_data_loaded['t']:.6f} s")
        log(f"  Has individual u_n: {has_individual}")
        log(f"  Has model_state: {has_model_state}")
        log(f"  MPI processes: {nprocs}")

        if has_individual:
            saved_u_n1 = restart_data_loaded['u_n1']
            saved_u_n = restart_data_loaded['u_n']
            saved_p = restart_data_loaded['p_n']

            expected_u_size = len(md1.variable("u"))

            log(f"  Expected u size: {expected_u_size}, saved: {len(saved_u_n)}")

            if len(saved_u_n) != expected_u_size:
                log("  ERROR: Size mismatch!")
                if HAS_MPI:
                    comm.Barrier()
                sys.exit(1)

            # ALL processes must set the same variables
            md1.set_variable("u_n1", saved_u_n1)
            md1.set_variable("u_n", saved_u_n)
            md1.set_variable("p_n", saved_p)

            log(f"  ✓ Restored u_n, v_n on all {nprocs} process(es)")
            log(f"  max|u_n| = {np.max(np.abs(saved_u_n)):.6e}")

        elif has_model_state:
            log("  WARNING: Using model_state fallback")
            try:
                md1.to_variables(restart_data_loaded['model_state1'])
                md2.to_variables(restart_data_loaded['model_state2'])
                md3.to_variables(restart_data_loaded['model_state3'])
                log(f"  ✓ Restored model_state on all {nprocs} process(es)")
            except Exception as e:
                log(f"  ✗ model_state FAILED: {e}")
                is_restart = False

        else:
            log("  ✗ No usable restart data!")
            is_restart = False

    if is_restart:
        start_step = restart_data_loaded['step'] + 1

        # Histories — only master needs them
        if is_master:
            time_history   = list(restart_data_loaded.get('time_history', []))
            cd_history   = list(restart_data_loaded.get('cd_history', []))
            cl_history   = list(restart_data_loaded.get('cl_history', []))
            p_diff_history = list(restart_data_loaded.get('p_diff_history', []))
            div_history = [0.0] * len(time_history)
            log(f"  Histories restored: {len(time_history)} entries")

        log(f"  Resuming from step {start_step}")
        log("=" * 60)
        log("")

    if not is_restart:

        ## INITIAL CONDITIONS ##

        u_n =  md1.interpolation("[0,0]", mf_v)     # u^n
        u_n1 = md1.interpolation("[0,0]", mf_v)     # u^{n-1}
        p_n =  md1.interpolation("0", mf_p)    # p^n

        md1.set_variable("u_n", u_n)    # set the previus step u, at the beginning is zero
        md1.set_variable("u_n1", u_n1)  # set the previus step of the previus step u, at the beginning is zero
        md1.set_variable("p_n", p_n)    # set the previus step p, at the beginning is zero

        start_step = 0

        if is_master:
            time_history   = []
            ux_history     = []
            uy_history     = []
            drag_history   = []
            lift_history   = []
            p_diff_history = []

        log(f"Fluid Dynamic Simulation — Dragonfly Wing in Open Domain")
        log(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log("")

    # Synchronize after restart loading
    if HAS_MPI:
        comm.Barrier()

    # =====================================================================
    #  LOGGING
    # =====================================================================

    log("=" * 60)
    log("Problem Parameters")
    log("=" * 60)
    log(f"  Channel: L = {L}, H = {H}")
    log(f"  Wing Leading Edge bottom position = ({dx}, {dy}), thickness = {t}")
    log(f"  Dragonfly Wing: L = {L_chord}")
    log(f"  Fluid: rho = {rho}, nu = {mu} (Re = {Re})")
    log(f"  Inlet: U_inlet = {U_inlet}")
    log(f"  Time parameter: dt = {dt}, steps = {num_steps}")
    log("")

    log("=" * 60)
    log("Mesh Information")
    log("=" * 60)
    log(f"  Total points:       {Mesh.nbpts()}")
    log(f"  Total convexes:     {Mesh.nbcvs()}")
    log(f"  Regions in the mesh are: {Mesh.regions()}")
    log("")

    log(f"Velocity DOFs: {mf_v.nbdof()}")
    log(f"Pressure DOFs: {mf_p.nbdof()}")

    n_vf = len(md1.variable("u"))
    n_pf = len(md2.variable("phi"))
    total_dofs = n_vf + n_pf

    log("=" * 60)
    log("Degrees of Freedom")
    log("=" * 60)
    log(f"  Fluid velocity          (v_f): {n_vf}")
    log(f"  Fluid Pressure                (p_f): {n_pf}")
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
        log(f"Resuming FSI-3 from step {start_step}, {remaining_steps} steps remaining")
    else:
        log("=" * 60)
        log("Starting Time Stepping")
        log("=" * 60)

    if is_master:
        progress = tqdm(desc=f"Wing Fluid Simulation [{nprocs} procs]", total=remaining_steps)

    export_every = 100  # Export results every N steps
    log_every = 1  # Log every N steps

    if is_restart:
        start_step = restart_data_loaded['step'] + 1
        t = restart_data_loaded['t']  # Load from restart
    else:
        t = 0.0  # Fresh start

    for step in range(start_step, num_steps):

        if is_master:
            progress.update(1)
        
        #################################
        # Step 1: solve Model 1 (i.e. Solve Tentative velocity )
        #################################
        t = t + dt
        
        T_ramp = 0.003 # Ramp duration (s)
        if t <= T_ramp:  
            ramp_factor = np.sin(np.pi * t / T_ramp)* U_inlet
        else:
            ramp_factor = U_inlet

        V_inlet_expr = f"{ramp_factor}*[1, 0]"
        V_inlet = md1.interpolation(V_inlet_expr, mf_v)
        md1.set_variable('V_inlet', V_inlet) 

        md1.solve("noisy", "max_iter", 100, "max_res", 1e-12, "lsolver", "mumps")
        u_star = md1.variable("u")

        #################################
        # STEP 2: Solve Model 2 (i.e. Solve Pressure correction)
        #################################
        # tange matrix is constant but ths changes
        # Updating fem data 
        md2.set_variable("u_star", u_star)   
        md2.solve("noisy", "max_iter", 100, "max_res", 1e-12, "lsolver", "mumps")
        phi= md2.variable("phi")

        #################################
        # STEP 3: Solve Model 3 (i.e. Solve Velocity correction)
        #################################
        # tange matrix is constant but ths changes
        # updating fem data:
        md3.set_variable("phi", phi)
        md3.set_variable("u_star", u_star)
        md3.solve("noisy", "max_iter", 100, "max_res", 1e-12, "lsolver", "mumps")
        u_new= md3.variable("u_new")

        ##################
        ## Values Update #
        ##################
        
        p_new = md1.variable("p_n") + phi

        md1.set_variable("u_n1", md1.variable("u_n").copy())
        md1.set_variable("u_n", u_new.copy())
        md1.set_variable("p_n", p_new.copy())
                
        
        #############
        ## Checks: ##
        #############
       
        # Boundary:

        # Extract the interpolated inlet velocity values at those DOFs
        V_inlet_at_dofs = V_inlet[inlet_dofs]

        # Extract the computed solution values at those DOFs
        u_new_at_inlet = u_new[inlet_dofs]

        # Compare the values
        diff = np.linalg.norm(u_new_at_inlet - V_inlet_at_dofs)
        relative_diff = diff / np.linalg.norm(V_inlet_at_dofs) if np.linalg.norm(V_inlet_at_dofs) > 0 else diff

        #################
        ## Divergence: ##
        #################
       
        md_force.set_variable("p_new", p_new)
        md_force.set_variable("u_new", u_new)
        
        # L2 norm of the velocity divergence
        div_norm2 = gf.asm_generic(mim, 0,'pow(Div_u_new,2)',FLUID, md_force)
        div_norm = np.sqrt(div_norm2)

        #########################
        # Compute drag and lift #
        #########################
        # Traction: σ·n = [μ(∇u + ∇u^T) - pI]·n
        traction = gf.asm_generic(mim, 0, "(mu*(Grad_u_new + Grad_u_new') - p_new*Id(2))*Normal",WING, md_force)
        
        Fx = -traction[0]
        Fy = -traction[1]
        
        # Drag and lift coefficients
        
        Cd = 2 * Fx / (rho * U_inlet**2 * L_chord)
        Cl = 2 * Fy / (rho * U_inlet**2 * L_chord)
        
        # Pressure difference
        try:
            p_front = gf.compute_interpolate_on(mf_p, p_new, p_front_point)[0]
            p_back = gf.compute_interpolate_on(mf_p, p_new, p_back_point)[0]
            p_diff = p_front - p_back
        except:
            p_diff = 0.0
        
        time_history.append(t)
        cd_history.append(Cd)
        cl_history.append(Cl)
        p_diff_history.append(p_diff)
        div_history.append(div_norm)

        #################################
        # Export results
        #################################

        if is_master:
        
            if step % export_every == 0: # export every 100 steps thus every 0.1s
                mf_v.export_to_vtu(f"{output_dir}/velocity_{step:06d}.vtu",
                                mf_v, u_new, "Velocity",
                                mf_p, p_new, "Pressure")
                
            # ---- Log step results ----
            if step % log_every == 0 or step == num_steps - 1:
                log(f"")
                log(f"Step {step+1}/{num_steps}, t = {t:.4f}")
                log(f"  Divergence: ‖div(u_new)‖ₗ₂ = {div_norm:.6e}")
                log(f"  Cd = {Cd:.6f}, Cl = {Cl:.6f}, ΔP = {p_diff:.6f}")
                log(f"  max|u_f| = {np.max(np.abs(u_new)):.6e}")
                log(f"  max|p_f| = {np.max(np.abs(p_new)):.6e}")

                np.savetxt(f"{output_dir}/force_coefficients_channel_restart.txt",
                        np.column_stack([time_history, cd_history, cl_history, p_diff_history, div_history]),
                        header="Time Cd Cl Pressure_Diff, div norm",
                        fmt='%.8e')

                restart_dict = {
                    'u_n1': md1.variable("u_n").copy(),
                    'u_n':   u_new.copy(),
                    'p_n':   p_new.copy(),
                    'model_state1': md1.from_variables(),
                    'model_state2': md2.from_variables(),
                    'model_state3': md3.from_variables(),
                    'step': step,
                    't':    t,
                    'time_history':   time_history,
                    'cd_history':   cd_history,
                    'cl_history':   cl_history,
                    'p_diff_history': p_diff_history,
                }

                restart_path = f"{output_dir}/restart_2.npy"
                np.save(restart_path, restart_dict, allow_pickle=True)
                log(f"  >> Saved restart: {restart_path} for step {step:06d}")



        if div_norm > 1e10:
            if is_master:
                log(f"‖div(u_new)‖ₗ₂ = {div_norm:.6e}")
                log("Warning: High velocity divergence detected, which may indicate numerical issues.")
                log(f"Inlet BC verification:")
                log(f"  Absolute difference: {diff:.6e}")
                log(f"  Relative difference: {relative_diff:.6e}")
                log(f"  Max absolute difference: {np.max(np.abs(u_new_at_inlet - V_inlet_at_dofs)):.6e}")

                mf_v.export_to_vtu(f"{output_dir}/velocity_last.vtu",
                    mf_v, u_new, "Velocity",
                    mf_p, p_new, "Pressure")
            break

        # ---- Synchronize ----
        if HAS_MPI:
            comm.Barrier()
    if is_master:
        progress.close()



