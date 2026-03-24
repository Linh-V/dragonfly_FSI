"""
This script implements the flow around a cylinder in a channel benchmark using GetFEM and the N-S equations
"""
import getfem as gf 
import numpy as np
import os


###########################
# DFG 2D-3 Benchmark: Cylinder in Channel
###########################
"""
The goal is to solve Navier-Stokes time variant problem
"""

# Generate the mesh
if __name__ == "__main__":
    #output directory to save results
    output_dir = "fluid/results_cylinder_channel_getfem_mono_quad_fenics"
    os.makedirs(output_dir, exist_ok=True)

    ##########
    ## MESH ##
    ##########
    """ Same mesh as in the fenics benchmark """

    Mesh= gf.Mesh('Import', 'gmsh','fluid/Mesh/cylinder_channel_from_python.msh')

    ############
    # REGIONS ##
    ############
    """
    regions flagging in gmsh does not work thus it's necessary to give a region for each line or physical surface. 
    """

    FLUID = 1
    WALLS = 10
    INLET = 7
    OUTLET = 8
    OBSTACLE = 5

    Mesh.region_merge(WALLS, 6)
    Mesh.region_merge(WALLS, 9)
    print("Regions in the mesh are:", Mesh.regions())

    ##################
    ## PROBLEM DATA ##
    ##################
    """ Problem parameters as in the benchmark description """

    # Geometry parameters
    L = 2.2      # Channel length
    H = 0.41     # Channel height
    c_x = 0.2    # Cylinder center x
    c_y = 0.2    # Cylinder center y
    r = 0.05     # Cylinder radius

    # Fluid properties
    mu = 0.001   # Dynamic viscosity (Pa·s)
    rho = 1.0    # Density (kg/m³)

    # Inlet velocity parameters
    U_max = 1.5  # Maximum inlet velocity (m/s)

    # Time parameters:

    T = 8.0      # Total simulation time (s) - reduced for testing
    dt = 1/1600  # Time step
    num_steps = int(T / dt)
    print(f"Total time steps: {num_steps}")

    ########################
    ## INTEGRATION METHOD ##
    ########################

    mim = gf.MeshIm(Mesh, gf.Integ('IM_QUAD(5)'))

    ##################
    ## FEM ELEMENTS ##
    ##################

    # Velocity: P2 elements (quadratic)
    mf_v = gf.MeshFem(Mesh, 2)
    mf_v.set_fem(gf.Fem('FEM_QK(2,2)'))
    
    # Pressure: P1 elements (linear)
    mf_p = gf.MeshFem(Mesh, 1)
    mf_p.set_fem(gf.Fem('FEM_QK(2,1)'))
   
    print(f"Velocity DOFs: {mf_v.nbdof()}")
    print(f"Pressure DOFs: {mf_p.nbdof()}")

    
    ####################
    ## SOLVER SETUP   ##
    ####################

    # Storage for results
    time_history = []
    cd_history = []
    cl_history = []
    p_diff_history = []
    div_history = []

    # Points for pressure evaluation
    p_front_point = np.array([[0.15], [0.2]])
    p_back_point = np.array([[0.25], [0.2]])


    ###################
    ##     Models    ##
    ###################
    """
    We implemet the time variant stokes problem.
    The time discrtization is achieved by the Crank Nicolson method
    """

    #################################
    ## Model: Tentative velocity ##
    #################################
    # Solve: rho/dt*(u* - u^n) + rho*(1.5*u^n - 0.5*u^{n-1})·∇u*
    #        + mu*∇²u* = 0
    
    md = gf.Model("real")
    

    md.add_fem_variable("u", mf_v) # u is the velocity variable
    md.add_fem_variable("p", mf_p) # p is the pressure variable
    md.add_fem_data("u_n", mf_v) # un is the previus step of u
    md.add_fem_data("u_n1", mf_v) # un1 is the previus step of un


    # data
    md.add_initialized_data("rho", rho)
    md.add_initialized_data("mu", mu)
    md.add_initialized_data("dt", dt)
    md.add_initialized_data("H", H)

    # MOMENTUM CONSERVATION: 
    md.add_linear_term(mim, '(rho/dt)*u.Test_u', FLUID)
    md.add_source_term(mim, '(rho/dt)*u_n.Test_u', FLUID)

    # Crank-Nicolson diffusion: 0.5*(mu*∇²(u+u_n))f
    md.add_linear_term(mim, ' 0.5*mu*(Grad_u):Grad_Test_u', FLUID) 
    md.add_source_term(mim, '-0.5*mu*(Grad_u_n):Grad_Test_u', FLUID)
     
    #Convection (Adams-Bashforth): (1.5*u_n - 0.5*u_n1)*0.5*Grad(u+un)
    md.add_linear_term(mim,
        '0.5*rho*(Grad_u.(1.5*u_n - 0.5*u_n1)).Test_u', FLUID)
    md.add_source_term(mim,
        '-0.5*rho*(Grad_u_n.(1.5*u_n - 0.5*u_n1)).Test_u', FLUID)
    
    # AB2: 1.5*ρ(u^n·∇)u^n - 0.5*ρ(u^{n-1}·∇)u^{n-1}
    # Pressure from previous step
    md.add_linear_term(mim, '- p:Div_Test_u', FLUID) 
    
    # MASS CONSERVATION:
    # Divergence free constraint 
    md.add_linear_term(mim, 'Div_u.Test_p', FLUID)


    #######################
    # Boundary conditions #
    #######################

    """V_inlet is zero at the first iteration 
    then it is a parabolic profile with a ramp function up to the maximum velocity U_max, 
    thus V_inlet = ramp(t)*[4*U_max*X(2)*(H-X(2))/(H*H), 0]
    where ramp(t) = sin(pi*t/8)"""


    inlet_dofs = mf_v.basic_dof_on_region(INLET)
    V_inlet= md.interpolation("[0,0]", mf_v)
    md.add_initialized_fem_data('V_inlet', mf_v, V_inlet)

    V_noslip = md.interpolation( "[0,0]" , mf_v)
    md.add_initialized_fem_data('V_noslip', mf_v, V_noslip)
    
    md.add_Dirichlet_condition_with_multipliers(mim, "u", mf_v, INLET, "V_inlet")
    md.add_Dirichlet_condition_with_multipliers(mim, "u", mf_v, WALLS, "V_noslip")
    md.add_Dirichlet_condition_with_multipliers(mim, "u", mf_v, OBSTACLE, "V_noslip")
    

    ################################
    ## Model to compute cl and cd ##
    ################################
    md_force = gf.Model("real")
    md_force.add_fem_data("p_new", mf_p)
    md_force.add_fem_data("u_new", mf_v)
    md_force.add_initialized_data("mu", mu)
    #########################
    ## Flow Initialization ##
    #########################
   
    ## INITIAL CONDITIONS ##

    u_n = u_n1 =  md.interpolation("[0,0]", mf_v)     # u^n
    md.set_variable("u_n", u_n)    # set the previus step u, at the beginning is zero
    md.set_variable("u_n1", u_n1)    # set the previus step u, at the beginning is zero


    for step in range(num_steps):
        
        #################################
        # Step 1: solve Model 1 (i.e. Solve Tentative velocity )
        #################################
        t = (step + 1) * dt 

        ramp_factor = np.sin(np.pi * t / 8) 
        V_inlet_expr = f"{ramp_factor}*[4*1.5*X(2)*(H-X(2))/(H*H), 0]"
        V_inlet = md.interpolation(V_inlet_expr, mf_v)
        md.set_variable('V_inlet', V_inlet) 
        
        md.solve("noisy", "max_iter", 100, "max_res", 1e-8, "lsolver", "superlu")
        u_new = md.variable("u")
        p_new = md.variable("p")
        
        ##################
        ## Values Update #
        ##################

        # Salva u_n PRIMA di aggiornarlo
        u_n_old = md.variable("u_n").copy()
        
        # Aggiorna i livelli temporali
        md.set_variable("u_n1", u_n_old)
        md.set_variable("u_n", u_new.copy())

        md_force.set_variable("u_new", u_new.copy())
        md_force.set_variable("p_new", p_new.copy())
        
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

        print(f"Inlet BC verification:")
        print(f"  Absolute difference: {diff:.6e}")
        print(f"  Relative difference: {relative_diff:.6e}")
        print(f"  Max absolute difference: {np.max(np.abs(u_new_at_inlet - V_inlet_at_dofs)):.6e}")


        #################
        ## Divergence: ##
        #################


        # L2 norm of the velocity divergence
        div_norm2 = gf.asm_generic(mim, 0,'pow((Trace(Grad_u_new)),2)',FLUID, md_force)
        div_norm = np.sqrt(div_norm2)
        print(f"‖div(u_new)‖ₗ₂ = {div_norm:.6e}")

        #########################
        # Compute drag and lift #
        #########################
        
        
        # Traction: σ·n = [μ(∇u + ∇u^T) - pI]·n
        traction = gf.asm_generic(mim, 0, "(mu*(Grad_u_new + Grad_u_new') - p_new*Id(2))*Normal",OBSTACLE, md_force)
        
        Fx = -traction[0]
        Fy = -traction[1]
        
        # Drag and lift coefficients
        D = 2 * r  # Diameter
        U_mean = 2.0 / 3.0 * U_max  # Average velocity for parabolic profile
        
        Cd = 2 * Fx / (rho * U_mean**2 * D)
        Cl = 2 * Fy / (rho * U_mean**2 * D)
        
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
        
        print(f"  Cd={Cd:.6f}, Cl={Cl:.6f}, ΔP={p_diff:.6f}")
        

        #################################
        # Export results
        #################################
        
        if step % 80 == 0: # export every 160 steps thus every  0.05s hence there are going to be 500 files
            mf_v.export_to_vtu(f"{output_dir}/velocity_{step:06d}.vtu",
                            mf_v, u_new, "Velocity",
                            mf_p, p_new, "Pressure")
            
        #################################
        # Save force coefficients
        #################################

        np.savetxt(f"{output_dir}/force_coefficients_channel_triangle.txt",
                np.column_stack([time_history, cd_history, cl_history, p_diff_history, div_history]),
                header="Time Cd Cl Pressure_Diff, div norm",
                fmt='%.8e')
