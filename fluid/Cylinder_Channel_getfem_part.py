import getfem as gf 
import numpy as np
import os
from Functions import verify_regions


###########################
# DFG 2D-3 Benchmark: Cylinder in Channel
###########################
"""
The goal is to solve the full Navier-Stokes equations for a cylinder immersed in a fluid,
following the benchmark description available at:
https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
"""

# Generate the mesh
if __name__ == "__main__":
    #output directory to save results
    output_dir = "fluid/results_cylinder_channel"
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
    #verify_regions(Mesh, base_filename=f"{output_dir}/mesh_regions")
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

    mim = gf.MeshIm(Mesh, gf.Integ('IM_QUAD(17)'))

    ##################
    ## FEM ELEMENTS ##
    ##################

    # Velocity: P2 elements (quadratic)
    mf_v = gf.MeshFem(Mesh, 2)
    mf_v.set_fem(gf.Fem('FEM_QK(2,3)'))
    
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
    """
    the points are located at the front and back of the cylinder at the mid-height of the channel, they are aligned
    with the cylinder center which is at y=0.2, and they are located on the cylinder surface.
    """
    p_front_point = np.array([[0.15], [0.2]])
    p_back_point = np.array([[0.25], [0.2]])


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
    then it is a parabolic profile with a ramp function up to the maximum velocity U_max, 
    thus V_inlet = ramp(t)*[4*U_max*X(2)*(H-X(2))/(H*H), 0]
    where ramp(t) = sin(pi*t/8)"""


    inlet_dofs = mf_v.basic_dof_on_region(INLET)
    V_inlet= md1.interpolation("[0,0]", mf_v)
    md1.add_initialized_fem_data('V_inlet', mf_v, V_inlet)

    V_noslip = md1.interpolation( "[0,0]" , mf_v)
    md1.add_initialized_fem_data('V_noslip', mf_v, V_noslip)
    
    md1.add_Dirichlet_condition_with_multipliers(mim, "u", mf_v, INLET, "V_inlet")
    md1.add_Dirichlet_condition_with_multipliers(mim, "u", mf_v, WALLS, "V_noslip")
    md1.add_Dirichlet_condition_with_multipliers(mim, "u", mf_v, OBSTACLE, "V_noslip")

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
    md2.add_Dirichlet_condition_with_multipliers(mim, "phi", 1, OUTLET)

    
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
    # Updating fem data: velocities, pressure and the inlet boundary condition
    
    #########################
    ## Flow Initialization ##
    #########################
   
    ## INITIAL CONDITIONS ##

    u_n =  md1.interpolation("[0,0]", mf_v)     # u^n
    u_n1 = md1.interpolation("[0,0]", mf_v)     # u^{n-1}
    p_n =  md1.interpolation("0", mf_p)    # p^n

    md1.set_variable("u_n", u_n)    # set the previus step u, at the beginning is zero
    md1.set_variable("u_n1", u_n1)  # set the previus step of the previus step u, at the beginning is zero
    md1.set_variable("p_n", p_n)    # set the previus step p, at the beginning is zero
    

    for step in range(num_steps):
        
        #################################
        # Step 1: solve Model 1 (i.e. Solve Tentative velocity )
        #################################
        t = (step + 1) * dt 

        ramp_factor = np.sin(np.pi * t / 8) 
        V_inlet_expr = f"{ramp_factor}*[4*1.5*X(2)*(H-X(2))/(H*H), 0]"
        V_inlet = md1.interpolation(V_inlet_expr, mf_v)
        md1.set_variable('V_inlet', V_inlet) 

        
        md1.solve("noisy", "max_iter", 100, "max_res", 1e-12, "lsolver", "superlu")
        u_star = md1.variable("u")

        #################################
        # STEP 2: Solve Model 2 (i.e. Solve Pressure correction)
        #################################
        # tange matrix is constant but ths changes
        # Updating fem data 
        md2.set_variable("u_star", u_star)   
        md2.solve("noisy", "max_iter", 100, "max_res", 1e-12, "lsolver", "superlu")
        phi= md2.variable("phi")

        #################################
        # STEP 3: Solve Model 3 (i.e. Solve Velocity correction)
        #################################
        # tange matrix is constant but ths changes
        # updating fem data:
        md3.set_variable("phi", phi)
        md3.set_variable("u_star", u_star)
        md3.solve("noisy", "max_iter", 100, "max_res", 1e-12, "lsolver", "superlu")
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

        print(f"Inlet BC verification:")
        print(f"  Absolute difference: {diff:.6e}")
        print(f"  Relative difference: {relative_diff:.6e}")
        print(f"  Max absolute difference: {np.max(np.abs(u_new_at_inlet - V_inlet_at_dofs)):.6e}")

        #################
        ## Divergence: ##
        #################
       
        
        md_force.set_variable("p_new", p_new)
        md_force.set_variable("u_new", u_new)
        
        
        # L2 norm of the velocity divergence
        div_norm2 = gf.asm_generic(mim, 0,'pow(Div_u_new,2)',FLUID, md_force)
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

        np.savetxt(f"{output_dir}/force_coefficients_channel.txt",
                np.column_stack([time_history, cd_history, cl_history, p_diff_history, div_history]),
                header="Time Cd Cl Pressure_Diff, div norm",
                fmt='%.8e')

    print("Simulation completed.")
