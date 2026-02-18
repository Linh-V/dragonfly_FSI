import getfem as gf 
import numpy as np
import os
import gmsh

###########################
# Benchmark: Transient Cylinder & Flexible beam assembly in a channel
###########################
"""
The goal is to solve a fully transient Navier-Stokes for a cylinder and flexible beam in a channel.
This is treating a  fully transient FSI with harmonic mesh motion.
The time discretization approximated thanks to the shifted Crank-Nicholson scheme.
following the benchmark description available at and the formulation in Problem 7.21:
http://www.thomaswick.org/links/lecture_notes_FSI_Nov_28_2019.pdf
"""

def create_CylBeam_mesh_FSI(
    mesh_file="MESH/cylinder_beam_channel_FSI.msh",
    level=0,
    visualize=False
):
    """
    Pure TRIANGULAR mesh for cylinder + rigid beam in channel.
    Five refinement levels (0–4) suitable for convergence studies.
    """

    assert 0 <= level <= 4, "level must be between 0 and 4"

    gmsh.initialize()
    gmsh.model.add("Cylinder_Beam_FSI_TRI")

    # -----------------------   
    # Geometry parameters
    # -----------------------
    L = 2.5
    H = 0.41

    c_x = 0.2
    c_y = 0.2
    r = 0.05

    L_beam = 0.35
    W_beam = 0.02

    gdim = 2

    # -----------------------
    # Refinement levels
    # -----------------------
    levels = [
        dict(lc_cyl=r/2.5,  lc_beam=W_beam/1.5, lc_far=0.15),
        dict(lc_cyl=r/3.5,  lc_beam=W_beam/2.2, lc_far=0.10),
        dict(lc_cyl=r/5.0,  lc_beam=W_beam/3.2, lc_far=0.065),
        dict(lc_cyl=r/7.0,  lc_beam=W_beam/4.5, lc_far=0.045),
        dict(lc_cyl=r/10.0, lc_beam=W_beam/6.0, lc_far=0.030),
    ]

    lc_cyl  = levels[level]["lc_cyl"]
    lc_beam = levels[level]["lc_beam"]
    lc_far  = levels[level]["lc_far"]

    # -----------------------
    # Geometry
    # -----------------------

    # Main rectangular channel
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H)

    # Rigid and fixed cylinder
    cylinder  = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

    # Flexible beam
    beam      = gmsh.model.occ.addRectangle(
        c_x + r, c_y - W_beam / 2, 0, L_beam, W_beam
    )

    # Defining the fluid domain
    fluid, _ = gmsh.model.occ.cut(
        [(2, rectangle)],
        [(2, cylinder), (2, beam)],
        removeObject=True,
        removeTool=False
    )

    gmsh.model.occ.synchronize()

    # -----------------------
    # Physical groups
    # -----------------------
    FLUID, INLET, OUTLET, WALLS  = 1, 2, 3, 4

    gmsh.model.addPhysicalGroup(2, [fluid[0][1]], FLUID)

    # Channel boundaries
    inflow, outflow, walls = [], [], []
    for dim, tag in gmsh.model.getBoundary(fluid, oriented=False):
        x, y, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        if np.isclose(x, 0.0, atol=1e-6):
            inflow.append(tag)
        elif np.isclose(x, L, atol=1e-6):
            outflow.append(tag)
        elif np.isclose(y, 0.0, atol=1e-6) or np.isclose(y, H, atol=1e-6):
            walls.append(tag)

    gmsh.model.addPhysicalGroup(1, inflow, INLET)
    gmsh.model.addPhysicalGroup(1, outflow, OUTLET)
    gmsh.model.addPhysicalGroup(1, walls, WALLS)

    # Solid (beam)
    BEAM = 5
    gmsh.model.addPhysicalGroup(2, [beam], BEAM)

    # Cylinder boundary (rigid)
    CYLINDER = 7
    cyl_bnd = gmsh.model.getBoundary([(2, cylinder)], oriented=False)
    gmsh.model.addPhysicalGroup(1, [b[1] for b in cyl_bnd], CYLINDER)

    # Interface + Attached beam (left boundary)
    INTERFACE = 6
    BEAM_LEFT = 8

    beam_bnd = gmsh.model.getBoundary([(2, beam)], oriented=True)
    left_side_lines = []
    for dim, tag in beam_bnd:
        coord = gmsh.model.getValue(dim, tag, [0.5])
        mid_x = coord[0]
        
        if abs(mid_x - (c_x + r)) < 1e-6:
                left_side_lines.append(tag)

    all_bnd_tags = [abs(b[1]) for b in beam_bnd]
    interface_tags = list(set(all_bnd_tags) - set(left_side_lines))

    gmsh.model.addPhysicalGroup(1, left_side_lines, BEAM_LEFT)
    gmsh.model.addPhysicalGroup(1, interface_tags, INTERFACE)

    # -----------------------
    # Mesh size fields (refining at the boundary of the beam and cylinder)
    # -----------------------
    f_cyl = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(
        f_cyl, "EdgesList", [b[1] for b in cyl_bnd]
    )
    th_cyl = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(th_cyl, "IField", f_cyl)
    gmsh.model.mesh.field.setNumber(th_cyl, "LcMin", lc_cyl)
    gmsh.model.mesh.field.setNumber(th_cyl, "LcMax", lc_far)
    gmsh.model.mesh.field.setNumber(th_cyl, "DistMin", r)
    gmsh.model.mesh.field.setNumber(th_cyl, "DistMax", 4*r)

    f_beam = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(
        f_beam, "EdgesList", [b[1] for b in beam_bnd]
    )
    th_beam = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(th_beam, "IField", f_beam)
    gmsh.model.mesh.field.setNumber(th_beam, "LcMin", lc_beam)
    gmsh.model.mesh.field.setNumber(th_beam, "LcMax", lc_far)
    gmsh.model.mesh.field.setNumber(th_beam, "DistMin", W_beam)
    gmsh.model.mesh.field.setNumber(th_beam, "DistMax", 4*W_beam)

    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(
        f_min, "FieldsList", [th_cyl, th_beam]
    )
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    # -----------------------
    # Mesh generation
    # -----------------------
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(mesh_file)

    if visualize:
        gmsh.fltk.run()

    gmsh.finalize()

if __name__ == "__main__":

    #output directory to save results
    output_dir = "results/results_Cylbeam_transient_benchmark"
    os.makedirs(output_dir, exist_ok=True)

    ##################
    ## MESH ##
    ##################

    # Generate the mesh
    create_CylBeam_mesh_FSI("MESH/cylinder_beam_channel_tri.msh", level=4, visualize=False)

    # Import Mesh 
    Mesh = gf.Mesh('Import', 'gmsh','MESH/cylinder_beam_channel_tri.msh')

    ############
    # REGIONS ##
    ############

    # Physical groups
    FLUID = 1
    INLET= 2
    OUTLET = 3
    WALLS = 4
    BEAM = 5
    INTERFACE = 6
    CYLINDER = 7
    BEAM_LEFT = 8

    ##################
    ## PROBLEM DATA ##
    ##################
    """ Problem parameters as in the benchmark description """

    # Geometry parameters
    L = 2.5         # Channel length
    H = 0.41        # Channel height
    c_x = 0.2       # Cylinder center x
    c_y = 0.2       # Cylinder center y
    r = 0.05        # Cylinder radius
    L_beam = 0.35   # Beamn length
    W_beam = 0.02   # Beam width

    # Fluid properties
    nu_f = 0.001        # Dynamic viscosity (Pa·s)
    rho_f = 1000.0      # Density (kg/m³)

    # Structural properties
    rho_s = 10000.0                             # Density (kg/m³)
    nu_s = 0.4                                  # Poisson ration
    mu_s = 500000.0                             # Shear modulus (2nd Lamé coefficient) (kg/m·s²)
    E = 2*mu_s*(1+nu_s)                              # Young's modulus (Pa)
    lambda_s = E*nu_s/((1+nu_s)*(1-2*nu_s))     # 1st Lamé coefficient

    # Structural damping 
    gamma_w = 0
    gamma_s = 0

    # Inlet velocity parameters
    U_mean = 1.0  # Mean inlet velocity (m/s)

    # Time discretization parameters
    T = 4                # Total simulation time (4s for testing & 15 for full simulation)
    k = 0.01                # Time step 
    num_steps = int(T/k)    # Number of steps

    ########################
    ## INTEGRATION METHOD ##
    ########################

    mim = gf.MeshIm(Mesh, gf.Integ("IM_TRIANGLE(9)"))

    #########################
    ## FEM ELEMENTS ##
    #########################

    # Fluid displacement: P2 elements (quadratic)
    mfu_fluid = gf.MeshFem(Mesh, 2)
    mfu_fluid.set_fem(gf.Fem('FEM_PK(2,2)'))

    # Fluid Velocity: P2 elements (quadratic)
    mfv_fluid = gf.MeshFem(Mesh, 2)
    mfv_fluid.set_fem(gf.Fem('FEM_PK(2,2)'))

    # Fluid Pressure: P1 elements (linear)
    mfp_fluid = gf.MeshFem(Mesh, 1)
    mfp_fluid.set_fem(gf.Fem('FEM_PK(2,1)'))

    # Solid displacement: P2 elements
    mfu_solid = gf.MeshFem(Mesh, 2)
    mfu_solid.set_fem(gf.Fem('FEM_PK(2,2)'))

    # Solid velocity: P2 elements
    mfv_solid = gf.MeshFem(Mesh, 2)
    mfv_solid.set_fem(gf.Fem('FEM_PK(2,2)'))

    # Solid pressure: P1 elements
    mfp_solid = gf.MeshFem(Mesh, 1)
    mfp_solid.set_fem(gf.Fem('FEM_PK(2,1)'))

    #################################
    ##     MODEL (Problem 7.21 )   ##
    #################################

    md = gf.Model("real")

    #FEM variables
    md.add_filtered_fem_variable("u_fluid", mfu_fluid, FLUID)   # Displacement of the fluid
    md.add_filtered_fem_variable("v_fluid", mfv_fluid, FLUID)   # Velocity of the fluid
    md.add_filtered_fem_variable("p_fluid", mfp_fluid, FLUID)   # Pressure of the fluid
    md.add_filtered_fem_variable("u_solid", mfu_solid, BEAM)   # Displacement of the solid (beam)
    md.add_filtered_fem_variable("v_solid", mfv_solid,BEAM)   # Velocity of the solid (beam)

    # FEM data
    md.add_fem_data("uf_n", mfu_fluid) # uf_n is the fluid displacement at the previous step
    md.add_fem_data("vf_n", mfv_fluid) # vf_n is the fluid velocity at the previous step
    md.add_fem_data("us_n", mfu_fluid) # us_n is the solid displacement at the previous step
    md.add_fem_data("vs_n", mfv_fluid) # vs_n is the solid velocity at the previous step

    # Problem data
    md.add_initialized_data("rho_f", rho_f)
    md.add_initialized_data("nu_f", nu_f)

    md.add_initialized_data("rho_s", rho_s)
    md.add_initialized_data("mu_s", mu_s)
    md.add_initialized_data("lambda_s", lambda_s)

    md.add_initialized_data("gamma_w", gamma_w)
    md.add_initialized_data("gamma_s",gamma_s)

    md.add_initialized_data("k", k) 

    md.add_initialized_data("H", H)

    # MACROS: 
    
    md.add_macro("F(u)", "Id(2)+Grad(u)")       # Deformation gradient
    md.add_macro("J(u)", "Det(F(u))")           # Deformation determinant
    
    # Transformed Fluid Cauchy stress tensor
    md.add_macro("sigma_f_p(p)", "-p*Id(2)")                                                  # Hydrostatic pressure
    md.add_macro("sigma_f_vu(u,v)", "2*rho_f*nu_f*(Grad(v)*Inv(F(u)) + (((Inv(F(u)))')*((Grad(v))')))")     # Dynamic viscosity

    # Transformed Cauchy Solid stress tensor
    md.add_macro("E(u)", "0.5*((F(u))'*F(u) - Id(2))")     # Green-Lagrange strain tensor
    md.add_macro("Sigma_s(u)", "2*mu_s*E(u) + lambda_s*Trace(E(u))*Id(2)")     # STVK linear elastic
 
    # Corrective term
    md.add_macro("g_f(u,v)", "rho_f*nu_f*((Inv(F(u)))'*(Grad(v))')") # correction for the do-nothing outflow condition

    # Mesh motion
    md.add_macro("sigma_mesh(u)", "Grad(u)/J(u)")   # Harmonic mesh

    # Continuity
    md.add_macro("Continuity(u, v)", "J(u) * Trace(Grad(v) * Inv(F(u)))")

    # Mesh velocity 
    md.add_macro("w(u)", "(u - uf_n) / k") # Mesh velocity (du_f/dt)

    # Solid velocity gradient
    md.add_macro("eps(v)", "0.5*(Grad(v) + Grad(v)')")

    # Fluid momentum (No body force: f = 0)
    md.add_nonlinear_term(mim, "(J(u_fluid)*rho_f *(v_fluid - vf_n) / k).Test_v_fluid", FLUID )         # Time-derivative
    md.add_nonlinear_term(mim, "rho_f*J(u_fluid)*((Inv(F(u_fluid))*(v_fluid - w(u_fluid))).Grad_v_fluid).(Test_v_fluid)", FLUID)                              # Convective term
    md.add_nonlinear_term(mim, "(J(u_fluid)*(sigma_f_p(p_fluid) + sigma_f_vu(u_fluid, v_fluid))*(Inv(F(u_fluid)))'):Grad_Test_v_fluid", FLUID)  # Internal stress term
    md.add_nonlinear_term(mim, "-(g_f(u_fluid, v_fluid)*Normal).Test_v_fluid", OUTLET)

    # Solid Momentum, 1st equation (No body force: f = 0, no damping introduced to the system)
    md.add_nonlinear_term(mim, "(rho_s*(v_solid - vs_n)/k).Test_v_solid", BEAM)            # Time-derivative
    md.add_nonlinear_term(mim, "(F(u_solid)*Sigma_s(u_solid)):Grad_Test_u_solid", BEAM)    # Internal stress
    md.add_nonlinear_term(mim, "gamma_w*(v_solid.Test_v_solid)", BEAM)      # Weak damping
    md.add_nonlinear_term(mim, "gamma_s*(eps(v_solid):Grad_Test_v_solid)", BEAM)        # Strong damping

    # Fluid mesh motion
    md.add_nonlinear_term(mim, "sigma_mesh(u_fluid):Grad_Test_u_fluid", FLUID)

    # Solid Momentum, 2nd equation
    md.add_nonlinear_term(mim, "rho_s*(((u_solid - us_n)/k - v_solid).Test_u_solid)", BEAM)     # Kinematic consistency

    # Fluid mass conservation
    md.add_nonlinear_term(mim, "Continuity(u_fluid, v_fluid)*Test_p_fluid", FLUID)

    # Neumann FSI coupling conditions - Stress balance at the interface
    md.add_nonlinear_term(mim, "((J(u_fluid)*(sigma_f_p(p_fluid) + sigma_f_vu(u_fluid, v_fluid))*((Inv(F(u_fluid)))'))*Normal).Test_v_solid", INTERFACE)   # Fluid traction term
    # md.add_nonlinear_term(mim, "-((F(u_solid)*Sigma_s(u_solid))*Normal).Test_v_fluid", INTERFACE)       # Solid traction term
    md.add_nonlinear_term(mim, "gamma_s*((eps(v_solid)*Normal):Test_v_solid)", INTERFACE)

    #######################
    # Boundary conditions #
    #######################

    # Equal displacement at the interface (u_f = u_s)
    md.add_Dirichlet_condition_with_multipliers(mim, "u_fluid", mfu_fluid, INTERFACE, "u_solid")

    # Equal displacement velocity at the interface (u_f = u_s)
    md.add_Dirichlet_condition_with_multipliers(mim, "v_fluid", mfv_fluid, INTERFACE, "v_solid")

    # No-slip conditions for fluid velocity (walls, interface)
    V_noslip = md.interpolation( "[0,0]" , mfv_fluid)
    md.add_initialized_fem_data('V_noslip', mfv_fluid, V_noslip)
    md.add_Dirichlet_condition_with_multipliers(mim, "v_fluid", mfv_fluid, WALLS, "V_noslip")
    md.add_Dirichlet_condition_with_multipliers(mim, "v_fluid", mfv_fluid, CYLINDER, "V_noslip")

    # Imposing the inlet fluid velocity
    V_inlet = md.interpolation("[0,0]", mfv_fluid) # Initial inlet velocity is 0 and will follow a ramp-up
    md.add_initialized_fem_data('V_inlet', mfv_fluid, V_inlet)
    md.add_Dirichlet_condition_with_multipliers(mim, "v_fluid", mfv_fluid, INLET, "V_inlet")

    # Fixing the beam's left boundary (rigid to the cylinder)
    U_fixed_solid = md.interpolation( "[0,0]" , mfu_solid)
    md.add_initialized_fem_data('U_fixed_solid', mfu_solid, U_fixed_solid)
    md.add_Dirichlet_condition_with_multipliers(mim, "u_solid", mfu_solid, BEAM_LEFT, "U_fixed_solid")

    # Mesh no-motion at the walls, inlet, outlet
    U_fixed_mesh = md.interpolation( "[0,0]" , mfu_fluid)
    md.add_initialized_fem_data('U_fixed_mesh', mfu_fluid, U_fixed_mesh)
    md.add_Dirichlet_condition_with_multipliers(mim, "u_fluid", mfu_fluid, INLET, "U_fixed_mesh" )
    md.add_Dirichlet_condition_with_multipliers(mim, "u_fluid", mfu_fluid, OUTLET, "U_fixed_mesh")
    md.add_Dirichlet_condition_with_multipliers(mim, "u_fluid", mfu_fluid, WALLS, "U_fixed_mesh")
    md.add_Dirichlet_condition_with_multipliers(mim, "u_fluid", mfu_fluid, CYLINDER, "U_fixed_mesh")

    #########################
    ## Flow Initialization ##
    #########################
   
    ## INITIAL CONDITIONS ##

    uf_n =  md.interpolation("[0,0]", mfu_fluid)     # uf_n
    vf_n =  md.interpolation("[0,0]", mfv_fluid)     # vf_n
    us_n =  md.interpolation("[0,0]", mfu_solid)     # us_n
    vs_n =  md.interpolation("[0,0]", mfv_solid)     # vsn

    md.set_variable("uf_n", uf_n)    # set the previus step u_fluid, at the beginning is zero
    md.set_variable("vf_n", vf_n)    # set the previus step v_fluid, at the beginning is zero
    md.set_variable("us_n", us_n)    # set the previus step u_solid, at the beginning is zero
    md.set_variable("vs_n", vs_n)    # set the previus step v_solid, at the beginning is zero

    for step in range(num_steps):

        t = (step + 1) * k 
        
        #################################
        # Updating the inlet velocity
        #################################

        if t < 2.0:
            ramp_factor = ((1 - np.cos(np.pi*t/2))/2)*U_mean
        else:
            ramp_factor = U_mean
        
        V_inlet_expr = f"{ramp_factor}*[1.5*4*X(2)*(H-X(2))/(H*H), 0]"
        V_inlet = md.interpolation(V_inlet_expr, mfv_fluid)
        md.set_variable('V_inlet', V_inlet) 

        ###################
        # Solving the model
        ###################

        md.solve("noisy", "max_iter", 100, "max_res", 1e-8, "lsolver", "mumps", "lsearch", "basic")

        ##################
        ## Values Update #
        ##################

        u_f = md.variable('u_fluid')
        v_f = md.variable('v_fluid')
        p_f = md.variable('p_fluid')
        u_s = md.variable('u_solid')
        v_s = md.variable('v_solid')

        # Create full-sized arrays for export
        u_f_full = np.zeros(mfu_fluid.nbdof())
        v_f_full = np.zeros(mfv_fluid.nbdof())
        p_full = np.zeros(mfp_fluid.nbdof())
        u_s_full = np.zeros(mfu_solid.nbdof())
        v_s_full = np.zeros(mfv_solid.nbdof())
        
        # Get DOF indices for each region
        fluid_u_dofs = mfu_fluid.basic_dof_on_region(FLUID)
        fluid_v_dofs = mfv_fluid.basic_dof_on_region(FLUID)
        fluid_p_dofs = mfp_fluid.basic_dof_on_region(FLUID)
        solid_u_dofs = mfu_solid.basic_dof_on_region(BEAM)
        solid_v_dofs = mfv_solid.basic_dof_on_region(BEAM)

        # Fill in the values at the appropriate DOF locations
        u_f_full[fluid_u_dofs] = u_f
        v_f_full[fluid_v_dofs] = v_f
        p_full[fluid_p_dofs] = p_f
        u_s_full[solid_u_dofs] = u_s
        v_s_full[solid_v_dofs] = v_s

        md.set_variable("uf_n", u_f_full)
        md.set_variable("vf_n", v_f_full) 
        md.set_variable("us_n", u_s_full)   
        md.set_variable("vs_n", v_s_full)

        #################################
        # Export results
        #################################

        if step % 10 == 0: # export every 10 steps thus every  0.1s to get 150 files

            print(f"{step}/{num_steps}")
            
            mfv_fluid.export_to_vtu(f"{output_dir}/results_{step:06d}.vtu",
                                    mfu_fluid, u_f_full, "Fluid Displacement",  
                                    mfv_fluid, v_f_full, "Fluid Velocity",
                                    mfp_fluid, p_full, "Fluid Pressure", 
                                    mfu_solid, u_s_full, "Beam Displacement")

            
            u_s_max = max(u_s)
            print(f"Max beam deflection: {u_s_max}")