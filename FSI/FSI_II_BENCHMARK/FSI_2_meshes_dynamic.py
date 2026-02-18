import getfem as gf
from Functions import verify_regions
import numpy as np
import os
from tqdm import tqdm

gf.util_trace_level(1)
gf.util_warning_level(1)
π = np.pi

##########################################
# Dynamic Fluid-Structure Interaction
# Two-mesh monolithic ALE formulation
# Theta-scheme (Crank-Nicolson) temporal
##########################################
"""The goal os to solve the first benchmarck proposed by turek """
output_dir = "FSI/FSI_Benchmark_Results_Dynamic"
os.makedirs(output_dir, exist_ok=True)

##################
## PROBLEM DATA ##
##################

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
H = 0.41
r = 0.05

# Time parameters
T = 15.0
dt = 0.001
theta = 0.5
num_steps = 30

print(f"FSI Dynamic (2-mesh) Simulation")
print(f"Total time steps: {num_steps}, dt = {dt}, T = {T}")
print(f"Theta = {theta}")

#############
## MESH ##
#############

Mesh_fluid = gf.Mesh('Import', 'gmsh', 'FSI/MESH_GMSH/TF_MESH_FLUID.msh')
Mesh_solid = gf.Mesh('Import', 'gmsh', 'FSI/MESH_GMSH/TF_MESH_BEAM.msh')

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

########################
## INTEGRATION METHOD ##
########################

mim_fluid = gf.MeshIm(Mesh_fluid, gf.Integ('IM_QUAD(5)'))
mim_solid = gf.MeshIm(Mesh_solid, gf.Integ('IM_QUAD(5)'))

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

mfv_solid = gf.MeshFem(Mesh_solid, 2)
mfv_solid.set_fem(gf.Fem('FEM_QK(2,2)'))

###########
## MODEL ##
###########

md = gf.Model("real")

###################
## FEM VARIABLES ##
###################

# Current time step unknowns
md.add_fem_variable("u_f", mfu_fluid)
md.add_fem_variable("v_f", mfv_fluid)
md.add_fem_variable("p_f", mfp_fluid)
md.add_fem_variable("u_s", mfu_solid)
md.add_fem_variable("v_s", mfv_solid)

# Previous time step data
md.add_fem_data("u_f_n", mfu_fluid)
md.add_fem_data("v_f_n", mfv_fluid)
md.add_fem_data("u_s_n", mfu_solid)
md.add_fem_data("v_s_n", mfv_solid)

# Lagrange Multipliers
md.add_filtered_fem_variable("mult_u", mfu_fluid, BEAM_INTERFACE)
md.add_filtered_fem_variable("mult_v", mfv_fluid, BEAM_INTERFACE)
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
md.add_initialized_data("dt", dt)
md.add_initialized_data("theta0", theta)
md.add_initialized_data("theta1", 1.0 - theta)
md.add_initialized_data("delta", 1.0e7)

#################################################
## WEAK FORMULATION ##
#################################################

########### MACROS (same names as static) ###########

# COMMON
md.add_macro("F(u)", "Id(2)+Grad(u)")
md.add_macro("J(u)", "Det(F(u))")

# FLUID STRESS TENSORS
md.add_macro('sigma_f_vu(v,u)', "2*rho_f*nu_f*(Grad(v)*Inv(F(u)) + (Inv(F(u)))'*(Grad(v))')")
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
"""
One-Step-θ scheme (Def. 8.12, Wick):

  A_T(U^{n,k}) + θ A_R(U^n) + A_P(U^n) + A_I(U^n) = -(1-θ) A_R(U^{n-1})

With J^{n,θ} = θ J^n + (1-θ) J^{n-1}

A_T (eq 110):
  (1/k) ρ_f J^{n,θ} (v_f^n - v_f^{n-1}) · ψ^v
  -(1/k) ρ_f J^{n-1} (grad(v_f) F^{-1}_{n-1}) (u_f^n - u_f^{n-1}) · ψ^v

A_R on fluid (eq 105):
  (ρ_f J (F^{-1} v_f · ∇) v_f, ψ^v)                     [convection]
  + (J σ_{f,vu} F^{-T}, ∇ψ^v)                            [viscous stress]
  - (ρ_f ν_f J F^{-T} ∇v^T ·n, ψ^v)_out                 [do-nothing]

A_R on mesh (eq 107):
  (J σ_{x,p} F^{-T}, ∇ψ^u)                               [mesh stress]

A_P (eq 106):
  (J σ_{f,p} F^{-T}, ∇ψ^v)                               [pressure: IMPLICIT, no θ]

A_I (eq 104):
  (div(J F^{-1} v_f), ψ^p)                                [continuity: IMPLICIT, no θ]
"""

# ==========================================
#  A_T: TIME TERMS (eq 110)
# ==========================================

# Fluid temporal: (1/k) * J^{n,θ} * ρ_f * (v_f^n - v_f^{n-1}) · ψ^v
# Split as: θ * J^n part + (1-θ) * J^{n-1} part
md.add_nonlinear_term(mim_fluid,
    "theta0*(rho_f/dt)*J(u_f)*(v_f - v_f_n).Test_v_f",
    FLUID)
md.add_nonlinear_term(mim_fluid,
    "theta1*(rho_f/dt)*J(u_f_n)*(v_f - v_f_n).Test_v_f",
    FLUID)

# ALE correction: -(1/k) * ρ_f * J^{n-1} * (grad(v_f) · F^{-1}_{n-1}) * (u_f^n - u_f^{n-1}) · ψ^v
# This convective term convects the mesh velocity, in case of fully eulerian it would be zero
# while in fully lagrangian,the mesh velocity would be equal to fluid velocity, hence it would
# be eqault to the convective term, but with opposite sign and they would cancel each other. 
# ALE correction: uses J^n and F^{-1}_n (CURRENT config)
md.add_nonlinear_term(mim_fluid,
    "-(rho_f/dt)*J(u_f)*(Grad(v_f).Inv(F(u_f))*(u_f - u_f_n)).Test_v_f",
    FLUID)

# ==========================================
#  A_P: PRESSURE (eq 106) — FULLY IMPLICIT
# ==========================================
# No θ weighting on pressure (it's in A_P, not A_R)

md.add_nonlinear_term(mim_fluid,
    "(J(u_f)*sigma_f_p(p_f)*(Inv(F(u_f)))'):Grad_Test_v_f",
    FLUID)

# ==========================================
#  A_I: INCOMPRESSIBILITY (eq 104) — FULLY IMPLICIT
# ==========================================
# No θ weighting

md.add_nonlinear_term(mim_fluid,
    "J(u_f)*Trace(Grad(v_f)*Inv(F(u_f)))*Test_p_f",
    FLUID)

# ==========================================
#  θ * A_E(U^n): REMAINING TERMS AT TIME n (eq 105, 112)
# ==========================================

# Convection at n: θ * ρ_f * J(u_f) * (grad(v_f) · F^{-1}(u_f) · v_f) · ψ^v
md.add_nonlinear_term(mim_fluid,
    "theta0*rho_f*J(u_f)*(Grad(v_f).(Inv(F(u_f))*v_f)).Test_v_f",
    FLUID)

# Viscous stress at n: θ * (J(u_f) * σ_{f,vu}(v_f, u_f) * F^{-T}(u_f)) : ∇ψ^v
md.add_nonlinear_term(mim_fluid,
    "theta0*(J(u_f)*sigma_f_vu(v_f, u_f)*(Inv(F(u_f)))'):Grad_Test_v_f",
    FLUID)

# Do-nothing at n: -θ * g_f · ψ^v on outlet
md.add_nonlinear_term(mim_fluid,
    "-theta0*(g_f(v_f, u_f)*Normal).Test_v_f",
    OUTLET)

# ==========================================
#  -(1-θ) * A_E(U^{n-1}): REMAINING TERMS AT TIME n-1 (eq 105, 113)
#  they are source terms
# ==========================================

# Convection at n-1 (source, goes to RHS with sign from -(1-θ))
md.add_source_term(mim_fluid,
    "-theta1*rho_f*J(u_f_n)*(Grad(v_f_n).(Inv(F(u_f_n))*v_f_n)).Test_v_f",
    FLUID)

# Viscous stress at n-1 (source, goes to RHS)
md.add_source_term(mim_fluid,
    "-theta1*(J(u_f_n)*sigma_f_vu(v_f_n, u_f_n)*(Inv(F(u_f_n)))'):Grad_Test_v_f",
    FLUID)

# Do-nothing at n-1 (source, goes to RHS)
md.add_source_term(mim_fluid,
    "theta1*(g_f(v_f_n, u_f_n)*Normal).Test_v_f",
    OUTLET)

# ==========================================
#  MESH MOTION (part of A_R, eq 104)
# ==========================================
# θ * mesh at n
md.add_nonlinear_term(mim_fluid,
    "theta0*Mesh_def(u_f):Grad_Test_u_f",
    FLUID)

# -(1-θ) * mesh at n-1 (RHS)
md.add_source_term(mim_fluid,
    "-theta1*Mesh_def(u_f_n):Grad_Test_u_f",
    FLUID)

#####################
## SOLID EQUATIONS ##
#####################
"""
A_T solid (eq 110-111):
  (1/k) ρ_s (v_s^n - v_s^{n-1}) · ψ^v_s
  (u_s^n - u_s^{n-1}) · ψ^u_s             [kinematic: du/dt = v]

θ * A_R solid:
  θ * (F(u_s) Σ_s(u_s)) : ∇ψ^v_s          [solid stress at n]
  -θ * v_s · ψ^u_s                          [kinematic coupling]

-(1-θ) * A_R solid (RHS):
  -(1-θ) * (F(u_s_n) Σ_s(u_s_n)) : ∇ψ^v_s [solid stress at n-1]
  +(1-θ) * v_s_n · ψ^u_s                    [kinematic coupling]
"""
# ==========================================
#  A_T: TIME TERMS (eq 111)
# ==========================================

# ---- A_T: Solid inertia ----
md.add_nonlinear_term(mim_solid,
    "(rho_s/dt)*(v_s - v_s_n).Test_v_s",
    BEAM)

# ---- A_T: Kinematic relation du/dt = v ----
# (u_s - u_s_n)/dt · ψ^u = [θ*v_s + (1-θ)*v_s_n] · ψ^u
# Rearranged: (u_s - u_s_n) · ψ^u - dt*[θ*v_s + (1-θ)*v_s_n] · ψ^u = 0

md.add_nonlinear_term(mim_solid,
    "(rho_s/dt)*(u_s - u_s_n).Test_u_s",
    BEAM)

# ==========================================
#  θ * A_E(U^n): EMAINING TERMS AT TIME n TERMS (eq 106 - 112)
# ==========================================

# kinematic relation at n
md.add_nonlinear_term(mim_solid,
                      "-theta0*rho_s*v_s.Test_u_s", BEAM)
# Solid stress at n 
md.add_nonlinear_term(mim_solid,
    "theta0*(PK1(u_s)):Grad_Test_v_s",
    BEAM)


# ==========================================
#  -(1-θ) * A_E(U^{n-1}): REMAINING TERMS AT TIME n-1 (eq 106, 113)
#  they are source terms
# ==========================================
# kinematic relation at n-1
md.add_source_term(mim_solid,
    "(theta1*rho_s*v_s_n).Test_u_s",
    BEAM)
#Solid stress at n-1 (RHS)
md.add_source_term(mim_solid,
    "-theta1*(PK1(u_s_n)):Grad_Test_v_s",
    BEAM)

#########################
## COUPLING CONDITIONS (eq 77)##
#########################

# Kinematic coupling: u_f = u_s on interface (Lagrange multiplier)
md.add_nonlinear_term(mim_fluid,
    "(u_f - Interpolate(u_s, solid_to_fluid)).Test_mult_u",
    BEAM_INTERFACE)
md.add_nonlinear_term(mim_fluid,
    "mult_u.Test_u_f",
    BEAM_INTERFACE)
# # Kinematic coupling: v_f = v_s on interface (Lagrange multiplier)
md.add_nonlinear_term(mim_fluid,
    "(v_f - Interpolate(v_s, solid_to_fluid)).Test_mult_v",
    BEAM_INTERFACE)
md.add_nonlinear_term(mim_fluid,
    "mult_v.Test_v_f",
    BEAM_INTERFACE)

# Dynamic coupling: fluid traction on solid (Interpolate inside macro args)
md.add_nonlinear_term(mim_solid,
    "-(J(Interpolate(u_f, fluid_to_solid))"
    "*(sigma_f_p(Interpolate(p_f, fluid_to_solid))"
    " + sigma_f_vu(Interpolate(v_f, fluid_to_solid), Interpolate(u_f, fluid_to_solid)))"
    "*Inv(F(Interpolate(u_f, fluid_to_solid)))'*Normal).Test_v_s",
    BEAM_INTERFACE)

# Dynamic coupling: solid traction reaction on fluid
# It seems like by imposing vf = vs this term is not necessary, becasue imposing the velocity, 
# impose already the stress on the fluid.
md.add_nonlinear_term(mim_fluid,
    "(PK1(Interpolate(u_s, solid_to_fluid))*Normal).Test_v_f",
    BEAM_INTERFACE)

#########################
## BOUNDARY CONDITIONS (penalization)##
#########################

# # Inlet (updated each time step)
# V_inlet = md.interpolation("[0,0]", mfv_fluid)
# md.add_initialized_fem_data('V_inlet', mfv_fluid, V_inlet)
# md.add_nonlinear_term(mim_fluid, "1e9*(v_f - V_inlet).Test_v_f", INLET)

# # No-slip walls
# md.add_nonlinear_term(mim_fluid, "1e9*v_f.Test_v_f", WALLS)

# # No-slip cylinder
# md.add_nonlinear_term(mim_fluid, "1e9*v_f.Test_v_f", CYLINDER)


# # Mesh fixity
# md.add_nonlinear_term(mim_fluid, "1e9*u_f.Test_u_f", WALLS)
# md.add_nonlinear_term(mim_fluid, "1e9*u_f.Test_u_f", CYLINDER)
# md.add_nonlinear_term(mim_fluid, "1e9*u_f.Test_u_f", INLET)
# md.add_nonlinear_term(mim_fluid, "1e9*u_f.Test_u_f", OUTLET)

# # Solid fixed left boundary
# md.add_nonlinear_term(mim_solid, "1e10*u_s.Test_u_s", BEAM_LEFT)
# md.add_nonlinear_term(mim_solid, "1e10*v_s.Test_v_s", BEAM_LEFT)

# ######################################
# ## BOUNDARY CONDITIONS (multiplies) ##
# ######################################


# ---- FLUID VELOCITY BCs ----

# INLET: Parabolic velocity profile (time-dependent, needs updatable data)
V_inlet = md.interpolation("[0,0]", mfv_fluid)
md.add_initialized_fem_data('V_inlet', mfv_fluid, V_inlet)
md.add_Dirichlet_condition_with_multipliers(mim_fluid, "v_f", mfv_fluid, INLET, "V_inlet")

# NO-SLIP on WALLS (homogeneous => no data needed)
md.add_Dirichlet_condition_with_multipliers(mim_fluid, "v_f", mfv_fluid, WALLS)

# NO-SLIP on CYLINDER (homogeneous)
md.add_Dirichlet_condition_with_multipliers(mim_fluid, "v_f", mfv_fluid, CYLINDER)

# ---- FLUID MESH DISPLACEMENT BCs (u_f = 0 on all external boundaries) ----

md.add_Dirichlet_condition_with_multipliers(mim_fluid, "u_f", mfu_fluid, WALLS)
md.add_Dirichlet_condition_with_multipliers(mim_fluid, "u_f", mfu_fluid, CYLINDER)
md.add_Dirichlet_condition_with_multipliers(mim_fluid, "u_f", mfu_fluid, INLET)
md.add_Dirichlet_condition_with_multipliers(mim_fluid, "u_f", mfu_fluid, OUTLET)

# ---- SOLID BCs (fixed left boundary) ----

md.add_Dirichlet_condition_with_multipliers(mim_solid, "u_s", mfu_solid, BEAM_LEFT)
md.add_Dirichlet_condition_with_multipliers(mim_solid, "v_s", mfv_solid, BEAM_LEFT)
#########################
## INITIAL CONDITIONS  ##
#########################

u_f_init = md.interpolation("[0,0]", mfu_fluid)
v_f_init = md.interpolation("[0,0]", mfv_fluid)
u_s_init = md.interpolation("[0,0]", mfu_solid)
v_s_init = md.interpolation("[0,0]", mfv_solid)

md.set_variable("u_f_n", u_f_init)
md.set_variable("v_f_n", v_f_init)
md.set_variable("u_s_n", u_s_init)
md.set_variable("v_s_n", v_s_init)

#########################
## TRACKING & HISTORY  ##
#########################

A = np.array([0.6, 0.2])

time_history = []
ux_history = []
uy_history = []
drag_history = []
lift_history = []
p_diff_history = []

# Force computation model (Wick eq. 254-255)
md_force = gf.Model("real")
md_force.add_fem_data("u_force", mfu_fluid)
md_force.add_fem_data("v_force", mfv_fluid)
md_force.add_fem_data("p_force", mfp_fluid)
md_force.add_initialized_data("nu_f", ν_fluid)
md_force.add_initialized_data("rho_f", rho_fluid)

# Same macros needed for force computation
md_force.add_macro("F(u)", "Id(2)+Grad(u)")
md_force.add_macro("J(u)", "Det(F(u))")
md_force.add_macro("sigma_f_vu(v,u)",
    "2*rho_f*nu_f*(Grad(v)*Inv(F(u)) + (Inv(F(u)))'*(Grad(v))')")
md_force.add_macro("sigma_f_p(p)", "-p*Id(2)")

# Pressure evaluation points
p_front_point = np.array([[0.15], [0.2]])
p_back_point = np.array([[0.25], [0.2]])

####################
## DOF SUMMARY    ##
####################

print("=" * 60)
print("Starting FSI Dynamic Analysis")
print("=" * 60)
print(f"Fluid mesh displacement DOFs: {mfu_fluid.nbdof()}")
print(f"Fluid velocity DOFs: {mfv_fluid.nbdof()}")
print(f"Pressure DOFs: {mfp_fluid.nbdof()}")
print(f"Solid displacement DOFs: {mfu_solid.nbdof()}")
print(f"Solid velocity DOFs: {mfv_solid.nbdof()}")
total_dofs = (mfu_fluid.nbdof() + mfv_fluid.nbdof() + mfp_fluid.nbdof()
              + mfu_solid.nbdof() + mfv_solid.nbdof())
print(f"Total DOFs: {total_dofs}")
print("=" * 60)

####################
## TIME STEPPING  ##
####################

export_every = 50
progress = tqdm(desc="Solving PDE", total=num_steps)

for step in range(num_steps):
    progress.update(1)

    t = (step + 1) * dt

    # ---- Update inlet BC with ramp ----
    if step < 15.0:
        ramp = 0.5 * (1.0 - np.cos(π * t / 2.0))
    else:
        ramp = 1.0

    V_inlet_expr = f"{ramp}*[4*1.5*U_mean*X(2)*(H-X(2))/(H*H), 0]"
    V_inlet = md.interpolation(V_inlet_expr, mfv_fluid)
    md.set_variable('V_inlet', V_inlet)

    # ---- Solve ----
        # ---- Solve with fallback ----
    nbit, converged = md.solve("noisy",
                                "max_iter", 500,
                                "max_res", 1e-8,
                                "lsolver", "mumps",
                                "lsearch", "simplest")

    if not converged:
        print(f"  MUMPS/simplest didn't converge ({nbit} iters), trying systematic...")
        nbit, converged = md.solve("noisy",
                                    "max_iter", 500,
                                    "max_res", 1e-7,
                                    "lsolver", "mumps",
                                    "lsearch", "systematic")

    if not converged:
        print(f"  systematic didn't converge ({nbit} iters), trying superlu...")
        nbit, converged = md.solve("noisy",
                                    "max_iter", 300,
                                    "max_res", 1e-6,
                                    "lsolver", "mumps")

    if not converged:
        print(f"  ALL solvers failed at step {step+1}, t={t:.6f}")
        break
            
    # ---- Extract solution ----
    u_f = md.variable("u_f")
    v_f = md.variable("v_f")
    p_f = md.variable("p_f")
    u_s = md.variable("u_s")
    v_s = md.variable("v_s")

    # ---- Update previous time step ----
    md.set_variable("u_f_n", u_f.copy())
    md.set_variable("v_f_n", v_f.copy())
    md.set_variable("u_s_n", u_s.copy())
    md.set_variable("v_s_n", v_s.copy())

    # ---- Displacement at point A ----
    A_col = A.reshape(2, 1)
    result = gf.compute_interpolate_on(mfu_solid, u_s, A_col)
    u_Ax = float(result[0])
    u_Ay = float(result[1])

    # ---- Drag and lift ----
        # ---- Drag and lift (Wick eq. 254-255) ----
    # (F_D, F_L) = integral of J * sigma_f * F^{-T} * n over circle + beam
    md_force.set_variable("u_force", u_f.copy())
    md_force.set_variable("v_force", v_f.copy())
    md_force.set_variable("p_force", p_f.copy())

    # Cylinder traction (u_f = 0 there, so F=I, J=1, but full formula for consistency)
    traction_cyl = gf.asm_generic(mim_fluid, 0,
        "(J(u_force)"
        "*(sigma_f_p(p_force) + sigma_f_vu(v_force, u_force))"
        "*Inv(F(u_force))'*Normal)",
        CYLINDER, md_force)

    # Beam interface traction (u_f ≠ 0, full ALE formula required)
    traction_beam = gf.asm_generic(mim_fluid, 0,
        "(J(u_force)"
        "*(sigma_f_p(p_force) + sigma_f_vu(v_force, u_force))"
        "*Inv(F(u_force))'*Normal)",
        BEAM_INTERFACE, md_force)

    Fx = -(traction_cyl[0] + traction_beam[0])
    Fy = -(traction_cyl[1] + traction_beam[1])
    
    # Pressure difference
    p_front = gf.compute_interpolate_on(mfp_fluid, p_f, p_front_point)[0]
    p_back = gf.compute_interpolate_on(mfp_fluid, p_f, p_back_point)[0]
    p_diff = p_front - p_back

    # ---- Store history ----
    time_history.append(t)
    ux_history.append(u_Ax)
    uy_history.append(u_Ay)
    drag_history.append(Fx)
    lift_history.append(Fy)
    p_diff_history.append(p_diff)

    # ---- Print progress ----
    print(f"\nStep {step+1}/{num_steps}, t = {t:.6f} s")
    print(f"  u_x(A) = {u_Ax:.8e},  u_y(A) = {u_Ay:.8e}")
    print(f"  Cd = {Fx:.6f},  Cl = {Fy:.6f},  ΔP = {p_diff:.6f}")
    print(f"  max|u_s| = {np.max(np.abs(u_s)):.6e}")
    print(f"  max|v_f| = {np.max(np.abs(v_f)):.6e}")

    # ---- Save histories every iteration ----
    np.savetxt(f"{output_dir}/displacement_history.txt",
               np.column_stack([time_history, ux_history, uy_history]),
               header="Time u_x(A) u_y(A)",
               fmt='%.10e')

    np.savetxt(f"{output_dir}/force_history.txt",
               np.column_stack([time_history, drag_history, lift_history, p_diff_history]),
               header="Time Fd Fn Pressure_Diff",
               fmt='%.10e')

    # ---- Export VTU ----
    if step % export_every == 0:
        mfu_fluid.export_to_vtu(
            f"{output_dir}/fluid_{step:06d}.vtu",
            mfu_fluid, u_f, "MeshDisplacement",
            mfv_fluid, v_f, "Velocity",
            mfp_fluid, p_f, "Pressure")

        mfu_solid.export_to_vtu(
            f"{output_dir}/solid_{step:06d}.vtu",
            mfu_solid, u_s, "Displacement",
            mfv_solid, v_s, "Velocity")

# =========================================================================
#  FINAL OUTPUT
# =========================================================================

print("\n" + "=" * 60)
print("FSI Dynamic Simulation Complete")
print("=" * 60)
