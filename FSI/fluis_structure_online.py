#!/usr/bin/env python
# -*- coding: UTF8 -*-
#
############################################################################
import numpy as np
import getfem as gf

gf.util_trace_level(1)
gf.util_warning_level(1)
π = np.pi


E = 1000.
ν = 0.4

L = 10.
H_fluid = 5.
H_beam = 0.5

beam_load = 10.

μ_fluid = 0.3
v_in = 1.
p_out =0.

NX_fluid = 20
NX_beam = 25
NY_fluid = 10
NY_beam = 3

# regions
XP_RG = 101
XM_RG = 102
YP_RG = 103
YM_RG = 104
#DIR_RG = 105

m1 = gf.Mesh("import", "structured",
             f"GT='GT_QK(2,2)';ORG=[0,0];SIZES=[{L},{H_fluid}];NSUBDIV=[{NX_fluid},{NY_fluid}]")
m2 = gf.Mesh("import", "structured",
             f"GT='GT_QK(2,2)';ORG=[0,{H_fluid}];SIZES=[{L},{H_beam}];NSUBDIV=[{NX_beam},{NY_beam}]")
for m in (m1,m2):
  m.set_region(XP_RG, m.outer_faces_with_direction([1,0], π/6))
  m.set_region(XM_RG, m.outer_faces_with_direction([-1,0], π/6))
  m.set_region(YP_RG, m.outer_faces_with_direction([0,1], π/6))
  m.set_region(YM_RG, m.outer_faces_with_direction([0,-1], π/6))
#m2.region_merge(DIR_RG, XP_RG)
#m2.region_merge(DIR_RG, XM_RG)

mim_f = gf.MeshIm(m1, 5)
mim_s = gf.MeshIm(m2, 5)

mfu_f_ = gf.MeshFem(m1, 2)
mfu_f_.set_fem(gf.Fem("FEM_Q2_INCOMPLETE(2)"))

mfv1_ = gf.MeshFem(m1, 2)
mfv1_.set_classical_fem(2)

mfp1 = gf.MeshFem(m1, 1)
mfp1.set_classical_fem(1)

mfu_s_ = gf.MeshFem(m2, 2)
mfu_s_.set_fem(gf.Fem("FEM_Q2_INCOMPLETE(2)"))

# remove fixed dofs
kept_dofs = list(set(range(mfu_f_.nbdof()))
                 -set(mfu_f_.basic_dof_on_region(XP_RG))
                 -set(mfu_f_.basic_dof_on_region(XM_RG))
                 -set(mfu_f_.basic_dof_on_region(YM_RG)))
mfu_f = gf.MeshFem("partial", mfu_f_, kept_dofs)
kept_dofs = list(set(range(mfv1_.nbdof()))
                 -set(mfv1_.basic_dof_on_region(YP_RG))
                 -set(mfv1_.basic_dof_on_region(YM_RG)))
mfv1 = gf.MeshFem("partial", mfv1_, kept_dofs)
kept_dofs = list(set(range(mfu_s_.nbdof()))
                 -set(mfu_s_.basic_dof_on_region(XP_RG))
                 -set(mfu_s_.basic_dof_on_region(XM_RG)))
mfu_s = gf.MeshFem("partial", mfu_s_, kept_dofs)


md = gf.Model("real")

md.add_fem_variable("v", mfv1)
md.add_fem_variable("p", mfp1)
md.add_fem_variable("u_f", mfu_f)
md.add_fem_variable("u_s", mfu_s)
md.add_filtered_fem_variable("mult", mfu_f, YP_RG)

md.add_interpolate_transformation_from_expression("beam", m1, m2, "X")
md.add_interpolate_transformation_from_expression("fluid", m2, m1, "X")



md.add_initialized_data("K", E/(3*(1-2*ν)))
md.add_initialized_data("G", E/(2*(1+ν)))
md.add_macro("Dev33(A)", "A-Id(2)/3*(Trace(A)+1)")
md.add_macro("F(u)", "Id(2)+Grad(u)")
md.add_macro("J(u)", "Det(F(u))")
md.add_macro("tauD(u)", "G*pow(J(u),-2/3)*Dev33(F(u)*F(u)')")
md.add_macro("tauH(u)", "K*log(J(u))")
md.add_nonlinear_term(mim_f, "((tauH(u_f)*Id(2)+tauD(u_f))*Inv(F(u_f)')):Grad(Test_u_f)") # Hyperlastic Mesh Deofrmation
md.add_nonlinear_term(mim_s, "((tauH(u_s)*Id(2)+tauD(u_s))*Inv(F(u_s)')):Grad(Test_u_s)") # Neo-Hookean-like hyperelastic material

md.add_nonlinear_term(mim_f, "mult.Test_u_f", YP_RG)   # "load" on the top of the fluid domain
md.add_nonlinear_term(mim_s, f"{beam_load}*Test_u_s(2)", YP_RG) # uniform load on the top of the beam

md.add_nonlinear_term(mim_f, "(u_f-Interpolate(u_s,beam)).Test_mult", YP_RG)   # bond fluid top surface to beam bottom surface

md.add_initialized_data("mu", μ_fluid)
md.add_initialized_data("v_in", v_in)
md.add_initialized_data("p_out", p_out)
if False: # Flow in undeformed domain
  md.add_nonlinear_term(mim_f, "mu*Grad(v):Grad(Test_v)-p*Div(Test_v)" # stokes problem
                              "+Div(v)*Test_p")
  md.add_nonlinear_term(mim_f, f"100*(v+v_in*Normal*(1-sqr(X(2)/{H_fluid/2}-1))).Test_v", XM_RG)
else:  # Flow in deformed domain
  md.add_nonlinear_term(mim_f, "mu*(Grad(v)*Inv(F(u_f))):(Grad(Test_v)*Inv(F(u_f)))-p*Trace(Grad(Test_v)*Inv(F(u_f)))" # stokes problem
                              "+Trace(Grad(v)*Inv(F(u_f)))*Test_p")
  md.add_nonlinear_term(mim_f, f"100*(v+v_in*Normalized(Inv(F(u_f))*Normal)*(1-sqr(X(2)/{H_fluid/2}-1))).Test_v", XM_RG)

md.add_nonlinear_term(mim_f, "100*(p-p_out)*Test_p", XP_RG)
md.add_nonlinear_term(mim_s, "Interpolate(p,fluid)*(J(u_s)*Inv(F(u_s))'*Normal).Test_u_s", YM_RG) # Only Pressure dynamic Coupling Missing the full fluid strain


md.solve("noisy", "max_iter", 20, "max_res", 1e-8,
         "lsearch", "simplest", "alpha max ratio", 3., "alpha min", 0.02, "alpha mult", 0.6)

mfu_f.export_to_vtu("fluid.vtu", mfu_f, md.variable("u_f"), "Displacements",
                                 mfv1, md.variable("v"), "Velocities",
                                 mfp1, md.variable("p"), "Pressure")
mfu_s.export_to_vtu("beam.vtu", mfu_s, md.variable("u_s"), "Displacements")


