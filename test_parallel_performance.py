import getfem as gf
import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def log(msg=""):
    if rank == 0:
        print(msg)

log("=" * 70)
log(f"GetFEM Parallel Performance Test — {size} MPI process(es)")
log("=" * 70)

#############################################
# TEST 1: 3D Poisson
#############################################

log("\n--- TEST 1: 3D Poisson Problem ---")

N = 20

m = gf.Mesh('cartesian',
            np.linspace(0, 1, N+1),
            np.linspace(0, 1, N+1),
            np.linspace(0, 1, N+1))

mf = gf.MeshFem(m, 1)
mf.set_fem(gf.Fem('FEM_QK(3,1)'))

mim = gf.MeshIm(m, gf.Integ('IM_GAUSS_PARALLELEPIPED(3,4)'))

ndof = mf.nbdof()
nelem = m.nbcvs()
log(f"  Mesh: {N}x{N}x{N} = {nelem} elements, {ndof} DOFs")

md = gf.Model('real')
md.add_fem_variable('u', mf)
md.add_Laplacian_brick(mim, 'u')
md.add_Dirichlet_condition_with_multipliers(mim, 'u', mf, 1)

md.add_initialized_fem_data('f', mf,
    md.interpolation('sin(3.14*X(1))*sin(3.14*X(2))*sin(3.14*X(3))', mf))
md.add_source_term_brick(mim, 'u', 'f')

comm.Barrier()
t0 = time.time()
md.solve('noisy', 'lsolver', 'mumps')
comm.Barrier()
t1 = time.time() - t0

log(f"  Solve time: {t1:.4f} s")
log(f"  max|u| = {np.max(np.abs(md.variable('u'))):.6e}")

#############################################
# TEST 2: 2D Elasticity
#############################################

log("\n--- TEST 2: 2D Elasticity ---")

N2 = 100

m2 = gf.Mesh('cartesian',
             np.linspace(0, 10, N2+1),
             np.linspace(0, 1, N2//10+1))

mf2 = gf.MeshFem(m2, 2)
mf2.set_fem(gf.Fem('FEM_QK(2,1)'))

mim2 = gf.MeshIm(m2, gf.Integ('IM_QUAD(3)'))

ndof2 = mf2.nbdof()
nelem2 = m2.nbcvs()
log(f"  Mesh: {nelem2} elements, {ndof2} DOFs")

md2 = gf.Model('real')
md2.add_fem_variable('u', mf2)

E = 1e6
nu = 0.3
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
md2.add_initialized_data('lambda', lam)
md2.add_initialized_data('mu', mu)
md2.add_isotropic_linearized_elasticity_brick(mim2, 'u', 'lambda', 'mu')
md2.add_initialized_data('gravity', [0, -1000])
md2.add_source_term_brick(mim2, 'u', 'gravity')

flst = m2.outer_faces()
fnor = m2.normal_of_faces(flst)
left_faces = flst[:, np.where(fnor[0,:] < -0.5)[0]]
LEFT = 101
m2.set_region(LEFT, left_faces)
md2.add_Dirichlet_condition_with_multipliers(mim2, 'u', mf2, LEFT)

comm.Barrier()
t0 = time.time()
md2.solve('noisy', 'lsolver', 'mumps')
comm.Barrier()
t2 = time.time() - t0

log(f"  Solve time: {t2:.4f} s")
log(f"  max|u| = {np.max(np.abs(md2.variable('u'))):.6e}")

#############################################
# TEST 3: 2D Nonlinear
#############################################

log("\n--- TEST 3: 2D Nonlinear Elasticity ---")

N3 = 100

m3 = gf.Mesh('cartesian',
             np.linspace(0, 1, N3+1),
             np.linspace(0, 1, N3+1))

mf3 = gf.MeshFem(m3, 2)
mf3.set_fem(gf.Fem('FEM_QK(2,1)'))

mim3 = gf.MeshIm(m3, gf.Integ('IM_QUAD(3)'))

ndof3 = mf3.nbdof()
nelem3 = m3.nbcvs()
log(f"  Mesh: {N3}x{N3} = {nelem3} elements, {ndof3} DOFs")

md3 = gf.Model('real')
md3.add_fem_variable('u', mf3)
md3.add_initialized_data('mu', 0.5e6)
md3.add_initialized_data('lam', 1e6)

md3.add_nonlinear_term(mim3,
    "lam/2*sqr(Trace(Grad_u+Grad_u'+Grad_u'*Grad_u))*Id(2):Grad_Test_u"
    "+ mu*(Grad_u+Grad_u'+Grad_u'*Grad_u):Grad_Test_u")

flst3 = m3.outer_faces()
fnor3 = m3.normal_of_faces(flst3)
left3 = flst3[:, np.where(fnor3[0,:] < -0.5)[0]]
right3 = flst3[:, np.where(fnor3[0,:] > 0.5)[0]]
m3.set_region(201, left3)
m3.set_region(202, right3)

md3.add_Dirichlet_condition_with_multipliers(mim3, 'u', mf3, 201)
md3.add_initialized_data('pull', [0.01, 0])
md3.add_Dirichlet_condition_with_multipliers(mim3, 'u', mf3, 202, 'pull')

comm.Barrier()
t0 = time.time()
md3.solve('noisy',
          'lsolver', 'mumps',
          'max_iter', 50,
          'max_res', 1e-8)
comm.Barrier()
t3 = time.time() - t0

log(f"  Solve time: {t3:.4f} s")
log(f"  max|u| = {np.max(np.abs(md3.variable('u'))):.6e}")

#############################################
# SUMMARY
#############################################

log("\n" + "=" * 70)
log(f"SUMMARY — {size} MPI process(es)")
log("=" * 70)
log(f"  Test 1 (3D Poisson,    {ndof:>6} DOFs): {t1:.4f} s")
log(f"  Test 2 (2D Elasticity, {ndof2:>6} DOFs): {t2:.4f} s")
log(f"  Test 3 (2D Nonlinear,  {ndof3:>6} DOFs): {t3:.4f} s")
log(f"  Total: {t1 + t2 + t3:.4f} s")
log("=" * 70)