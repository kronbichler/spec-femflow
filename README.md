# Navier-Stokes solver FemFlow

The FemFlow program solves the compressible Navier--Stokes equations with high-order
finite element methods. The equations are solved on locally refined meshes
and use a splitting methods, to treat the hyperbolic terms with a discontinuous
Galerkin method on hexahedral elements of a 3D mesh (polynomial degree 4, 
over-integration on 6 points per direction) explicitly in time with Heun's method,
while the parabolic terms are treated implicitly with a Crank-Nicolson scheme.
The associated linear system is solved separately for velocity and temperature
with a conjugate gradient method preconditioned by the matrix diagonal. 
The code relies on the deal.II finite element library, in
particular its matrix-free infrastructure to quickly compute the underlying 
integrals with sum factorization. In an optimally compiled way, the code is heavy
on SIMD vectorization and makes use of caches.

The program run time can be controlled by the parameter passed to the function.
For example, running up to physical time 0.5, we get with the gcc-12 compiler:
```
$ time ./femflow 0.5
Number of degrees of freedom: 320,000 ( = 5 [vars] x 512 [cells] x 125 [dofs/cell/var] )
Number of degrees of freedom: 705,000 ( = 5 [vars] x 1,128 [cells] x 125 [dofs/cell/var] )
Time step size: 0.00178508, minimal h: 0.392699, initial transport scaling: 0.0357016

Time:       0, dt:   0.0018, kinetic energy:  0.1250009, dissipation:  0.0004687
Time:   0.102, dt:   0.0018, kinetic energy:  0.1249549, dissipation:  0.0004688
Time:   0.202, dt:   0.0018, kinetic energy:  0.1249063, dissipation:  0.0004696
Time:     0.3, dt:   0.0018, kinetic energy:   0.125048, dissipation:   0.000472
Number of degrees of freedom: 1,020,000 ( = 5 [vars] x 1,632 [cells] x 125 [dofs/cell/var] )
Time:   0.401, dt:   0.0018, kinetic energy:  0.1249357, dissipation:  0.0004742
Time:   0.501, dt:   0.0018, kinetic energy:  0.1247678, dissipation:  0.0004765
Velocity solver average iterations: 2.8
Energy solver average iterations: 2 with accumulated error 0.0039


real	5m31.263s
user	5m30.573s
sys	0m0.380s
```

When compiled with `clang++-16 -std=c++17 -march=native -O3 -fopenmp-simd -ffp-contract=fast`,
a compiler that gets reasonably good code out of the abstractions in this benchmark,
the following run time is obtained:
```
$ time ./femflow 0.5
Number of degrees of freedom: 320,000 ( = 5 [vars] x 512 [cells] x 125 [dofs/cell/var] )
Number of degrees of freedom: 705,000 ( = 5 [vars] x 1,128 [cells] x 125 [dofs/cell/var] )
Time step size: 0.00178508, minimal h: 0.392699, initial transport scaling: 0.0357016

Time:       0, dt:   0.0018, kinetic energy:  0.1250009, dissipation:  0.0004687
Time:   0.102, dt:   0.0018, kinetic energy:  0.1249549, dissipation:  0.0004688
Time:   0.202, dt:   0.0018, kinetic energy:  0.1249063, dissipation:  0.0004696
Time:     0.3, dt:   0.0018, kinetic energy:   0.125048, dissipation:   0.000472
Number of degrees of freedom: 1,020,000 ( = 5 [vars] x 1,632 [cells] x 125 [dofs/cell/var] )
Time:   0.401, dt:   0.0018, kinetic energy:  0.1249357, dissipation:  0.0004742
Time:   0.501, dt:   0.0018, kinetic energy:  0.1247678, dissipation:  0.0004765
Velocity solver average iterations: 2.8
Energy solver average iterations: 2 with accumulated error 0.0039


real	0m49.049s
user	0m48.777s
sys	0m0.188s

```
