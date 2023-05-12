# Navier-Stokes solver FemFlow

Output of program
```
time ./femflow 0.5
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
Energy solver average iterations: 1 with accumulated error 0.0014


real	5m44.027s
user	5m43.150s
sys	0m0.428s

```
