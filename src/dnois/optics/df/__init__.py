"""
This package provides some functions and modules to compute various diffraction integral.
In general, they compute the complex amplitude on a (*target*) plane given that on another
(*source*) plane parallel to it. In their documentations and signatures, coordinates
on source plane are denoted as :math:`(u,v)` while those on target plane are denoted as
:math:`(x,y)`. Both source field and target field are sampled in a
evenly-spaced grid on respective plane. This is typically used to compute
the pulse response of an optical system given the pupil function :math:`U`.
It is assumed to have non-zero value only on a finite region around the origin,
covered by the region from which :math:`U` is sampled.

The theoretical bases of them include Maxwell's equations, scalar diffraction theory,
Sommerfeld radiation condition, which gives Rayleigh-Sommerfeld solution.
If Fresnel's approximation is valid, it is simplified to Fresnel diffraction.
If far field condition is satisfied further, it is simplified to Fraunhofer diffraction.
See "Goodman, Joseph W. Introduction to Fourier optics. Roberts and Company publishers, 2005."
for more details.

There are mainly three numerical ways implemented to compute these diffraction integrals.
*Fourier transform* (``ft``) accomplishes computation by only one Fourier transform.
*Angular spectrum* (``as``) transforms source field to frequency domain,
multiplies corresponding transfer function with analytical expression and
transforms it inversely. In contrast, *convolution* (``conv``) computes the convolution
between source field and a kernel function so the transfer function is not analytical.
As a result, there are six functions (or modules) available:

+-------------------+----------------------------------------------------------------------------------+
|                   |                                 Numerical method                                 |
| Mathematical form +------------------+------------------------------+--------------------------------+
|                   |Fourier transform |       Angular spectrum       |           Convolution          |
+===================+==================+==============================+================================+
|Fraunhofer         |:func:`fraunhofer`|none                          |none                            |
+-------------------+------------------+------------------------------+--------------------------------+
|Fresnel            |:func:`fresnel_ft`|:func:`fresnel_as`            |:func:`fresnel_conv`            |
+-------------------+------------------+------------------------------+--------------------------------+
|Rayleigh-Sommerfeld|none              |:func:`rayleigh_sommerfeld_as`|:func:`rayleigh_sommerfeld_conv`|
+-------------------+------------------+------------------------------+--------------------------------+

Note that in some literature the first Rayleigh-Sommerfeld solution is also called
angular spectrum method, whereas it refers to a numerical method here.
"""
from .prop import *

from . import prop
