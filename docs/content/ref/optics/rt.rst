#####################################
Ray Tracing
#####################################
.. automodule:: dnois.optics.rt

*********************************
Characterizing rays
*********************************
.. autosummary::
    :toctree: ../../generated/optics/rt/ray

    BatchedRay
    NoValidRayError

**********************************
Optical surfaces
**********************************
.. autosummary::
    :toctree: ../../generated/optics/rt/surf

    surface_types
    Context
    CircularSurface
    Surface
    SurfaceList

Apertures
============================================
.. autosummary::
    :toctree: ../../generated/optics/rt/surf/aperture

    Aperture
    CircularAperture

Specific surface types
=================================
.. autosummary::
    :toctree: ../../generated/optics/rt/surf/types

    CircularStop
    Conic
    EvenAspherical
    Fresnel
    Planar
    PolynomialPhase
    Spherical
    Standard
    Stop

************************************
Ray-tracing-based optical systems
************************************
.. autosummary::
    :toctree: ../../generated/optics/rt/sys

    CoaxialRayTracing

.. _configuration_for_newtons_method:

***********************************
Configuration for Newton's method
***********************************
TODO
