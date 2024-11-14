"""
This package implements various optical systems, which takes as inputs the
scene information and computes the radiance field impinging on the sensor plane.
See :doc:`/content/guide/optics/imodel` for more information.

The content concerning optical system can be roughly divided into two parts:
:ref:`modeling_optical_systems` and :ref:`image_formation`.
The former refers to the calculation of optical response, such as point spread
function (PSF) of the optical system, while the latter means the ways to
render the image produced by it given a ground truth image and optical response.
Note that optical response is not always specified by PSF. For example,
a group of rays can dictate rendering process as well in ray-tracing-based
systems.
"""
from .df import *
from .rt import *
from .system import *

from . import df, rt, system
