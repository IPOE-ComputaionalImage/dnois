###################################
Optical System
###################################

.. automodule:: dnois.optics

.. _modeling_optical_systems:

************************************
Modeling optical systems
************************************

.. _abstract_models:

Abstract models
====================================
TODO

.. autosummary::
    :toctree: ../generated/optics/abstract

    IdealOptics
    Pinhole
    RenderingOptics
    StandardOptics

Implemented models
=====================================

The optical response of a given system varies when different ways
are chosen to model it. Currently, there are two categories of methods
to model optical systems: :doc:`optics/df` and :doc:`optics/rt`.
The following content implements :ref:`abstract_models` in these two ways.

.. toctree::
    :maxdepth: 2

    optics/df
    optics/rt

.. _image_formation:

***********************************
Image formation
***********************************
TODO: description

.. autosummary::
    :toctree: ../generated/optics/formation

    simple
    depth_aware
    space_variant
    superpose

**************************************
Optics
**************************************
TODO: description

.. autosummary::
    :toctree: ../generated/optics/o

    circle_of_confusion
    imgd
    objd
    norm_psf
