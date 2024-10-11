{% set excluded = inherited_members | list + ["__init__", "training"] %}

#############################################
{{ name | escape }}
#############################################

.. currentmodule:: dnois.mt

.. autoclass:: {{ name | escape }}

    :param str name: Name of the material.
    :param float min_wl: Minimum applicable wavelength in ``default_unit``. Default: 0.
    :param float max_wl: Maximum applicable wavelength in ``default_unit``. Default: infinity.
    :param str default_unit: Default unit for wavelength.

***********************
Methods
***********************

.. automethod:: {{ name }}.n

***********************
Attributes
***********************

{% for attr in attributes %}
.. autoattribute:: {{ name }}.{{ attr }}
{% endfor %}
   