{% set excluded = inherited_members | list + ["__init__", "training"] %}

#############################################
{{ name | escape }}
#############################################

.. currentmodule:: {{ module }}

.. autoclass:: {{ name | escape }}

{% if methods | reject("in", excluded) | list | length %}

***********************
Methods
***********************

{% for m in methods | reject("in", excluded) %}
.. automethod:: {{ name }}.{{ m }}
{% endfor %}
{% endif %}

{% if attributes | reject("in", excluded) | list | length %}

***********************
Attributes
***********************

{% for attr in attributes | reject("in", excluded) %}
.. autoattribute:: {{ name }}.{{ attr }}
{% endfor %}
{% endif %}
