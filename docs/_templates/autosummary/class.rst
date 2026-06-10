{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :member-order: bysource

   {% block attributes %}
   {% set _excluded = ["T_destination", "call_super_init", "dump_patches", "training", "rep_size"] %}
   {% set _attrs = attributes | reject("in", _excluded) | list %}
   {% if _attrs %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in _attrs %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}
   {% endif %}
   {% endblock %}
