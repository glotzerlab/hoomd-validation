{% extends base_script %}
{% block body %}
module load openmpi/4.0.5-gcc10.2.0
{{ super() }}
{% endblock %}
