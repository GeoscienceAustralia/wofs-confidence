Wofs Confidence Tool
====================

Use the ``wofs_confidence`` command to do a wofs confidence layer run. The configuration of
the tool can be specified using ``cfg`` option to refer to a ``yaml`` file. An example config
file is located in the ``configs`` directory. Other options it require are ``m`` that specify
a pickled model file (serialised file by the Python pickle module) and ``c`` option that specify
coverage in a ``yaml`` file.

The front-end of the ``wofs_confidence`` generate a tile file and a yaml config file that are
fed into a shell script that run the confidence layer tile computation script as a PBS job.
The PBS application file computes a single wofs confidence tile based on the yaml config file it
is fed. 

