WOfS Filtered Statistics
##########################

Water Observations from Space Filtered Statistics is a set of gridded datasets derived from the WOfS-STATS dataset.
WO-FILT-STATS adds to the WO-STATS dataset to deliver surface water statistics of the WOfS data, filtered for a computed
confidence level. The confidence is produced using a logistic regression that compares the WOfS water summary to
several other datasets that inform of the presence of water in the landscape.

``wofs_filtered`` Computes the confidence layer, using the provided pickled model file. Then uses this
confidence layer to mask the WOfS water summary, and produces the filtered summary dataset.

The NetCDF file produced has two datasets:

- ``confidence``: The confidence (or probability) that a water observation in this location is correct.

- ``wofs_filtered_summary``: The percentage of clear observations on which water was detected, masked where the confidence is less than the provided threshold


Setup
=====

An example config file is located in ``configs/template_client.yaml``

Settings specified in config file:

Datacubes
---------
Specify in which datacube to find each required product.
Datacube names must match those in the provided datacube.conf file

.. code-block:: yaml

    datacubes:
        - prod: ['wofs_summary', 'wofs_filtered_summary',]


Factors
-------
This provides the required information on the input datasets (factors) used in calculating the confidence layer.

.. code-block:: yaml

  factors
    - {name: 'clearobs', product: 'wofs_summary', band: 'count-clear'}

The factor is called ``clearobs`` in the model, the product in the datacube is called ``wofs_summary``, and the relevant band is ``count-clear``

Output File Details
-------------------
``wofs_filtered_summary`` section defines output file information

Name to use for the confidence & filtered summary bands.

.. code-block:: yaml

    confidence: 'confidence'
    confidence_filtered: 'wofs_filtered_summary'

File name template with placeholders {} for tile index

.. code-block:: yaml

  filename: 'wofs_filtered_summary_{}_{}.nc'

The base directory where files will be written

.. code-block:: yaml

  filtered_summary_dir: '/g/data/u46/users/bt2744/work/data/wofs'

Confidence Filtering
--------------------
Path to the trained model, and the filtering threshold. A threshold of 0.1 will filter out all

.. code-block:: yaml

  confidence_filtering:
    trained_model_path: '/g/data/u46/wofs/confidence_albers/tmp/confidence_model.pkl'
    threshold: 0.1

Coverage
--------

Path to a file containing a list of tiles to process, called when ``--retile`` is used

.. code-block:: yaml

    tile_list_path: 'g/data/u46/users/bt2744/work/data/wofs/tile_list'


Running
=======

Run for Single Tile
-------------------

Use the ``wofs_filtered`` command to do a wofs confidence layer run.

--tile to specify which tile to process

--config option to refer to a ``yaml`` file

The configuration of the tool can be specified using --config option to refer to a ``yaml`` file.
An example config file, ``template_client``, is located in the ``configs`` directory.


Batch Run
---------

``scripts/job.pbs`` manages parallelization for larger runs

.. code-block::

  qsub job.pbs
