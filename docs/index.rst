rdmpy: UK Rail Incidents & Delay Propagation
==============================================

A comprehensive toolkit for analyzing UK rail network incidents and delay propagation. Integrating train schedules, incident records, and delay data from the Rail Data Marketplace to link system-level performance metrics with component-level operational events.

This toolkit enables researchers and engineers to build unified datasets connecting network-wide delays and cancellations with temporal and spatial patterns of initial incidents, train movements, and passenger impacts—addressing a critical gap in data availability for systems engineering in the rail domain.

Overview
--------

The rdmpy repository serves to:

* **Integrate diverse data sources**: Combine schedule, incident, and delay data from the Rail Data Marketplace
* **Preprocess railway operations data**: Clean and structure raw data into analysis-ready formats
* **Support delay propagation analysis**: Track how local incidents cascade across the rail network
* **Enable system-level assessment**: Evaluate station performance, identify delay patterns, and explore cause-and-effect relationships
* **Provide interactive visualizations**: Explore data through multiple perspectives (aggregate, incident, station, time-based, and train-level views)

Repository Structure
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ├─ rdmpy/                     # Main package
   │  ├─ preprocessor/           # Data preprocessing module
   │  └─ outputs/                # Analysis tools and data loaders
   ├─ demo/                      # Interactive Jupyter notebooks
   │  ├─ aggregate_view.ipynb
   │  ├─ incident_view.ipynb
   │  ├─ station_view.ipynb
   │  ├─ time_view.ipynb
   │  └─ train_view.ipynb
   ├─ docs/                      # Documentation (Sphinx)
   ├─ tests/                     # Unit and integration tests
   ├─ processed_data/            # Preprocessed datasets by station
   └─ LICENSE, README.md

Key Features
~~~~~~~~~~~~

* **Unified Dataset**: Connects system-level performance with component-level events
* **Flexible Preprocessing**: Process all stations or target specific categories (A, B, C1, C2)
* **Multiple Analytical Perspectives**: Five interactive demos for different levels of network analysis
* **Station Performance Assessment**: Evaluate performance under varying demand scenarios
* **Delay Propagation Tracking**: Understand how incidents ripple across the network
* **Open & Extensible**: Built for integration with new data sources and analysis methods

Quick Start
~~~~~~~~~~~

1. **Install the toolkit**:

   .. code-block:: bash

      pip install -e .

2. **Download data** from the `Rail Data Marketplace <https://raildata.org.uk/>`_:

   - NWR Historic Delay Attribution data (Transparency files)
   - NWR Schedule data (CIF_ALL_FULL_DAILY_toc-full.json.gz)

3. **Preprocess the data**:

   .. code-block:: bash

      python -m rdmpy.preprocessor --all-categories

4. **Explore the data** using the Jupyter notebooks in the `demo/` folder

For detailed setup instructions, see :doc:`getting_started`.

License
-------

This project is licensed under the MIT License. See the LICENSE file for details.

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   getting_started
   api
   how_to_contribute
   contributors

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
