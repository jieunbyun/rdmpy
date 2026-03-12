User Guide
==========

This guide walks through the practical workflow for preparing and processing rail data.

Data Cleaning
~~~~~~~~~~~~~

Before running the preprocessor, you must clean the schedule data file.

The NWR Schedule data comes as a newline-delimited JSON (NDJSON) file containing five sections:

1. JsonTimetableV1 - Header/metadata
2. TiplocV1 - Location codes
3. JsonAssociationV1 - Train associations
4. JsonScheduleV1 - Schedule data (**this is what we need**)
5. EOF - End of file marker

**How to Clean the Schedule**

Run the schedule cleaning script:

.. code-block:: bash

   python demo/data/schedule_cleaning.py

This extracts the JsonScheduleV1 section and saves it as a cleaned pickle file:

.. code-block:: text

   CIF_ALL_FULL_DAILY_toc-full_p4.pkl

**Important:** The "p4" suffix refers to the 4th section being extracted. The preprocessor expects this cleaned file and will use it automatically.

Data Pre-Processing
~~~~~~~~~~~~~~~~~~~

After cleaning the schedule data, run the preprocessor to match schedules with delays and organize results by station.

**What the Preprocessor Does**

The preprocessor:

1. Loads the cleaned schedule data
2. Loads the delay attribution (Transparency) files
3. Matches scheduled trains with actual delays
4. Organizes data by station code
5. Saves results in the ``processed_data/`` folder

**Output Structure**

After preprocessing, the ``processed_data/`` folder is organized as:

.. code-block:: text

   processed_data/
   ├── <STANOX_CODE_1>/
   │   ├── MO.parquet
   │   ├── TU.parquet
   │   └── ...
   ├── <STANOX_CODE_2>/
   │   ├── MO.parquet
   │   ├── TU.parquet
   │   └── ...
   └── ...

Each station has its data organized by day of the week (MO, TU, WE, TH, FR, SA, SU for Monday to Sunday).

**Running the Preprocessor**

The preprocessor can be run with different options:

Process All Stations
---------------------

To process all category stations (A, B, C1, C2):

.. code-block:: bash

   python -m rdmpy.preprocessor --all-categories

This is recommended for comprehensive network analysis. **Note:** This takes approximately 1 full day to complete.

Process by Category
--------------------

To process stations by DFT category:

.. code-block:: bash

   python -m rdmpy.preprocessor --category-A
   python -m rdmpy.preprocessor --category-B
   python -m rdmpy.preprocessor --category-C1
   python -m rdmpy.preprocessor --category-C2

Process a Single Station
-------------------------

To test or process a specific station:

.. code-block:: bash

   python -m rdmpy.preprocessor <STANOX_CODE>

Replace ``<STANOX_CODE>`` with the station's numeric code (e.g., ``50001``).

**Important Considerations**

- **Partial Processing Impact**: If you only process a subset of stations (e.g., one category), the aggregate demos will show incomplete network data. See the :doc:`troubleshooting` guide for details.
- **Processing Time**: Full preprocessing takes significant time. Run during off-peak hours if possible.
- **Disk Space**: Ensure adequate disk space for processed data files.
- **No Interruption**: Avoid interrupting the preprocessor mid-run to prevent data inconsistency.

Next Steps
~~~~~~~~~~

After preprocessing completes:

1. Run the demos in the ``demo/`` folder for different analytical perspectives
2. Explore the data using the analysis tools

See the :doc:`api` for detailed API documentation.
