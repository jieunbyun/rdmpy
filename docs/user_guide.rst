User Guide
==========

This guide walks through the practical workflow for preparing and processing the data used in rdmpy.

Data Download
-------------

Before you can clean and preprocess the data, you need to download the necessary files from the Rail Data Marketplace.

**Where to Find the Data**

All required datasets are available from the `Rail Data Marketplace (RDM) <https://raildata.org.uk/>`_. You will need to create an account to access these files.

**Required Files**

You need to download two main datasets:

1. **NWR Historic Delay Attribution (Transparency Data)**
2. **NWR Schedule Data**

**File Specifications and Location**

For Delays - Search "NWR Historic Delay Attribution"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under "data files", you will find .zip files organized by year and period. Download and extract them to find files named:

.. code-block:: text

   Transparency_23-24_P12.csv
   Transparency_23-24_P13.csv
   Transparency_24-25_P01.csv
   ...

**File Naming Convention:**

- ``Transparency`` refers to the Rail Delivery Group (RDG) transparency initiative for public operational data
- ``23-24`` stands for the financial year (April to March)
- ``P01`` is the month within the financial year (starting in April)

You may also find files named like ``202425 data files 20250213.zip`` or ``Transparency 25-26 P01 20250516.zip``, where the date at the end indicates the last entry date in the data itself.

For Schedule Data - Search "NWR Schedule"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under "data files", you will find:

.. code-block:: text

   CIF_ALL_FULL_DAILY_toc-full.json.gz

**File Details:**

- ``CIF`` = Common Interface Format
- ``toc-full`` = Train Operating Companies (TOC) Full Extract
- Format = Daily formats (but the full extent of data is weekly, containing all daily scheduled trains for a standard week)

**Setup Instructions**

Once downloaded, follow these steps:

1. Create a ``data/`` folder inside the ``demo/`` folder if it doesn't exist
2. Save all downloaded .csv files and the .json.gz file in ``data/`` without creating subfolders
3. For detailed specifications of each file and how to modify entries for different rail months/years, refer to:
   - ``incidents.py`` for delay file specifications
   - ``schedule.py`` for schedule file specifications

The tool will automatically detect and load these files from the ``data/`` folder.

**Reference Files**

Additional reference data files are provided in the ``reference/`` folder, including:

- Station reference files with latitude and longitude
- Station description and classification information

These are the only files directly provided and do not need to be downloaded separately.

Data Cleaning
-------------

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
--------------------

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

You can find further information on the preprocessor's functionality and troubleshooting tips in the :doc:`troubleshooting` guide.

Next Steps
----------

After preprocessing completes:

1. Run the demos in the ``demo/`` folder for different analytical perspectives
2. Explore the data using the analysis tools

See the :doc:`api` for detailed API documentation.
