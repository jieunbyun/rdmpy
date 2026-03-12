Getting Started
===============

Installation
------------

We recommend using a virtual environment to manage dependencies. Follow these steps to set up your environment:

**Step 1: Create a Virtual Environment**

.. code-block:: bash

   python -m venv rdmpy_env

**Step 2: Activate the Virtual Environment**

On Linux/macOS:

.. code-block:: bash

   source rdmpy_env/bin/activate

On Windows:

.. code-block:: bash

   rdmpy_env\Scripts\activate

**Step 3: Install rdmpy**

.. code-block:: bash

   pip install rdmpy

To install in editable mode (for development):

.. code-block:: bash

   pip install -e .

Verify Installation
--------------------

After installation, you can verify that rdmpy is installed correctly by running:

.. code-block:: python

   import rdmpy
   print("rdmpy version:", rdmpy.__version__)


Development Setup
-----------------

Please read the :doc:`how_to_contribute` guide for the contributions welcomed.
If you'd like to contribute to rdmpy or modify the source code locally, follow these steps:

**Step 1: Fork and Clone the Repository**

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/your-username/rdmpy.git
   cd rdmpy

**Step 2: Create a Development Branch**

.. code-block:: bash

   git checkout -b feature/your-feature-name

**Step 3: Install in Development Mode**

With your virtual environment activated, install the project in editable mode:

.. code-block:: bash

   pip install -e .


Setting Up Your Development Environment
----------------------------------------

To set up a complete development environment with all dependencies:

.. code-block:: bash

   # Install core dependencies
   pip install -r requirements.txt
   
   # For documentation development
   pip install -r docs/requirements.txt
   
   # For testing
   pip install pytest


Running Tests
-------------

Run the test suite to ensure everything is working correctly:

.. code-block:: bash

   pytest tests/


Next Steps
-------------------------------
Please refer to the :doc:`user_guide` for instructions on how to download, preprocess data and use the analysis tools. For contributing to the project, see the :doc:`how_to_contribute` guide.
