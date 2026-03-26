.. _using_measured_data:

Using Measured Environmental Data
==================================

This example runs one scenario using measured environmental inputs:

- GEBCO bathymetry (seafloor)
- Copernicus temperature/salinity converted to sound speed via Leroy's equation

It demonstrates how to wire real geophysical datasets into the Blue Pebble pipeline,
replacing the analytical models used in the other examples with data-driven alternatives
for bathymetry and sound speed profile.

.. note::

   This example requires external data files (GEBCO bathymetry and Copernicus ocean
   reanalysis) that are not bundled with the repository.  The figures below were
   pre-generated from a local run with the measured data.  To regenerate them, set
   ``save_figures = True`` in the Simulation Parameters section of the script and run
   it with the data files present.

   Data can be obtained from:

   - GEBCO Compilation Group, `The GEBCO Grid (GEBCO_2024 Grid) <https://www.gebco.net>`_.
   - E.U. Copernicus Marine Service Information (https://doi.org/10.48670/moi-00016).

:download:`Download the script <../../examples/using_measured_data.py>`

.. gallery-script:: ../../examples/using_measured_data.py
