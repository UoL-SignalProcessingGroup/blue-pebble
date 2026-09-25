.. _using_measured_data:

Using GEBCO and Copernicus Environmental Data
=============================================

This example runs one scenario in the environment of a real area south of the Faroe Islands,
built from oceanographic datasets:

- GEBCO bathymetry (seafloor)
- Copernicus temperature/salinity converted to sound speed via the NPL equation

Only the environment comes from data. The target signals, ambient noise and array output are
simulated, as in the other examples.

It demonstrates how to wire real geophysical datasets into the Blue Pebble pipeline,
replacing the analytical models used in the other examples with data-driven alternatives
for bathymetry and sound speed profile.

.. note::

   This example requires external data files (GEBCO bathymetry and Copernicus ocean
   analysis) that are not bundled with the repository.  The figures below were
   pre-generated from a local run with the data files.  To regenerate them, run
   ``docs/scripts/generate_using_measured_data_figs.py`` with the data files present.

   Data can be obtained from:

   - GEBCO Compilation Group, `The GEBCO Grid (GEBCO_2024 Grid) <https://www.gebco.net>`_.
   - E.U. Copernicus Marine Service Information (https://doi.org/10.48670/moi-00016).

:download:`Download the script <../../examples/using_measured_data.py>`

.. gallery-script:: ../../examples/using_measured_data.py
