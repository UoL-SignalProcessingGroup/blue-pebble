.. _modelling_acoustic_sources:

Modelling Acoustic Sources
===========================

This example previews several standalone acoustic source and noise models, then combines
them into a simple composite soundscape. It is intended as a quick orientation for how
different source classes behave before they are embedded in a full propagation and
beamforming pipeline.

Passive sonar scenes often contain a mix of biological, anthropogenic, and ambient
contributors. Understanding the isolated time-frequency signature of each component
makes it easier to interpret later BTRs, spectrograms, and received mixtures.

.. note::

   The **Measured Vessel Noise** and **Composite Soundscape** sections require an
   external WAV recording (``SanctSound_CI05_03_largeship_20190925T135956Z.wav``)
   that is not bundled with the repository.  The remaining sections — whale call,
   snapping shrimp, commercial vessel tonals, and ambient white noise — are fully
   synthetic and can be run without any external data.

   The WAV file is available from
   `Sanct Sounds <https://sanctsound.ioos.us/sounds.html#Vessels>`_.
   Place it in ``docs/examples/measured_data/`` before running the script.

:download:`Download the script <../../examples/modelling_acoustic_sources.py>`

.. gallery-script:: ../../examples/modelling_acoustic_sources.py
