# `bluepebble.models.propagation`

Propagation models for analytical, rtrs-backed and Bellhop eigenray acoustic simulation workflows.

## Choosing a ray model

{class}`~bluepebble.models.propagation.rtrsAcousticPropagationModel` and
{class}`~bluepebble.models.propagation.BellhopAcousticPropagationModel` both implement
{class}`~bluepebble.models.propagation.SpectrumPropagationModel`, so either can be the
`propagation_model` of any simulator. rtrs traces 3D rays in-process; Bellhop runs the external
`bellhopcxx` program (see the installation page) in 2D vertical planes.

- Bellhop's `receiver_geometry="reference"` (the default) makes one run for the whole array in
  the plane towards its reference point, with every sensor at its own range and depth. That is
  exact when the bathymetry is the same on every bearing. Use `"per_sensor"` (one run per
  sensor) when the seabed varies across the bearings the array spans.
- Ray models need fine enough beams to converge. In deep isovelocity water, Bellhop's default
  2000 beams put the transfer function within 0.3% of the analytic field. rtrs needed an
  elevation resolution of about 0.05 degrees (its default is 1 degree) to agree with it.

```{eval-rst}
.. automodule:: bluepebble.models.propagation
   :members:
```
