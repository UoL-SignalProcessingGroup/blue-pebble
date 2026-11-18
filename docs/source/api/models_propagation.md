# `bluepebble.models.propagation`

Propagation models for analytical, rtrs-backed and Bellhop eigenray acoustic simulation workflows.

## Choosing a ray model

{class}`~bluepebble.models.propagation.rtrsAcousticPropagationModel` and
{class}`~bluepebble.models.propagation.BellhopAcousticPropagationModel` both implement
{class}`~bluepebble.models.propagation.SpectrumPropagationModel`, so either can be the
`propagation_model` of any simulator. rtrs traces 3D rays in-process; Bellhop runs the external
`bellhopcxx` program (see the installation page) in 2D vertical planes.

- Bellhop has two on/off settings for the array. By default it traces to every sensor in one
  run, in the vertical plane towards the array's reference point, with every sensor at its own
  range and depth; that is exact when the bathymetry is the same on every bearing.
  `separate_planes=True` (2D only) makes one run per sensor, in its own plane, for a seabed that
  varies across the bearings the array spans. `far_field=True` traces to the reference point
  only and gives every sensor the same arrivals, delayed by its plane-wave offset, so its
  transfer functions cost little more than one sensor's. It suits a compact array with the
  target in its far field, and ignores wavefront curvature and multipath differences across the
  array. Far field has one plane, so `separate_planes=True` with it is ignored, with a warning.
  The older `receiver_geometry` still works but is deprecated.
- {class}`~bluepebble.models.propagation.Bellhop3DAcousticPropagationModel` runs Bellhop3D once
  for the whole array, with every sensor at its own bearing, over a gridded seabed;
  `separate_planes` doesn't apply and is ignored, with a warning. It traces rays in three dimensions, so it
  follows paths that leave their vertical plane: over a 3% cross-slope at 3 km the
  bottom-reflected delay was within 3 us of the exact image-source delay, where the 2D model
  was 10-20 us out.
  Full 3D is about ten times slower than 2D, so prefer the 2D model when the seabed is the same
  on every bearing.
- Ray models need fine enough beams to converge. In deep isovelocity water, Bellhop's default
  2000 beams put the transfer function within 0.3% of the analytic field. rtrs needed an
  elevation resolution of about 0.05 degrees (its default is 1 degree) to agree with it.

```{eval-rst}
.. automodule:: bluepebble.models.propagation
   :members:
```
