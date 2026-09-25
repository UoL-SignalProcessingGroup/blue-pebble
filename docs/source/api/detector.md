# `bluepebble.detector`

Public detector algorithms, detector metrics helpers, and passive sonar detector entrypoints.

```{eval-rst}
.. automodule:: bluepebble.detector
   :members:
```

## Shared CFAR detector behaviour

`CACFARDetector` and `OSCFARDetector` share their threshold modes, noise calibration and peak
consolidation through a common base class, documented here.

```{eval-rst}
.. autoclass:: bluepebble.detector.algorithms._CFARDetectorBase
   :members: detect, detection_snr_map
```
