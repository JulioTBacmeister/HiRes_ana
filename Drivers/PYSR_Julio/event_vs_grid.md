
Footprint-average correction: SH x0.810, NH x0.734
(an event score times this is the pointwise r it implies)

| model | frame | fit | detail | SH r | NH r | SH r (pointwise) | NH r (pointwise) |
|---|---|---|---|---|---|---|---|
| NN tilt_levs+tilt+precl | event | both | 6 preds, MSE | 0.790 | 0.785 | 0.640 | 0.576 |
| NN tilt_levs+tilt+precl | event | both | 6 preds, |e|^5 | 0.762 | 0.758 | 0.617 | 0.557 |
| NN tilt_levs+tilt+precl | event | sh | 6 preds, |e|^5 | 0.761 | 0.710 | 0.617 | 0.521 |
| NN tilt_levs+tilt+precl | event | sh | 6 preds, MSE | 0.795 | 0.709 | 0.644 | 0.520 |
| PySR tilt | gridded | sh | cx=35 | 0.638 | 0.707 | 0.638 | 0.707 |
| PySR tilt_both | gridded | both | cx=34 | 0.635 | 0.703 | 0.635 | 0.703 |
| PySR full_both | gridded | both | cx=39 | 0.695 | 0.698 | 0.695 | 0.698 |
| NN tilt_levs+precl | event | sh | 2 preds, |e|^5 | 0.620 | 0.675 | 0.502 | 0.496 |
| PySR dp_both | gridded | both | cx=19 | 0.569 | 0.652 | 0.569 | 0.652 |
| PySR dp | gridded | sh | cx=23 | 0.574 | 0.570 | 0.574 | 0.570 |
| NN tilt_levs+tilt+fgf | event | sh | 9 preds, |e|^5 | 0.688 | 0.558 | 0.557 | 0.410 |
| NN tilt_levs+tilt | event | sh | 5 preds, |e|^5 | 0.685 | 0.539 | 0.555 | 0.396 |
| NN tilt_levs+fgf | event | sh | 5 preds, |e|^5 | 0.482 | 0.425 | 0.390 | 0.312 |

Calibration slopes on the event models (1.0 = correctly dispersed):
  NN tilt_levs+tilt                5 preds, |e|^5         SH 1.02   NH 0.65
  NN tilt_levs+precl               2 preds, |e|^5         SH 0.94   NH 0.92
  NN tilt_levs+tilt+precl          6 preds, |e|^5         SH 0.94   NH 0.74
  NN tilt_levs+fgf                 5 preds, |e|^5         SH 0.91   NH 1.07
  NN tilt_levs+tilt+fgf            9 preds, |e|^5         SH 0.99   NH 0.71
  NN tilt_levs+tilt+precl          6 preds, MSE           SH 1.09   NH 0.88
  NN tilt_levs+tilt+precl          6 preds, |e|^5         SH 0.99   NH 0.88
  NN tilt_levs+tilt+precl          6 preds, MSE           SH 1.01   NH 0.89
