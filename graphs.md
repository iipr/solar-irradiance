# Graphs

- [Sensor distribution map](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/fTFqB4Wx6PW8kgJ/preview).
  This plot shows the location of the pyranometers relative to one another
- [Boxplots of the skill per horizon](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/k7DqRJCeCJ3Q7n6).
  For each model, a plot depicting how skill distributes across sensors for every horizon.
- [Robustness tests](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/yBekwdywy5P4rWC).
  For each model that works with irradiance maps, we simulated how the skill evolves when fewer
  sensors are available. Thus, the irradiance map is interpolated with the available data on each prediction step.
  These plots were produced using data from [here](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/RMYmSm8pmd2BtYn).
- [Sample predictions](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/Sb7GLEyQiKFELs2).
  Series plots of the true vs. predicted variable (GHI or CSI) for each forecast horizon on a random day
  of each month from the test set.
- [Model representation as graphs](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/ZsKnRZXSFD8omXC).
  Obtained using Keras' [`plot_model`](https://keras.io/visualization/) function.
- [Animated irradiance maps](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/igoK6fyydLTkS5f).
  Animated graphs showing sample irradiance maps of shapes of 8 × 8 and 10 × 10.
  The red dots represent the values of the recorded standardised GHI for a certain pyranometer on each instant.
  The map (surface) is obtained using nearest neighbour interpolation with those values.

