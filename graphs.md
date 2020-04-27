# Graphs

- [Sensor distribution map](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/fTFqB4Wx6PW8kgJ/preview).
  This plot shows the location of the pyranometers relative to one another
- [Boxplots of the skill per horizon](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/k7DqRJCeCJ3Q7n6).
  For each model, a plot depicting how skill distributes across sensors for every horizon.
- [Skill maps](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/DaqFP2GBpokj8YZ).
  For each model, a plot depicting how skill varies depenging on the location of the sensor for every horizon.
  If a map (or part of it) is showed in white, it means that the scored skill in that area is negative.
- [Robustness tests](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/yBekwdywy5P4rWC).
  For each model that works with irradiance maps, simulations were carried out to show how the skill evolves when fewer
  sensors are available. Thus, the irradiance map is interpolated with the available data on each prediction step.
  These plots were produced using results that are stored [here](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/RMYmSm8pmd2BtYn).
- [Sample predictions](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/Sb7GLEyQiKFELs2).
  Series plots of the true vs. predicted variable (GHI or CSI) for each forecast horizon on a random day
  of each month from the test set (April 2010, December 2010 and July 2011).
- [Model representation as graphs](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/ZsKnRZXSFD8omXC).
  Obtained using Keras' [`plot_model`](https://keras.io/visualization/) function.
- [Animated irradiance maps](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/igoK6fyydLTkS5f).
  Animated graphs showing sample irradiance maps of shapes of 8 × 8 and 10 × 10.
  The red dots represent the values of the recorded standardised GHI for a certain pyranometer on each instant.
  The map (surface) is obtained using nearest neighbour interpolation with those values.

