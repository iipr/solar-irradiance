# Deep Learning for solar irradiance forecasting
This repository gathers and expands results obtained with a Deep Learning
system for solar irradiance forecasting. It is organised as follows:

- **Datasets**: The data that was used in the experiments is subject to distribution restrictions.
    Quoting the [website](https://midcdmz.nrel.gov/apps/sitehome.pl?site=OAHUGRID) where the raw data was obtained from:
    _These data and any subset may not be publically redistributed by any means._
    Thus, a [Jupyter Notebook](https://github.com/iipr/solar-irradiance/blob/master/etl-data/etl-data.ipynb)
    is provided to reproduce how the raw data was transformed to train the models.
    Detailed information can be found [here](https://github.com/iipr/solar-irradiance/blob/master/data.md).

- **Result tables**: Skill scored by each model for every horizon alongside model
    hyperparameters. Also, robustness tests for all models that work with irradiance maps.
    Detailed information can be found [here](https://github.com/iipr/solar-irradiance/blob/master/tables.md).

- **Graphs**: Sensor distribution map, boxplots of the skill per horizon, skill maps, robustness tests,
    sample predictions, model representation as graphs, animated irradiance maps...
    Detailed information can be found [here](https://github.com/iipr/solar-irradiance/blob/master/graphs.md).

![](https://delicias.dia.fi.upm.es/nextcloud/index.php/s/fTFqB4Wx6PW8kgJ/preview)

