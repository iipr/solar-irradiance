# Spatio-Temporal Solar Irradiance Forecaster (ST-SIF)

This repository gathers and expands results obtained with a Deep Learning
system for solar irradiance forecasting. It is organised as follows:

- **Data**: The data that was used in the experiments is subject to distribution restrictions.
    Quoting the [website](https://midcdmz.nrel.gov/apps/sitehome.pl?site=OAHUGRID) where the raw data was obtained from:
    _These data and any subset may not be publically redistributed by any means._
    Thus, a [Jupyter Notebook](https://github.com/iipr/solar-irradiance/blob/master/data/etl-data.ipynb)
    is provided to reproduce how the raw data was transformed to train the models.
    Detailed information can be found [here](https://github.com/iipr/solar-irradiance/blob/master/data/data.md).

- **Result tables**: Skill scored by each model for every horizon alongside model
    hyperparameters. Also, robustness tests for all models that work with irradiance maps.
    Detailed information can be found [here](https://github.com/iipr/solar-irradiance/blob/master/tables/tables.md).

- **Graphs**: Sensor distribution map, boxplots of the skill per horizon, skill maps, robustness tests,
    sample predictions, model representation as graphs, animated irradiance maps...
    Detailed information can be found [here](https://github.com/iipr/solar-irradiance/blob/master/graphs/graphs.md).

- **Source**: Python source code developed to train and test the solar irradiance models, which can be found
    [here](https://github.com/iipr/solar-irradiance/blob/master/src/).

![](https://github.com/iipr/solar-irradiance/blob/master/graphs/sensor_distribution.png)

