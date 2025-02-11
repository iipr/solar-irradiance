# Results tables

- [Big table](https://github.com/iipr/solar-irradiance/blob/master/tables/results.ods)
  that summarises results for all relevant models. Some important columns are:
    - **Key**: Key or abbreviated name for a model with certain characteristics.
               From the key, the target variable, timestep and network type can be understood.
    - **Model**: The kind of neural network. The detailed architecture can be depicted in the
                 [model graph](https://github.com/iipr/solar-irradiance/blob/master/graphs/model-graph/).
    - **Target**: Variable that is predicted.
    - **Time granularity**: Frequency at which the data is recorded.
    - **Timestep**: Size of the time window that is fed as input.
    - **Forecast horizon**: Number of steps ahead for the prediction. There can be more than one,
                            and it uses the same units as the time granularity.
    - **Median of skills [%]**: For every forecast horizon, median skill scored by the 17 sensors.
    - **Mean of medians [%]**: Mean of the median of skills.

- [Robustness tests](https://github.com/iipr/solar-irradiance/blob/master/results/robustness-test/)
  for all models that work with irradiance maps. Rows represent how many sensors where excluded,
  and columns show independent repetitions of the test where different sensors were randomly excluded.
  Each file in the folder shows results for a single model, including:
    - Which sensors where excluded (their data were not available) for each repetition.
    - RMSE scored by the model using only the remaining sensors.
    - For every forecast horizon, median skill scored for the 17 locations where sensors are placed.
      With this information, plots were produced showing how skill evolves when fewer sensors are available,
      they can be depicted [here](https://github.com/iipr/solar-irradiance/blob/master/graphs/robustness-test/).
    - Mean of the previous values.  

