# Datasets

The data used in this project was originally obtained from [1].
It comprises almost 20 months of measurements, sampled every second from 5am to 8pm.
Several HDF5 datasets can be obtained using the included
[Jupyter Notebook](https://github.com/iipr/solar-irradiance/blob/master/data/etl-data.ipynb):

- Raw Global Horizontal Irradiance (GHI), including *NaNs* (-99999).
- Raw GHI where *NaNs* are filled using neighbours and previous recordings.
- Standardised GHI.
- Clear-Sky Index (CSI).
- Irradiance maps of standardised GHI. Map shapes of 8 × 8 and 10 × 10.

```
[1] Sengupta, M.; Andreas, A. (2010). Oahu Solar Measurement Grid (1-Year Archive):
1-Second Solar Irradiance; Oahu, Hawaii (Data); NREL Report No. DA-5500-56506.
http://dx.doi.org/10.5439/1052451 
```

