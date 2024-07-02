# WeatherPredictor
A supervised machine learning model to forecast hourly temperatures for the next 24 hours based on past data. This uses numpy, pandas, matplotlib, statsmodels, and sklearn.

Made by Vinith Yedidi, July 2 2024

## The Data
I obtained a dataset of weather observations at Boston Logan International Airport from 1943-2023 from the University of Iowa. This data can be found here: https://mesonet.agron.iastate.edu/request/download.phtml.

## The Model
I applied basic machine learning techniques to the dataset in order to find a pattern and forecast future temperature values. The model's structure regresses the hourly in temperature, delta_t, with correlated factors of weather such as humidity, windspeed, pressure, hourly change in temperature, hourly change in humidity, and the hour of day. I found that all of these features have some correlation by examining a correlation heatmap (see "Graphs"). 

I then used a VAR model using the statsmodels library. VAR is a type of multivariate model that allows you to regress all the regressors along with the independent variable, allowing for multistep forcasting. In this case, we want to forecast all of the weather determineants hourly so as to forecast the hourly temp based on our previous forecasts. Naturally, it loses precision after some time, in this case about 12 hours, but considering how few determinants are in the model, it does fairly well.

I used hour as the exogenous variable (as it's predetermined what hour it's going to be), and all the other varibles as endogenous so that they can be forecasted on an hourly basis.

(To see more about this, check this out: https://otexts.com/fpp2/VAR.html. I used this to learn about forecasting and VAR models).

## The Results
As seen in the Python file, the error metrics for temparature prediction are as follows:
| Auto Regressive Error: | NOAA Comparative Error |
| ---------------------- | ---------------------- |
| RSME: 16.77            | RMSE: 6.43             |
| MAE:  14.22            | MAE:  5.83             |

These numbers could be lowered if more regressors were introuced, or if the forecasting length was reduced. It appears from the graphs that accuracy in forecasting diminishes greatly after 12 hours. However, given the constraints of this model, I believe that this is a very good result.

## Conclusion
To improve on this model, I would introduce factors such as discrete solutions to the fundamental equations of weather forecasting, built off of the Navier Stokes equations (https://maths.ucd.ie/~plynch/Publications/pcam0159_proof_2.pdf). Of course solutions to these equations are difficult, as it hasn't even been proven that there *are* solutions. But using discrete methods and vast amounts of data from NOAA to increase the resolution of the approximations, it's possible to arrive at satisfactory results. Unfortunately, it takes the NOAA supercomputer to make real forecasts like these, but in the future I would like to delve into it.

I learned a lot about machine learning methods and how they can be used to accurately forecast. VAR is suitable for choosing exogenous and endogenous variables and iteratively forecasting the endogenous variables to predict temperature hours in advance. The model performs fairly well, though there are improvements in scale and technique to be made. Overall, this was a very fun project and I hope to do more in machine learning in the future.

- Vinith Yedidi

## Sources
https://otexts.com/fpp2/
https://maths.ucd.ie/~plynch/Publications/pcam0159_proof_2.pdf
