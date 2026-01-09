# Potential topics for using Deep NN in Forecasting

- risk forecasting (e.g. Value at Risk, tail losses, expected shortfall) --> Timo
- conformal predictions
- transfer learning (maybe use for xLSTM)
- Predicting port congestion, shipping delays, or raw material shortages with AIS Data: Real-time location data of global cargo ships. Model this as a graph network to see how a delay in Shanghai has effects on prices in Germany.

## Data sources

- Polymarket or Manifold.markets
- analytics data for stock return forecasting (reports from investment banks and research firms). according to [this paper](https://www.sciencedirect.com/science/article/pii/S0957417421009441) it has rarely been used
- [Gdelt](https://www.gdeltproject.org/data.html#documentation). Useful paper: https://ojs.aaai.org/index.php/AAAI/article/view/30383. Potentially good to do an event study. Maybe in combination with Polymarket data.