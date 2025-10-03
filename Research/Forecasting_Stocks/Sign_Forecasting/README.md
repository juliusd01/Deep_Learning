# Sign Forecasting of Asset Returns using DL

Central Idea: 
![alt text](img/basic_idea.png)

- volatility dynamics and non-zero asset return make asset return sign forecastable
- forecastability is highest when average return is high and volatility is 
- biggest responsiveness in signs is at medium time horizons (20-40 trading days = 2-3 months)


also look at [this](https://www.sciencedirect.com/science/article/pii/S106294081730400X?casa_token=y68X7c_uhg8AAAAA:yBf9pr5pouUoMQ0M-ni2ZGBplqIOhsoDMeLGgn0DPkdHEIq4AAJ7_TWebZhtmtDVMMpR3ea_rsU)

## Use cases for Deep Learning
- just substituting logistic function in empirical part of Christofferson and Diebold will not be worth writing a paper about
- first option: I replicate their results, use a DL model on same data and then build third model with more data sources
- second option: move away from their paper but still do sign forecasting, just where DL models can be more powerful, e.g. high-frequency returns or using detailed data from order books
- sign forecasting for other assets, e.g. bonds while doing sentiment analysis (maybe focus on the time horizon)


# Data

See folder 'data'.

### Macro Databases 
- [FRED-MD](https://www.stlouisfed.org/research/economists/mccracken/fred-databases)
- Seulki used cubic spline interpolation to get the quartely data also as monthly data
- Potentially dropout variables on Stock Market or just use these values as the dependent variable