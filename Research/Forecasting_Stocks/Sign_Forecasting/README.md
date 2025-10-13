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
- Potentially dropout variables on Stock Market as it might be too closely related to S&P 500 price
- These variables are not standardized as the NN should handle different scales inherently

## Variables
#### Forecasting directional movements of stock returns (daily)
- Daily returns lags 1-20, lag40, lag60, ... lag, 240 used in [this](https://pdf.sciencedirectassets.com/273054/1-s2.0-S1544612322X00032/1-s2.0-S1544612321003202/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJ%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDvb7kARhedri2mkifmQ6YA6GQ%2BsWiICTXxzzRmuJ1XWgIgNirfMh8FDamOSBCqwYbp4HrgdSx7F5jJjkjgRSv6RDgqswUISBAFGgwwNTkwMDM1NDY4NjUiDDlj6kxrnJjqpj9%2FKyqQBXfF574aKoqSO%2FvTboHoo9u0WYQndP%2BzU4Wpzp0GaGoqTitW4nGBxyoXDlFCcmetFx7uN8znh9FJGYhP1j49DnTL1x1Fz08wsQhnlZxhk7gnifwxUhed7LV9kEGij%2BBdODR%2BY%2B8wywKcShut0VolnHT0ZxAI9rq9sevw7iV4OY3U4dxAIg8tsr1220t5%2BmvctgZensyKPy18zMWQQ3OhtVP2X8%2B4z%2F6OqfdRh7n1FOSFrnddAJSOeOBCL%2FXit8%2FlawE9h5ekx6MTCTEjBfIAI5bdC7s0whUZsRhabe26l%2BxNSchBbuUNvDhv0sOMNhuacnyeX5PC6kx3NVSfjG7cwygdoy5TQpTBjr8Bv1FU6PbqgBjbEFk4pIi9jq7MLQn%2FIq6IRsfe6MyFD4E2dYBvyDASpCap5MOf4QplXq%2BTVbV3%2FvHR%2BhwNJiym8BeSTk5yPJGkJp9CfgHc9uT7BGw87jcRqUAnaRb1z%2FEsFbLjyah%2BykUvF1K8FVPWMP4KeMHu169nhSOVtj5VlM33ThZ9jvp6cqSOR6z7fPz1jloBqQ1uV6saglhdiuBS%2FObFlQ%2F5HjImUQNRcHlpMuaA3pcpNYFtj31M%2Bm18YyV10uhYN5dqk3aM%2F4zNXJtMcf1HI%2FPVoaPnPGTopYHHVpmeRg0j5XR8N7zz5tZEWeI4iNzYT6MYQqPweuPUwH1t0H538NjhRGBdL5fpe9isPiwclXAbX7S%2FR8qRad%2FlgRaMojF%2BsK4DyHy009S2GhIeEjVtEDI98Am7pkH78JdLtjXl%2BEaRqbWr0oOUhdVfS%2Fq9bWDbefe%2BCWd%2BTDSrR1o1tWb1yOdPYkW44o8lTfhh3fJoBNt%2FR0PUq53eZinhHaVVAZkPbUKpMPuktMcGOrEBLce%2FpD82zji8qYqWWudGI2KetK2rf3ibURMConDkfOgTsvKhx1%2FvHtApKbPPssLQc2v81J8zvSiTXcNoPDBXfEAbFoC1PeTOdmsWXmG1auesEpHUw9Rl3DIJL3qVNmXyNu72ME8387NHopCW36JZdbN2uSak24cTfhjOhUrOaEAJCz%2B%2B2fduy6TIYMGHOz%2BZ6hSbklbInvpeuSHoEHidcEHLgkD6xLlUghGhWSQOtSMs&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20251013T154201Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRE6DBNH5%2F20251013%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=1deb177052509cf051bb9147f2f5e26d9344c4d484454cdf9b14595f6c34f10c&hash=a20690de278e1cdb0c78ddd640ec779b2e024cf2894ecb63b1d7c078e6ef9f54&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1544612321003202&tid=spdf-8012b1d0-4b20-426b-b223-3f88be022e2f&sid=510151aa67e462423919b33547c6ea2f32c9gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1e015c5b0b0809560000&rr=98dff1cb194f6add&cc=de) paper.

![alt text](img/image.png)

#### Directional Predictability of stock returns (daily)
- log realized variances, realized skewness (arguments from Christofferson)
- S&P returns, realized betas, VIX, log realized variance of S&P 500 (they predicted single stocks from Dow Jones)
- variance premium from Bollerslev 2014, also very good predictor for monthly stock returns
- first three PC of US bonds with >40 different maturities

![alt text](img/image-1.png)



### Architecture of Neural Networks
- build separate networks for sign forecast and volatility forecast
- creating one network with two output neurons would restrict the network to learn weights for both outputs until the last layer and only then differentiate and create own weights for sign and volatility. this seems like an unnecessary restriction.