## ToDos
- try to overfit the data and achieve near perfect accuracy for the training data
- Integrate mlflow to log all the metrics
- understand xLSTM conceptionally and do a clean implementation for Finance/Economics use-case (best case it gives better results than SOTA models, if not then also a result)
- think of potential other trading strategies (esp. classification problem -> is this the best framing for the DL model? )
- Train a RF for means of comparison

## Think about

### shared weights across companies
currently, every couple of sequence and target is treated as independent. the weights of my lstm do not differ for each company. they are trained on all 500 companies together. the input is just the 240-day sequence (actually that is also quite long for LSTMs - i should look into gradient clipping or investigate vanishing or exploding gradients even in LSTMs).
- it seems not obvious that using the same models for each company is a good idea. i am lacking industry-specific or company-specific effects. higher volatility / growth rates of tech stocks is not modelled.

### sliding vs expanding window
- what is actually the reason to not use an expanding window?
- might we get a better performance for recent years, where krauss and fischer noted a meltdown of lstm arbitrage

### training data
- why use only the training data of S&P500 companies, when there are many more available? I guess it kind of makes sense because in other markets there might not be same growth dynamics/ different return sequences and then distribution of training and test is not equal. on the other side, this could be interpreted as adding noise to the model which might help in generalizing better to unseen data (though this seems slightly questionable)


## 1st paper outline
- increase layers of LSTM (in general complexity of the model)
- include additional variables (interest rates (plural), oil, gold, maybe vix)
- add heterogeneity 

## 2nd paper
- be first to apply xlstm to economic/financial data 

