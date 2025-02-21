# 20.02.2024
## functions.py
- `image_to_edge_features()`, `reconstruct_image_from_edges()`, `save_chart()` for testing purposes
- ### `ohlc_to_edge_features()`
  - creates edges from the ohlc value to be used for input into the ml

## main.py
- for now only testing of edge features
- testing seemes fine so i can continue

## get_binance_data.py
- fetches different amount of ohlc data for different timeframes from binance withut an api key

## TODO:
- figure out what sould the bot predict exactly
- find best features (with the edges) to give to the ai
- send it edge features from multiple images(different timeframes)
- possibly give it my past trades
- train and backtest on data
- connect it to my binance account and make it run maybe on old notebook



# 21.02.2024
## functions.py
- ### `add_target_column(df, bars_forward)`
  - adds target feature:
      - 1 if in the next `bars_forward` bars the price goes up by 2%
      - -1 if in the next `bars_forward` bars the price goes down by 2%
      - 0 otherwise
   
- ### `test(df, tp, model)`
    - tests real buying or selling scenarios with predictions from the model.
    - can be changed for it to test also strategy itself without the model
    - TODO: add otpion to test either a model or a strategy
 
- ### `final_df(df)`
    - creates dataframe with edge features and the target feature
    - the function calls the `add_target_column` function with `bars_forward = 5`, this can be changed
    - TODO: maybe make the function intake `bars_forward` too
 
 - ### `test_random_forest(df)`
     - trains and tests random forest regressor on the data
     - it already has balanced weights
     - maybe play more with test size and n_estimators

## main.py
  - right now it gets a 17000 row df and trains the bot on first 2000 rows
  - then it transforms dataframe with `final_df()`
  - creates and tests the model
  - tests the strategy with the model.

## TODO:
  - play more with the strategy (try different tp and sl, or bars predicted)
  - play with the model a little more
  - try to use a different library and a better image recognicion model
  - right now when testing 10000 hourly bars it did 5 trades, with 20% winrate and a loss of 2.50
