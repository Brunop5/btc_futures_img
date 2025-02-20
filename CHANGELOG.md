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
