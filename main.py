from functions import *

if __name__ == "__main__":
    data = pd.read_csv("1h_test.csv", index_col=0, parse_dates=True)
    train_data = data.head(2000)
    df = final_df(train_data)
    rf = test_random_forest(df)
    test(data.tail(10000), 0.02, rf)
    
    #df = add_target_column(data, 5)
    #print(len(df[df['target']==1]), len(df[df['target']==-1]))
    #print(test(df, 0.015))

    #data = data.tail(100)
    #save_chart(data)
    #edge_features2 = ohlc_to_edge_features(data)
    #rec_img2 = reconstruct_image_from_edges(edge_features2)