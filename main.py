from functions import *


if __name__ == "__main__":
    data = pd.read_csv("15m_test.csv", index_col=0, parse_dates=True)
    data = data.tail(50)
    save_chart(data)
    edge_features2 = ohlc_to_edge_features(data)
    rec_img2 = reconstruct_image_from_edges(edge_features2)