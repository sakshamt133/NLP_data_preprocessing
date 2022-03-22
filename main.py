import pandas as pd
from dataset import ReadDataset

df = pd.read_csv("D:\\Datasets\\NLP\\sentiment\\IMDB\\movie.csv")
d = ReadDataset(df)
d.create_dict('text')
d.token_and_pad('text')
print(d.data.shape)

