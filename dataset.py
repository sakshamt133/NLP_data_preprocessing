import nltk
import numpy as np
from nltk.corpus import stopwords


class ReadDataset:
    def __init__(self, df):
        self.df = df
        self.stem = nltk.stem.PorterStemmer()
        self.stop_words = stopwords.words('english')
        self.wtoi = {
            '<UNK>': 1,
            '<PAD>': 2
        }
        self.itow = {
            1: '<UNK>',
            2: '<PAD>'
        }
        self.freq = {}
        self.data = []

    def create_dict(self, column, min_freq=1,  max_size=50000, include_stop=False):
        for row in np.array(self.df[column]):
            for word in nltk.word_tokenize(row):
                word = self.stem.stem(word)

                if include_stop is False and word in self.stop_words:
                    break

                if word in self.wtoi:
                    continue

                if word not in self.freq:
                    self.freq[word] = 1

                if len(self.wtoi) > max_size:
                    break

                if word in self.freq:
                    self.freq[word] += 1
                    frequency = self.freq[word]

                    if frequency >= min_freq:
                        if word not in self.wtoi:
                            self.wtoi[word] = len(self.wtoi) + 1
                            self.itow[len(self.itow)+1] = word
        print(f"len of dictionary is {len(self.wtoi)} ")

    def token_and_pad(self, column, batch_size=32):
        sentences = np.array(self.df[column])
        sort_sents = sorted(sentences, key=len, reverse=True)
        longest_len = len(nltk.word_tokenize(sort_sents[0]))
        print(f"len of longest sentence is {longest_len} {sort_sents[0]}")
        temp_data = []
        temp_ = 0

        for sent in sort_sents:
            temp = longest_len - len(nltk.word_tokenize(sent))
            data = []
            i = 0
            for word in sent:
                word = word.lower()
                word = self.stem.stem(word)

                if word in self.stop_words:
                    continue

                if word not in self.wtoi:
                    data.append(self.wtoi['<UNK>'])

                else:
                    data.append(self.wtoi[word])

            while i < temp:
                data.append(self.wtoi['<PAD>'])
                i += 1

            temp_data.append(data)
            temp_ += 1

            if temp_ == batch_size:
                temp_data = np.array(temp_data)
                self.data.append(temp_data)
                temp_ = 0
                temp_data = []

        self.data = np.array(self.data)
