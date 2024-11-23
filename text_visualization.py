##################################################
# 2. Text Visualization
##################################################

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

from text_preprocessing import preprocess_reviews
processed_reviews = preprocess_reviews() # df["reviewText"] in son hali processed_reviews
processed_reviews.head()

# Terim Frekanslarının Hesaplanması
tf = processed_reviews.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

# Barplot
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

tf[tf["tf"] > 500].plot.bar(x="words", y="tf") # frekansı 500 den büyük olanları görselleştir
plt.show()


# Wordcloud-Kelime Bulutu kelimelerin frekansına göre oluşturulur
# tüm satırları tek bir cümle gibi birleştirmeliyiz

text = " ".join(i for i in processed_reviews)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
# frekansı büyük olan kelimeler daha büyük görselleştirildi

# özelleştirelim
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")


# Şablonlara Göre Wordcloud - metinlerin kelime grubunu şablonla birleştirelim

tr_mask = np.array(Image.open("tr.png"))

wc = WordCloud(background_color="white",
               max_words=1000,
               mask=tr_mask,
               contour_width=3,
               contour_color="firebrick")

wc.generate(text)
plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

wc.to_file("wordcloud_şablon.png")
