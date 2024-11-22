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

##################################################
# Text Preprocessing
##################################################
df = pd.read_csv("amazon_reviews.csv", sep=",")
df.head()
df.columns

# Normalizing Case Folding
print(df["reviewText"]) # bu değişkene odaklanacağız


# ilk atılması gereken adımlardan biri tüm harfleri belirli bir standarda koymak (normalize etmek) çünkü bazıları büyük bazıları küçük
df["reviewText"] = df["reviewText"].str.lower() # hepsini küçük harf yapalım

# Noktalama İşaretleri ( Punctuations )
df["reviewText"] = df["reviewText"].str.replace('[^\w\s]', '', regex=True) # noktalama işaretlerinin yerine boşluk getir

# Numbers
df["reviewText"] = df["reviewText"].str.replace('\d', '', regex=True)

# Stopwords -> dilde anlam taşımayan kelimeler
import nltk
# nltk.download('stopwords')
sw = stopwords.words('english')

# metinlerde her satırı gezip stopwords varsa onları silmeliyiz ya da stopwords dışındakileri seçmeliyiz
# öncelikle cümleleri boşluklara göre split edip list comp yapısıyla kelimelerin hepsini gezip stopwords olmayanları seçelim, seçtiklerimizi tekrar join ile birleştirelim
df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


# Rarewords -> nadir geçen kelimeler
# nadir geçen kelimeleri çıkarmak için kelimelerin frekansını hesaplayıp kaç kere geçtiğini hesaplamalıyız
temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()

# frekansı 1 ya da 1 den küçük olanları drop edelim
drops = temp_df[temp_df <= 1]
df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))

# Tokenization -> metni parçalarına ayırmak, programatik şekilde
# nltk.download('punkt_tab')
df["reviewText"].apply(lambda x: TextBlob(x).words).head() # tüm satırlarda apply ile gez TextBlob(x) metodu çalıştırıldıktan sonra kelimeler getirilsin

# Lemmatization -> kelimeleri köklerine ayırmak, stemming metodu da aynı amaçla kullanılır
#nltk.download('wordnet')
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
# capabilities->capability oldu



# Terim Frekanslarının Hesaplanması
tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
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

text = " ".join(i for i in df.reviewText)

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
