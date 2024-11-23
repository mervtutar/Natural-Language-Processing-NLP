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

############################################################
# Sentiment Analysis -> cümlenin pozitif ya da negatif olmasıyla ilgili değerlendirmeler

# modüler yapıyı kullanalım
from text_preprocessing import preprocess_reviews
processed_reviews = preprocess_reviews() # df["reviewText"] in son hali processed_reviews
processed_reviews.head()
import nltk
# nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
sia.polarity_scores("The film was awesome") # {'neg': 0.0, 'neu': 0.423, 'pos': 0.577, 'compound': 0.6249} compound (-1,1) bilgisiyle ilgileneceğiz, 0 üstü pozitif
sia.polarity_scores("I liked this music but it is not good as the other one") # {'neg': 0.207, 'neu': 0.666, 'pos': 0.127, 'compound': -0.298} negatif

processed_reviews[0:10].apply(lambda x: sia.polarity_scores(x)["compound"]) # 10 cümlenin polarity skorunda compounda bakalım
'''
0   0.00
1   0.00
2   0.40
3   0.65
4   0.86
5   0.00
6   0.87
7   0.82
8   0.00
9   0.92
Name: reviewText, dtype: float64
'''
# data frame e bir değişken olarak bu bilgiyi ekleyelim
df = pd.read_csv("amazon_reviews.csv", sep=",")
df["polarity_score"] = processed_reviews.apply(lambda x: sia.polarity_scores(x)["compound"]) # 10 cümlenin polarity skorunda compounda bakalım
# data frame e bir değişken olarak bu bilgiyi ekleyelim

# overall(ürüne verilen puan) ve polarity score birlikte değerlendirilerek bir inceleme yapılabilir

############################################################
# Feature Engineering

# polarity scoreları unsupervised learning şeklinde çıkardık bundan sonra supervised learninge gidebiliriz
# polarity score u belirli bir değerin üstünde olanları pozitif işaretleyelim, altında olanları negatif işaretleyelim 1,0 ikili sınıflandırma gibi
# böylece label oluşturalım ve sınıflandırma problemine dönüştürelim.

# compound değeri sıfırdan büyük olanlara  pozitif diğerlerine negatif atayalım
df["sentiment_label"] = processed_reviews.apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df["sentiment_label"] .value_counts()
'''
sentiment_label
pos    3944
neg     971
Name: count, dtype: int64
'''

# sentiment_label a göre grupla ve overall yani verilen paunalrın ortalamasını göster
df.groupby("sentiment_label")["overall"].mean()
'''
sentiment_label
neg   4.09
pos   4.71
Name: overall, dtype: float64
'''

# labelları binary olarak encode etmeliyiz modele sokmak için
df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"]) # pos ve neg leri 1,0 binary target feature haline getirdik

y = df["sentiment_label"] # bağımlı değişken, label
X = processed_reviews # bağımsız değişken


# Count Vectors - kelimeleri sayısal temsillere dönüştürmeliyiz, vektörleri oluşturmalıyız

# Count Vectors: frekans temsiller
# TF-IDF Vectors: normalize edilmiş frekans temsiller
# Word Embeddings (Word2Vec, GloVe, BERT vs)

# words
# kelimelerin nümerik temsilleri

# characters
# karakterlerin numerik temsilleri

# ngram
a = """Bu örneği anlaşılabilmesi için daha uzun bir metin üzerinden göstereceğim.
N-gram'lar birlikte kullanılan kelimelerin kombinasyolarını gösterir ve feature üretmek için kullanılır"""

TextBlob(a).ngrams(3) # 3lü ngram

from sklearn.feature_extraction.text import CountVectorizer

corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']

# word frekans
vectorizer = CountVectorizer()
X_c = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out() # corpustaki eşsiz kelimeleri getir
X_c.toarray() # eşsiz kelimeler corpusta var mı

# n-gram frekans
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2)) # 22 li featurelar üret
X_n = vectorizer2.fit_transform(corpus)
vectorizer2.get_feature_names_out() # ngram ile oluşan kelime öbekleri
X_n.toarray() # eşsiz kelimeler corpusta var mı

# count vector yöntemi kelimelerin, karakterlerin ya da ngramların frekanslarını sayar böylece elimizdeki metin ölçülebilir olur

vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(X)

vectorizer.get_feature_names_out()[10:15]
X_count.toarray()[10:15]

# Count Vectorizer
# 1 Eşsiz Tüm Terimleri Sütunlara, Bütün Dokümanları Satırlara Yerleştir
# 2 Terimlerin Dokümanlarda Geçme Frekanslarını Hücrelere Yerleştir


# TF-IDF
# 1 Count Vectorizer'ı Hesapla (Kelimelerin her bir dokümandaki frekansı)
# 2 TF-Term Frequency'yi Hesapla (t teriminin ilgili dokümandaki frekansı / dokümandaki toplam terim sayısı)
# 3 IDF - Inverse Document Frequency'i Hesapla 1+ loge ((toplam döküman sayısı+1) / (içinde t terimi olan döküman sayısı+1))
# 4 TF * IDF'i Hesapla
# 5 L2 Normalizasyonu Yap, Satırların kareleri toplamının karekökünü bul, ilgili satırdaki tüm hücreleri bulduğun değere böl

# CountVectorizer yöntemi yanlılıklar ortaya çıkarabilir, TF-IDF bnu önlemek içindir.
# TF-IDF kelime vektörü oluşturma yöntemidir, sözcüğün bulunduğu dökümanı ne kadar temsil ettiğini gösteren bir istatistiksel değerdir
# TF-IDF count vectorizer ile frekansları hesaplar sonra kelimelerin dökümanlar içindeki ağırlığını ve kelimelerin bütün corpustaki ağırlıklarını bulur. TF*IDF hesaplar.
# Daha sonra standartlaştırma normalizasyon işlemi yapılır.


from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_word_vectorizer = TfidfVectorizer() # kelime temelli
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X) # feature


tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3)) # ngram temelli
X_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(X) # feature


############################################################
# Sentiment Modeling

# 1. Text Preprocessing
# 2. Text Visualization
# 3. Sentiment Analysis
# 4. Feature Engineering
# 5. Sentiment Modeling


# Logistic Regression
log_model = LogisticRegression().fit(X_tf_idf_word, y)

cross_val_score(log_model,
                X_tf_idf_word,
                y,
                scoring="accuracy",
                cv=5).mean()
# accuracy 0.830111902339776

new_review = pd.Series("this product is great") # yeni yorumu tekrar tf-idf ten geçirmeliyiz
new_review = TfidfVectorizer().fit(X).transform(new_review)
log_model.predict(new_review) # 1 pozitif yorum

new_review = pd.Series("look at that shit very bad")
new_review = TfidfVectorizer().fit(X).transform(new_review)
log_model.predict(new_review) # 0 negatif yorum

new_review = pd.Series("it was good but I am sure that it fits me")
new_review = TfidfVectorizer().fit(X).transform(new_review)
log_model.predict(new_review) # 1 pozitif yorum

# orjinal veri setinden yorumları çekip onları soralım
random_review = pd.Series(df["reviewText"].sample(1).values) #  very fast and reliable, couldnt ask for a bett...
new_review = TfidfVectorizer().fit(X).transform(random_review)
log_model.predict(new_review) # 1 pozitif yorum


######################
# Random Forests
######################


# Count Vectors
rf_model = RandomForestClassifier().fit(X_count, y)
cross_val_score(rf_model, X_count, y, cv=5, n_jobs=-1).mean() # 0.8439471007121059

# TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)
cross_val_score(rf_model, X_tf_idf_word, y, cv=5, n_jobs=-1).mean() # 0.8284842319430317

# TF-IDF N-GRAM
rf_model = RandomForestClassifier().fit(X_tf_idf_ngram, y)
cross_val_score(rf_model, X_tf_idf_ngram, y, cv=5, n_jobs=-1).mean() # 0.7865717192268565

# 3 farklı yöntemle feature ürettik, Count Vectors daha iyi sonuç verdi

###############################
# Hiperparametre Optimizasyonu
###############################
# random forest hiperparametrelere sahip, bunları optimize edelim

rf_model = RandomForestClassifier(random_state=17)

# hiperparametreler
rf_params = {"max_depth": [8, None],
             "max_features": [7, "auto"],
             "min_samples_split": [2, 5, 8], # bir yaprakta kaç sample olacak
             "n_estimators": [100, 200]} # kaç ağaç fit edilecek

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=1).fit(X_count, y)

rf_best_grid.best_params_ # ön tanımlı değerler {'max_depth': None, 'max_features': 7, 'min_samples_split': 2, 'n_estimators': 100}


rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_count, y)

cross_val_score(rf_final, X_count, y, cv=5, n_jobs=-1).mean() # 0.8128179043743643
