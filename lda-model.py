# Bibliotecas

import pandas as pd
from os import path
import gensim
import gensim.downloader
import nltk
nltk.download('stopwords')
import gensim.corpora as corpora
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.models.coherencemodel import CoherenceModel

import os
os.environ['GENSIM_DATA_DIR'] = "C:/Users/jlaba/OneDrive/Documents/MBA/TCC/models"

# Carrega os dados pré-processados
dataset_folder = "C:/Users/jlaba/OneDrive/Documents/MBA/TCC/reddit datasets"

try: 
    df_all = pd.read_csv(path.join(dataset_folder, "adhd_comments_processed_lda.csv"))
except:
    print("Error loading file")

# Cria lista para gerar o corpus
docs = df_all['body_processed'].values.tolist()
data_words = [str(d).split(" ") for d in docs]

# Cria dicionário
id2word = corpora.Dictionary(data_words)

texts = data_words
# Term Document Frequency (TF-IDF)
corpus = [id2word.doc2bow(text) for text in texts]

print(corpus[:1][0][:30])


# Gera modelos para diferentes números de tópicos

results = []

for t in range(2, 10):
    lda_model = gensim.models.LdaModel(corpus, id2word=id2word, num_topics=t)

    cm = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
    score = cm.get_coherence()
    tup = t, score
    results.append(tup)

results = pd.DataFrame(results, columns=['topic', 'score'])

# Visualização dos resultados

s = pd.Series(results.score.values, index=results.topic.values)
_ = s.plot(xlabel="Número de tópicos", ylabel="Score de Coerência", kind="bar")
