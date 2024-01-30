# Bibliotecas 

import pandas as pd
from os import path
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from tqdm import tqdm
import contractions

####### PRÉ-PROCESSAMENTO DOS DADOS #######

# Carga dos datasets
dataset_folder = "C:/Users/jlaba/OneDrive/Documents/MBA/TCC/reddit datasets"

# Verifica se o dataset já concatenado existe; caso contrário, carrega os dois arquivos separadamente
try: 
    df_all = pd.read_csv(path.join(dataset_folder, "adhd_comments_all.csv"))
except:
    df_adhd_women = pd.read_csv(path.join(dataset_folder, "adhdwomen-comment.csv"))
    df_adhd = pd.read_csv(path.join(dataset_folder, "ADHD-comment.csv"))

    df_all = pd.concat([df_adhd_women, df_adhd])

# Remove campos vazios
df_all = df_all.dropna()

# Remove linhas onde a coluna 'body' corresponde a 'deleted' ou 'removed'

columns_to_clean = ['body']

# Verifica se o conteúdo da célula corresponde a 'deleted' ou 'removed'
def check_exact_match(cell):
    return cell == '[deleted]' or cell == '[removed]'

# Aplica a função check_exact_match para deletar os dados
df_all = df_all[~df_all[columns_to_clean].applymap(check_exact_match).any(axis=1)]


####### PROCESSAMENTO DOS DADOS PARA A ENTRADA DO MODELO #######

# Converte as tags de POS geradas pela função pos_tag para que possa ser utilizada na lematização
def get_wordnet_pos(treebank_tag):
    
    # Mantém apenas:
    # Verbos (POS inicia em V)
    # Pronomes (POS inicia em N)
    # Adjetivos (POS inicia em J) 
    # Advérbios (POS inicia em R)

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# Pré-processamento do texto
def preprocess_text(text):
    
    # Converte para letra minúscula
    text = text.lower()
    
    # Expande contrações (e.g. you're should be you are)
    text = contractions.fix(text)
    
    # Remove URLs
    text = re.sub(r'\b(?:https?|ftp):\/\/[-A-Za-z0-9+&@#\/%?=~_|!:,.;]*[-A-Za-z0-9+&@#\/%=~_|]', ' ', text)
    
    # Substitui barras por espaços
    text = text.replace('/', ' ')

    # Remove caracteres especiais com exceção de apóstofres
    text = re.sub(r'[^a-zA-Z\s\';]', ' ', text)

    # Remove caracteres repetidos (e.g. "sooooo" vira "so", "whaaat" vira "what", etc)
    text = re.sub(r'(\w)(\1{2,})', r'\1', text)
    
    # Remove múltiplos espaços e os substitui por espaço simples
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenização
    tokens = word_tokenize(text)
    
    # Remove stopwords
    custom_words = ['something', 'im', 'dont', 'thing', 'going', 'getting', 'people', 
                    'someone', 'everyone', 'nothing', 'ive', 'id', 'thank', 'thats', 
                    'didnt', 'much', 'lot', 'isnt', 'stuff', 'right', 'sure', 'word', 
                    'way', 'anything', 'everything', 'others', 'cant', "i'm", "don't", 
                    "i've", "i'd", "that's", "didn't", "isn't", "can't", "s", "guy", "wa", 
                    "day", "week", "month", "adhd", "'s", "ty", "w", "ty", "haha", "omg", 
                    "lol", "lmao", "lmfao", "wtf", "thx", "yep", "yup", "nope", "etc", 
                    "reddit"]
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.union(custom_words)
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    # Aplica a tag de POS em todos os tokens
    tagged_tokens = pos_tag(filtered_tokens)
    
    # Lematização
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    
    for word, tag in tagged_tokens: 
        word_pos = get_wordnet_pos(tag)
        if word_pos is not None:
            lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=word_pos))
    
    final_tokens = lemmatized_tokens
    
    # Final text
    final_text = ' '.join(final_tokens)
    
    return final_text

# Adicionado para mostrar barra de progresso enquanto o código aplica a função
tqdm.pandas()

# Cria uma nova coluna para armazenar os dados pré-processados
df_all['body_processed'] = df_all['body'].progress_apply(lambda x: preprocess_text(x))

# Exporta os dados processados para CSV
df_all.to_csv(path.join(dataset_folder, "adhd_comments_processed_lda.csv"), index=False)