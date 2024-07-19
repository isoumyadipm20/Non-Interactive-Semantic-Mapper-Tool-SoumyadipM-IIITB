from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import pickle
import os
import ast

app = Flask(__name__)
CORS(app)

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
path_vec = os.path.join(current_dir, 'glove_wiki_vectors.kv')
path_embedding = os.path.join(current_dir, 'oov_word_embeddings.pkl')
path_sdg = os.path.join(current_dir, 'sdg_key.csv')

# Load models and data
wv = KeyedVectors.load(path_vec)
with open(path_embedding, 'rb') as f:
    oov_word_embeddings = pickle.load(f)
sdg = pd.read_csv(path_sdg, sep=";", dtype={'Id': str})

sdg['new_description'] = sdg['new_description'].apply(ast.literal_eval)
sdg['keywords'] = sdg['keywords'].apply(ast.literal_eval)

# Helper functions
def sent_vec_avg(sent):
    vector_size = wv.vector_size
    missing_words = []
    vectors = []
    for w in sent:
        ctr = 0
        wv_res = np.zeros(vector_size)
        for word in w:
            if word in wv:
                wv_res += wv[word]
                ctr += 1
            elif word in oov_word_embeddings:
                wv_res += oov_word_embeddings[word]
                ctr += 1
            else:
                missing_words.append(w)
        if ctr > 0:
            wv_res = wv_res / ctr
        vectors.append(wv_res)
    vec_arr = np.array(vectors)
    return vec_arr

sdg['vec'] = sdg['keywords'].apply(sent_vec_avg)

def tokenize(document):
    tokens = word_tokenize(document.lower())
    tokens = [token for token in tokens if token.isalpha()]
    return tokens

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

def get_top_p_percent_words(tokens, p):
    k = max(int(len(set(tokens)) * p), 1)
    word_freq = Counter(tokens)
    top_k_words = [word for word, _ in word_freq.most_common(k)]
    return top_k_words

def topic_modeling(tokens, n_topics):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([' '.join(tokens)])
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(X)
    return lda_model, vectorizer

def choose_top_topics(lda_model, vectorizer, n_topics, n_words=10):
    topics = lda_model.components_
    topic_keywords = []
    for topic_idx, topic in enumerate(topics):
        top_n_indices = topic.argsort()[-n_words:][::-1]
        topic_keywords.append([vectorizer.get_feature_names_out()[i] for i in top_n_indices])
    return topic_keywords

def extract_keywords_from_topics(topic_keywords):
    key_words = []
    for item in topic_keywords:
        key_words.extend(item)
    return key_words

def merge_sets(list1, list2):
    merged_set = list1 + list2 
    return list(set(merged_set))

def document_keywords_extraction(text, p_percent=0.2, n_topics=3):
    tokens = tokenize(text)
    tokens = lemmatize(tokens)
    tokens = remove_stopwords(tokens)
    top_k_words = get_top_p_percent_words(tokens, p_percent)
    lda_model, vectorizer = topic_modeling(tokens, n_topics)
    top_topics = choose_top_topics(lda_model, vectorizer, n_topics)
    keywords_from_topics = extract_keywords_from_topics(top_topics)
    merged_set = merge_sets(top_k_words, keywords_from_topics)
    return merged_set

def count_common_words(row, input_list):
    set_1 = set(row)
    set_2 = set(input_list)
    common_words = len(set_1.intersection(set_2))
    words_list = list(set_1.intersection(set_2))
    return common_words, words_list

def sent_vec(sent):
    vector_size = wv.vector_size
    vectors = []
    miss_w = []
    for w in sent:
        wv_res = np.zeros(vector_size)
        if w in wv:
            wv_res = wv[w]
            vectors.append(wv_res)
        elif w in oov_word_embeddings:
            wv_res = oov_word_embeddings[w]
            vectors.append(wv_res)
        else:
            miss_w.append(w)
    vec_arr = np.array(vectors)
    return vec_arr

def find_similarity_semantic(sdg_df, doc_df):
    ind_list = sdg_df['vec'].tolist()
    id_list = sdg_df['Id'].tolist()
    similarity_values = {}
    values = {}
    for ind in range(len(ind_list)):
        max_list = []
        for i in ind_list[ind]:
            similarity_matrix = cosine_similarity([i], doc_df)
            max_similarity = np.max(similarity_matrix)
            max_list.append(max_similarity)
        similarity_values[id_list[ind]] = np.mean(max_list)
        values[id_list[ind]] = max_list
    return similarity_values, values

def scoring(row):
    word_s = row['norm_word_count'] * 0.5
    sem_s = row['semantic_score'] * 0.5
    score = (word_s + sem_s) * 100
    rounded_score = round(score, 2)
    return rounded_score

def perform_semantic_mapping(doc_keywords):
    common_word_df = pd.DataFrame(sdg[['Id', 'Description']], columns=['Id', 'Description', 'comm_word_count', 'comm_words'])
    common_word_df[['comm_word_count', 'comm_words']] = sdg['new_description'].apply(lambda x: pd.Series(count_common_words(x, doc_keywords)))
    
    doc_vec = sent_vec(doc_keywords)
    similarity_score, average_values = find_similarity_semantic(sdg, doc_vec)
    
    sem_score = common_word_df.copy()
    sem_score['semantic_score'] = sem_score['Id'].map(similarity_score)
    sem_score['semantic_values'] = sem_score['Id'].map(average_values)
    
    sdg_scoring = sem_score.copy()
    scaler = MinMaxScaler()
    sdg_scoring['norm_word_count'] = scaler.fit_transform(sdg_scoring[['comm_word_count']])
    sdg_scoring['final_score_%'] = sdg_scoring.apply(scoring, axis=1)
    
    return sdg_scoring

def get_relevant_sentences(text, keywords):
    sentences = sent_tokenize(text)
    relevant_sentences = []
    for sentence in sentences:
        if any(keyword.lower() in sentence.lower() for keyword in keywords):
            relevant_sentences.append(sentence)
    return relevant_sentences

def get_top_10(df, id_col, desc_col, score_col, words_col, text):
    sorted_df = df.sort_values(by=score_col, ascending=False)
    top_10 = sorted_df.head(10)[[id_col, desc_col, score_col, words_col]]
    top_10 = top_10.rename(columns={
        id_col: 'Id',
        desc_col: 'Description',
        score_col: 'mapping_score',
        words_col: 'common_words'
    })
    top_10['relevant_sentences'] = top_10.apply(lambda row: get_relevant_sentences(text, row['common_words']), axis=1)
    return top_10.to_dict('records')

@app.route('/process_text', methods=['POST', 'OPTIONS'])
def process_text():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    data = request.json
    text = data['text']
    
    # Process the text using your semantic mapper
    doc_keywords = document_keywords_extraction(text)
    
    # Perform semantic mapping
    sdg_scoring = perform_semantic_mapping(doc_keywords)
    
    # Calculate the overall semantic mapping score
    overall_score = sdg_scoring['final_score_%'].mean()
    
    # Set a threshold of 30%
    threshold = 30.0
    
    # Check if the overall score meets the threshold
    if overall_score < threshold:
        results = {
            'error': 'The website is irrelevant and has no significant semantic similarity to the SDG goals.'
        }
    else:
        # Get top 10 goals, targets, and indicators
        goals = sdg_scoring[sdg_scoring['Id'].str.count('\.') == 0]
        targets = sdg_scoring[sdg_scoring['Id'].str.count('\.') == 1]
        indicators = sdg_scoring[sdg_scoring['Id'].str.count('\.') == 2]
        
        top_10_goals = get_top_10(goals, 'Id', 'Description', 'final_score_%', 'comm_words', text)
        top_10_targets = get_top_10(targets, 'Id', 'Description', 'final_score_%', 'comm_words', text)
        top_10_indicators = get_top_10(indicators, 'Id', 'Description', 'final_score_%', 'comm_words', text)
        
        results = {
            'original_text': text,
            'overall_score': overall_score,
            'goals': top_10_goals,
            'targets': top_10_targets,
            'indicators': top_10_indicators
        }
    
    response = jsonify(results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(debug=True)
