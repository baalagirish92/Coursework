import nltk
import numpy as np
import plotly.offline as plt
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocessing(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text.lower())
    vocab= set(word_tokens)
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in vocab if word not in stop_words and word.isalpha()]
    final_text = []
    for word in filtered_text:
        final_text.append(lemmatizer.lemmatize(word))
    return final_text



def get_coordinates(model, words):
    arr = []
    labels = []
    for wrd_score in words:
        try:
            wrd_vector = model[wrd_score]
            arr.append(wrd_vector)
            labels.append(wrd_score)
        except:
            pass
    # print(arr[:10])
    tsne = TSNE(n_components=3, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    z_coords = Y[:, 2]
    return x_coords, y_coords, z_coords


def word_embed_load(filename):
    word_embed={}
    with open(filename, 'r',encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            word_embed[word] = vector
    return word_embed


def plot_graph(x1,y1,z1,token):
    plot = go.Scatter3d(x=x1,
                         y=y1,
                         z=z1,
                         mode='markers+text',
                         text=token,
                         textposition='bottom center',
                         hoverinfo='text',
                         marker=dict(size=5, opacity=0.8))
    return plot



with open(r'F:\Semester3\KEDH\englisch_bgb_100.txt', 'r',encoding="utf8") as f:
    data = f.read()

words = preprocessing(data)
print(words[:10])

glove_embed = word_embed_load(r"F:\Semester3\KEDH\glove.6B.200d.txt")
print("Glove Embeddings done")
x, y, z = get_coordinates(glove_embed, words)
print("Glove Embeddings TSNE applied")
plot1=plot_graph(x,y,z,words)

law_embed =  word_embed_load(r"F:\Semester3\KEDH\Law2Vec.200d.txt")
print("law Embeddings done")
x, y, z = get_coordinates(law_embed, words)
print("law Embeddings TSNE applied")
plot2=plot_graph(x,y,z,words)


#data= [plot1, plot2]
layout = go.Layout(title='German Civil Code')
fig = go.Figure()
fig.add_trace(plot1)
fig.add_trace(plot2)

plt.plot(fig)