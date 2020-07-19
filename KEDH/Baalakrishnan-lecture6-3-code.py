
import gensim
import pprint
from nltk.corpus import stopwords

def preprocess(sentence):
    stop_words = set(stopwords.words("english"))
    return [w for w in sentence.lower().split() if w not in stop_words]

def find_most_similar(word_embed, word):
    return word_embed.most_similar(topn=5, positive=[word])


def cosine_sim_check(word_embed, word_pairs):
    return word_embed.similarity(word_pairs[0], word_pairs[1])


def wm_distance(sentence1, sentence2,model):
    return model.wmdistance(preprocess(sentence1), preprocess(sentence2))


input_wordlist = ["article", "act", "action", "crime", "felony", "punishment", "security", "fraud", "privacy",
                  "intellectual",
                  "terrorism", "immigrant", "illegal", "drugs", "appeal", "abuse", "alcohol", "complaint", "indictment",
                  "motion"]



from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_input_file=r"F:\Semester3\KEDH\glove.6B.200d.txt", word2vec_output_file="gensim_glove_vectors.txt")

from gensim.models.keyedvectors import KeyedVectors
glove_embed = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
law_embed= KeyedVectors.load_word2vec_format(r"F:\Semester3\KEDH\Law2Vec.200d.txt", binary=False)

print("Glove Embeddings, law Embeddings  loaded")


glove_sim_list = {}
for word in input_wordlist:
    glove_sim_list[word] = find_most_similar(glove_embed, word)
print("Top 5 most similar words obtained from  Glove model")
pprint.pprint(glove_sim_list)

Law_sim_list = {}
for word in input_wordlist:
    Law_sim_list[word] = find_most_similar(law_embed, word)
print("Top 5 most similar words obtained from  law2vec model")
pprint.pprint(Law_sim_list)


######################################################################################################

print("Cosine similarity between (“sentence”, “writing”) by GloVe and Law2vec")
print("By GloVe: %f" %cosine_sim_check(glove_embed,["sentence","writing"]))
print("By Law2vec: %f" %cosine_sim_check(law_embed,["sentence","writing"]))

print("Cosine similarity between (“sentence”, “reading”) by GloVe and Law2vec")
print("By GloVe: %f" %cosine_sim_check(glove_embed,["sentence","reading"]))
print("By Law2vec: %f" %cosine_sim_check(law_embed,["sentence","reading"]))

print("Cosine similarity between (“sentence”, “death”) by GloVe and Law2vec")
print("By GloVe: %f" % cosine_sim_check(glove_embed,["sentence","death"]))
print("By Law2vec: %f" % cosine_sim_check(law_embed,["sentence","death"]))


print("Cosine similarity between (“punishment”, “death”) by GloVe and Law2vec")
print("By GloVe: %f" % cosine_sim_check(glove_embed,["punishment","death"]))
print("By Law2vec: %f" % cosine_sim_check(law_embed,["punishment","death"]))


###############################################################################################################


print("Word Movers distance between I got capital punishment, I got sentenced to death")
print("By GloVe: %f" % wm_distance("I got capital punishment","I got sentenced to death",glove_embed))
print("By Law2Vec: %f" % wm_distance("I got capital punishment","I got sentenced to death",law_embed))

print("Word Movers distance between let me finish my sentence, I got sentenced to death")
print("By GloVe: %f" % wm_distance("let me finish my sentence","I got sentenced to death",glove_embed))
print("By Law2Vec: %f" % wm_distance("let me finish my sentence","I got sentenced to death",law_embed))

print("Word Movers distance between let me finish my sentence, I am still writing")
print("By GloVe: %f" % wm_distance("let me finish my sentence", "I am still writing",glove_embed))
print("By Law2Vec: %f" % wm_distance("let me finish my sentence", "I am still writing",law_embed))


print("Word Movers distance between let me finish my sentence, I am still reading")
print("By GloVe: %f" % wm_distance("let me finish my sentence", "I am still reading",glove_embed))
print("By Law2vec: %f" % wm_distance("let me finish my sentence", "I am still reading",law_embed))