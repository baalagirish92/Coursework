from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocessing(value):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(value)
    stop_words = set(stopwords.words("english"))
    filtered_text =sorted([word for word in word_tokens if word not in stop_words])
    final_text=[]
    for word in filtered_text:
        final_text.append(lemmatizer.lemmatize(word))
    return final_text


dict = {'vehicle':'A device used for transportation of goods and people with ease',
        'politics': 'the field related to government and political parties',
        'justice':'the administration of the law or authority in maintaining this',
        'food':'any nutritious substance',
        'patience':'the ability to tolerate problems, delay and suffering'}
for k, v in dict.items():
    print("My definition of %s" % str(k))
    print(str(v))
    our_defn = preprocessing(v)
    for i in range(len(wn.synsets(str(k)))):
        print("Definition of %s :" % str(wn.synsets(str(k))[i]))
        print(wn.synsets(str(k))[i].definition())
        wordNetDefn = preprocessing(wn.synsets(str(k))[i].definition())
        print("Number of Matching Words: %d" % len(set(our_defn) & set(wordNetDefn)))
        print()
