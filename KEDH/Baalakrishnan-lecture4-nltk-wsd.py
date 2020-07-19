#importing lesk from the nltk wsd
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

# the two sentence for which we need to disambiguate the word 'word' are given as inputs to sentence1 and sentence2
sentence1= "A rock is classified according to characteristics such as mineral and chemical composition"
sentence2 = "Queen are a British rock band formed in London in 1970"

# sentence split for input to lesk
sentence1_split = sentence1.split()
sentence2_split = sentence2.split()

# A simplified Lesk algorithm matches the context words in the given sentence to the one in the Synset Defintion
print("Sentence 1 : %s" %sentence1)
print("Using LESK for Disambiguation")
print(lesk(sentence1_split,'rock','n'))
print(lesk(sentence1_split,'rock','n').definition())
print()
print("Sentence 2 : %s" %sentence2)
print("Using LESK for Disambiguation")
print(lesk(sentence2_split,'rock','n'))
print(lesk(sentence2_split,'rock','n').definition())

