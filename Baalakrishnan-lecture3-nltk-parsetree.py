import nltk
from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
grammar1 = nltk.CFG.fromstring("""
S -> NP VP
VP -> V NP | V NP PP
PP -> P NP
V -> 'saw' | 'ate' | 'walked'
NP -> 'John' | 'Mary' | 'Bob' | Det N | Det N PP
Det -> 'a' | 'an' | 'the' | 'my'
N -> 'man' | 'dog' | 'cat' | 'telescope' | 'park'
P -> 'in' | 'on' | 'by' | 'with'
""")

grammar2 = nltk.CFG.fromstring("""
S -> NP VP
NP -> Det Nom | PropN
Nom -> Adj Nom | N
VP -> V Adj | V NP | V S | V NP PP
PP -> P NP
PropN -> 'Buster' | 'Chatterer' | 'Joe'
Det -> 'the' | 'a'
N -> 'bear' | 'squirrel' | 'tree' | 'fish'
Adj -> 'angry' | 'frightened' | 'little' | 'tall'
V -> 'chased' | 'said' | 'thought' | 'was' | 'put'
P -> 'on'
""")


def tree_display(grammar, sentence):
    from nltk.parse import RecursiveDescentParser
    rd = RecursiveDescentParser(grammar)
    sentence1 = sentence.split()
    for t in rd.parse(sentence1):
        treeform=t
        print(t)
        t.draw()
    #t1 = Tree.fromstring(str(treeform))
    #t1.draw()


tree_display(grammar1, "Bob saw John")
tree_display(grammar2, "Joe chased Buster")
tree_display(grammar1, "the dog ate the cat")
tree_display(grammar1, "Mary walked the dog by the park")
tree_display(grammar2, "the fish was frightened")
