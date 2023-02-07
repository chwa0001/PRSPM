"""Django's command-line utility for administrative tasks."""
import nltk
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet


#Text Processing
#lower casing the sentences

def dummy_fun(doc):
    return doc

def lower_casing(sentence):
    new_sentence = sentence.lower()
    return new_sentence

#Abbreviation expansion
def expand_abbriviation(sentence):
    replacement_patterns = [
        (r'won\'t', 'will not'),
        (r'can\'t', 'cannot'),
        (r'i\'m', 'i am'),
        (r'ain\'t', 'is not'),
        (r'(\w+)\'ll', '\g<1> will'),
        (r'(\w+)n\'t', '\g<1> not'),
        (r'(\w+)\'ve', '\g<1> have'),
        (r'(\w+)\'s', '\g<1> is'),
        (r'(\w+)\'re', '\g<1> are'),
        (r'(\w+)\'d', '\g<1> would')]
    patterns = [(re.compile(regex), repl) for (regex, repl) in replacement_patterns]

    new_sentence = sentence
    for (pattern, repl) in patterns:
        (new_sentence, count) = re.subn(pattern, repl, new_sentence)
    return new_sentence

def remove_numbers(sentence):
    # define the pattern to keep
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
    return re.sub(pattern, '', sentence)

#Punctuation removal
def punctuation_removal(sentence):
    # Remove the all the punctuations except '
    new_sentence = re.sub(';|\\\|:|,|!|\?|\"|<|>|\(|\)|\[|\]|\{|\}|@|#|\+|\=|\-|\_|~|\&|\*|\^|%|\||\$|/|`|\.|\'',
                          ' ', sentence,count=0, flags=0)
    return new_sentence

#Sentence tokenization
def tokenization(sentence):
    new_sentence = nltk.word_tokenize(sentence)
    return new_sentence

#Stopwords removal
def stopword_removal(sentence):
    stoplist = stopwords.words('english')
    new_sentence = [word for word in sentence if word not in stoplist]
    return new_sentence

#lemmatization
def get_wordnet_pos(word):
    pack = nltk.pos_tag([word])
    tag = pack[0][1]
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV

    else:
        return None


def lemmatization(sentence):
    lemmatizer = nltk.stem.WordNetLemmatizer()

    new_sentence = [lemmatizer.lemmatize(word, get_wordnet_pos(word) or wordnet.NOUN) for word in sentence]

    return new_sentence

def text_preprocessing(raw_sentence):
    sentence = lower_casing(raw_sentence)
    sentence = expand_abbriviation(sentence)
    sentence = remove_numbers(sentence)
    sentence = punctuation_removal(sentence)
    sentence = tokenization(sentence)
    sentence = stopword_removal(sentence)
    sentence = lemmatization(sentence)
    sentence = ' '.join(sentence)
    return sentence
