import re

import string
from spellchecker import SpellChecker
from contractions import contractions


def remove_space(text):
    text = text.strip().split()
    return " ".join(text)


def remove_punct(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_contractions(text, contractions):
    for word in contractions.keys():
        if "" + word + "" in text:
            text = text.replace("" + word + "", "" + contractions[word] + "")
    return text


def correct_spellings(text):
    spell = SpellChecker(language='en')
    corrected_text = []

    misspelled_words = spell.unknown(text.split())

    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))

        else:
            corrected_text.append(word)

    return " ".join(corrected_text)





#text = "You've been through how'll be there."
#print(remove_contractions(text.lower(), contractions))
#print(correct_spellings('There is somehing crazy about rher'))
#print(remove_punct("I love India! #IndiaMeriJaannnn ##Sunday"))


