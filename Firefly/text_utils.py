from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')


def clean_text(text):
    text = text.lower()
    for char in text:
        if (not char.isalpha()) and not (char == ' ' or char == '_' or char == '-'):
            text = text.replace(char, ' ')
    text = text.replace('_', '-')
    return text


def remove_stopwords(tokens):
    good_tokens = [token for token in tokens if token not in STOPWORDS]
    return good_tokens


def clean_and_tokenize(text):
    tokens = clean_text(text).split(' ')
    tokens = [token for token in tokens if token != '']
    good_tokens = remove_stopwords(tokens)
    return good_tokens


def phrase_contains(tokens1, tokens2):
    found_phrase = False
    for i in range(0, len(tokens1) - len(tokens2) + 1):
        is_equal = True
        for j in range(0, len(tokens2)):
            if tokens1[i+j] != tokens2[j]:
                is_equal = False
        if is_equal:
            found_phrase = True
    return found_phrase


def phrase_replace(tokens1, tokens2):
    for i in range(0, len(tokens1) - len(tokens2)):
        is_equal = True
        for j in range(0, len(tokens2)):
            if tokens1[i+j] != tokens2[j]:
                is_equal = False
        if is_equal:
            for j in range(0, len(tokens2)):
                tokens1[i+j] = '*'
    return tokens1