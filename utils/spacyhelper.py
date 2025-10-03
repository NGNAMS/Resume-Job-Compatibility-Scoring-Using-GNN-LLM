import spacy

nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    # Use SpaCy's nlp model to process the text
    #nlp.vocab["of"].is_stop = False
    #nlp.vocab["is"].is_stop = False
    doc = nlp(text)

    # Remove stop words and punctuation, and lemmatize the tokens
    processed_tokens = [
        token.text.lower().strip() for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_ != '-PRON-'
    ]

    # Join tokens back to a single string
    processed_text = ' '.join(processed_tokens)
    return processed_text