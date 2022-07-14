from typing import Tuple
import spacy
from spacy.attrs import intify_attrs
import pandas as pd

'''
this file goes through, and runs spacy to label the text.
Then with this information we can begin to chunk related
Phrases and word types together so that when we create the triplets
our noun phrases are better specified
'''

 # returns all the entities in the doc object
def get_tags_spacy(nlp, text):
    doc = nlp(text)    
    entities_spacy = [] # Entities that Spacy NER found
    for ent in doc.ents:
        entities_spacy.append([ent.text, ent.start_char, ent.end_char, ent.label_])
    return entities_spacy

def tag_all(nlp, text):
    doc = nlp(text)
    return doc

# Filter a sequence of spans so they don't contain overlaps
def filter_spans(spans):
    get_sort_key = lambda span: (span.end - span.start, span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    return result

#We can re tokenize the sentence  using entities that include multiple words
#  this will allow us to create triplets with more detailed entities instead of just single words
def tag_chunks(doc):
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        string_store = doc.vocab.strings
        for span in spans:
            start = span.start
            end = span.end
            retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': 'ENTITY'}, string_store))

 # does the same as tag_chunks but uses spans and not just entities
def tag_chunks_spans(doc, spans, ent_type):
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        string_store = doc.vocab.strings
        for span in spans:
            start = span.start
            end = span.end
            retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': ent_type}, string_store))


# We need to associate each entity with the corresponding entity type. (example: DATE, GPE, etc)
# This is done by relating the data in the df to the tokens in the text at the same location
def associate_entities(spacy_entities, df):
    newColumn = []

    for groupStart, groupEnd in zip(df["Start"], df["End"]):
        entities = []
        for entList in spacy_entities:
            entStart = entList[1]
            entEnd = entList[2]

            # everything is off by one for some reason. This +2 fixed it
            if (groupStart - 2 <= entStart) and (entEnd <= groupEnd + 2):
                entities.append((entList[3], (entStart, entEnd)))

        newColumn.append(entities)

    df["RelevantEntities"] = newColumn 
    # returns dataframe with new column that corresponds to the relevant entity type
    return df

def tagger(model, text) -> Tuple:
    df_out = pd.DataFrame(columns=['Document#', 'Sentence#', 'Word#', 'Word', 'EntityType', 'EntityIOB', 'Lemma', 'POS', 'POSTag', 'Start', 'End', 'Dependency'])
    corefs = []
    nlp = spacy.load(model)
    entities_spacy = get_tags_spacy(nlp, text)
    document = tag_all(nlp, text)

    tag_chunks(document)    
    
    # chunk - somethin OF something
    spans_change = []
    for i in range(2, len(document)):
        w_left = document[i-2]
        w_middle = document[i-1]
        w_right = document[i]
        if w_left.dep_ == 'attr':
            continue
        if w_left.ent_type_ == 'ENTITY' and w_right.ent_type_ == 'ENTITY' and (w_middle.text == 'of'): # or w_middle.text == 'for'): #  or w_middle.text == 'with'
            spans_change.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change, 'ENTITY')
    
    # chunk verbs with multiple words: 'were exhibited'
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'VERB'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk: verb + adp; verb + part 
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'ADP' or w_right.pos_ == 'PART'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk: adp + verb; part  + verb
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_right.pos_ == 'VERB' and (w_left.pos_ == 'ADP' or w_left.pos_ == 'PART'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')
    
    # chunk groups of verbs into verb phrases
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'VERB'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk all between LRB- -RRB- (something between brackets)
    start = 0
    end = 0
    spans_between_brackets = []
    for i in range(0, len(document)):
        if ('-LRB-' == document[i].tag_ or r"(" in document[i].text):
            start = document[i].i
            continue
        if ('-RRB-' == document[i].tag_ or r')' in document[i].text):
            end = document[i].i + 1
        if (end > start and not start == 0):
            span = document[start:end]
            try:
                assert (u"(" in span.text and u")" in span.text)
            except:
                pass
            spans_between_brackets.append(span)
            start = 0
            end = 0
    tag_chunks_spans(document, spans_between_brackets, 'ENTITY')
            
    # chunk entities
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.ent_type_ == 'ENTITY' and w_right.ent_type_ == 'ENTITY':
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'ENTITY')

    # after finishing the chunking, we now can put all the data we need into the 
    # pandas dataframe and then pass it off to the triplet creator
    doc_id = 1
    count_sentences = 0
    prev_dep = 'nsubj'
    for token in document:
        if (token.dep_ == 'ROOT'):
            if token.pos_ == 'VERB':
                df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_]
            else:
                df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, prev_dep]
        else:
            df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_]
                  
        if (token.text == '.'):
            count_sentences += 1
        prev_dep = token.dep_

    # this gets us the df AND the entities, since we need to pass the entities through.
    # we don't want to have to recompute them and we don't need to associate them in 
    # the dataframe
    associate_entities(entities_spacy, df_out)
    return df_out, corefs