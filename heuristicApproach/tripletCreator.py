from html import entities
from typing import Tuple
import networkx as nx
import re, pandas as pd
import matplotlib.pyplot as plt


# takes in a sentence and returns the predicates with their associated values
def get_predicate(s):
    pred_ids = {}
    for w, index, spo in s:
        # add if it is a predicate, not just a 's' to end the word, and not escaped
        if spo == 'predicate' and w != "'s" and w != "\"": 
            pred_ids[index] = w
    predicates = {}
    for key, value in pred_ids.items():
        predicates[key] = value
    return predicates

# takes in a sentence and returns subjects based on the start and end positions
def get_subjects(s, start, end, adps):
    subjects = {}
    for w, index, spo in s:
        if index >= start and index <= end:
            if 'subject' in spo or 'entity' in spo or 'object' in spo:
                subjects[index] = w
    return subjects

# takes in a sentence and returns objects based on the start and end positions
def get_objects(s, start, end, adps):
    objects = {}
    for w, index, spo in s:
        if index >= start and index <= end:
            if 'object' in spo or 'entity' in spo or 'subject' in spo:
                objects[index] = w
    return objects

# returns a list of adverbs/prepositions based on a sentence with a start/end marker
def get_positions(s, start, end):
    adps = {}
    for w, index, spo in s:        
        if index >= start and index <= end:
            if 'of' == spo or 'at' == spo:
                adps[index] = w
    return adps


def create_triples(df_text: pd.DataFrame, corefs: list):
    sentences = []
    aSentence = []

    # iterates through the dataframe columns and appends sentences based on heuristics
    for index, row in df_text.iterrows():
        d_id, s_id, word_id, word, ent, ent_iob, lemma, cg_pos, pos, start, end, dep, relationalEntities = row.items()
        if 'subj' in dep[1]:
            aSentence.append([word[1], word_id[1], 'subject'])
        elif 'ROOT' in dep[1] or 'VERB' in cg_pos[1] or pos[1] == 'IN':
            aSentence.append([word[1], word_id[1], 'predicate'])

        elif 'obj' in dep[1]:
            aSentence.append([word[1], word_id[1], 'object'])

        elif ent[1] == 'ENTITY':
            aSentence.append([word[1], word_id[1], 'entity'])      

        # end of sentence detected so append the sentence and start a new one
        elif word[1] == '.':
            sentences.append(aSentence)
            aSentence = []
        else:
            aSentence.append([word[1], word_id[1], pos[1]])
        


    # now that we have each sentence from above, we can iterate through them
    # and get all the relationships from the sentence
    relations = []
    for s in sentences:
        if len(s) == 0: continue
        preds = get_predicate(s) # Get all verbs
        if preds:
            if (len(preds) == 1):
                predicate = list(preds.values())[0]
                # 'is' is the default short predicate since anything shorter than 2
                # isn't meaningful
                if (len(predicate) < 2):
                    predicate = 'is'
                ents = [e[0] for e in s if e[2] == 'entity']
                for i in range(1, len(ents)):
                    relations.append([ents[0], predicate, ents[i]])

            pred_ids = list(preds.keys())
            pred_ids.append(s[0][1])
            pred_ids.append(s[len(s)-1][1])
            pred_ids.sort()
                    
            # once we have the predicates we can extract the parts of speech from inside them
            # and use the subj obj pairs to construct a full triplet
            for i in range(1, len(pred_ids)-1):
                predicate = preds[pred_ids[i]]
                adps_subjs = get_positions(s, pred_ids[i-1], pred_ids[i])
                subjs = get_subjects(s, pred_ids[i-1], pred_ids[i], adps_subjs)
                adps_objs = get_positions(s, pred_ids[i], pred_ids[i+1])
                objs = get_objects(s, pred_ids[i], pred_ids[i+1], adps_objs)

                for k_s, subj in subjs.items():                
                    for k_o, obj in objs.items():
                        obj_prev_id = int(k_o) - 1
                        if obj_prev_id in adps_objs: # at, in, of
                            relations.append([subj, predicate + ' ' + adps_objs[obj_prev_id], obj])
                        else:
                            relations.append([subj, predicate, obj])
    
    ### Read coreferences from the passed in list, and remove any unnecessary characters 
    coreferences = []
    for val in corefs:
        if val[0].strip() != val[1].strip():
            if len(val[0]) <= 50 and len(val[1]) <= 50:
                co_word = val[0]
                real_word = val[1].strip('[,- \'\n]*')
                real_word = re.sub("'s$", '', real_word, flags=re.UNICODE)
                if (co_word != real_word):
                    coreferences.append([co_word, real_word])
            else:
                co_word = val[0]
                real_word = ' '.join((val[1].strip('[,- \'\n]*')).split()[:7])
                real_word = re.sub("'s$", '', real_word, flags=re.UNICODE)
                if (co_word != real_word):
                    coreferences.append([co_word, real_word])
                
    # Resolve corefs by looping through the relations and substituting appropriate entities
    triples_object_coref_resolved = []
    triples_all_coref_resolved = []
    for s, p, o in relations:
        coref_resolved = False
        for co in coreferences:
            if (s == co[0]):
                subj = co[1]
                triples_object_coref_resolved.append([subj, p, o])
                coref_resolved = True
                break
        if not coref_resolved:
            triples_object_coref_resolved.append([s, p, o])

    # do the same as above but with the subject / obj order reversed
    for s, p, o in triples_object_coref_resolved:
        coref_resolved = False
        for co in coreferences:
            if (o == co[0]):
                obj = co[1]
                triples_all_coref_resolved.append([s, p, obj])
                coref_resolved = True
                break
        if not coref_resolved:
            triples_all_coref_resolved.append([s, p, o])

    # we have all the triplets resolved so now we can get their associated
    # entity types then return
    entitiesTypes = getEntitiesFromTriplets(triples_all_coref_resolved, df_text)
    return(triples_all_coref_resolved, entitiesTypes)

# once we have the triplets we want to return the associated entity types
def getEntitiesFromTriplets(triplets, df: pd.DataFrame):
    allOutput = []

    for s, p, o in triplets:
        output = ["","",""]
        for sentence in df["Word"]:
            index = df[df["Word"] == sentence].index[0]
            if sentence in s:
                output[0] = df["RelevantEntities"][index]
            if sentence in o:
                output[2] = df["RelevantEntities"][index]
        allOutput.append(output)

    # [print(t, e) for t, e in zip(triplets, allOutput)]
    return allOutput
    


        