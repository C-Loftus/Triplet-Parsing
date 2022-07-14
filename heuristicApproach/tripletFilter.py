import re, networkx as nx
import itertools


# graph is used for tracking entity relationships
def get_graph(triples):
    G = nx.DiGraph()
    for (s, p, o) in triples:
        G.add_edge(s, o, key=p)

    return G

def get_entities_with_capitals(G):
    entities = []
    for node in G.nodes():
        if (any(ch.isupper() for ch in list(node))):
            entities.append(node)
    return entities

# shortest path used for deducing new triplet relations
def get_paths_between_capitalised_entities(triples):
    
    g = get_graph(triples)
    ents_capitals = get_entities_with_capitals(g)
    paths = []
    for i in range(0, len(ents_capitals)):
        n1 = ents_capitals[i]
        for j in range(1, len(ents_capitals)):
            try:
                n2 = ents_capitals[j]
                path = nx.shortest_path(g, source=n1, target=n2)
                if path and len(path) > 2:
                    paths.append(path)
                path = nx.shortest_path(g, source=n2, target=n1)
                if path and len(path) > 2:
                    paths.append(path)
            except Exception:
                continue
    return g, paths

# we need to process it again since the graph adds more triplets
# by looking at node relationships
def get_final_entities(final_triples, mapper):
    finalEnts = []

    for (subj, pred, obj) in final_triples:
        tempEnt = []
        if subj in mapper:
            tempEnt.append(mapper[subj])
        else:
            tempEnt.append("")

        if obj in mapper:
            tempEnt.append(mapper[obj])
        else:
            tempEnt.append("")
        finalEnts.append(tempEnt)

    return finalEnts


# goes through graph and adds new triplets based on data from the graph.
# these triplets are not directly from the text but rather from the
# relational properties
def get_paths(doc_triples):
    triples = []

    g, paths = get_paths_between_capitalised_entities(doc_triples)
    for p in paths:
        path = [(u, g[u][v]['key'], v) for (u, v) in zip(p[0:], p[1:])]
        length = len(p)
        if (path[length-2][1] == 'in' or path[length-2][1] == 'at' or path[length-2][1] == 'on'):
            if [path[0][0], path[length-2][1], path[length-2][2]] not in triples:
                triples.append([path[0][0], path[length-2][1], path[length-2][2]])
        elif (' in' in path[length-2][1] or ' at' in path[length-2][1] or ' on' in path[length-2][1]):
            if [path[0][0], path[length-2][1], path[length-2][2]] not in triples:
                triples.append([path[0][0], 'in', path[length-2][2]])
    for t in doc_triples:
        if t not in triples:
            triples.append(t)
    return triples


# flattens data structure for parsing
def mergeEntities(entities):
    final = []
    for entlist in entities:
        for ent in entlist:
            if ent == "":
                final.append("")
            else: 
                try: 
                    final.append(ent[0][0])
                except:
                    final.append("")
    return final

# filters the triplets by excluding short / semantically not useful triplets
def filterTriplets(stopWords, prefilteredTriples, entities):
    linkedTriplets = list(itertools.chain(*prefilteredTriples))
    linkedEntities = mergeEntities(entities)

    assert len(prefilteredTriples) == len(entities), "Error: triplet and entity counts are not equal"
    assert (len(linkedEntities)) == (len(linkedEntities))

    mapper = {}
    for (t, e) in zip(linkedTriplets, linkedEntities):
        mapper[t] = e

    all_triples = get_paths(prefilteredTriples)
    filtered_triples = []    


    for count, (s, p, o) in enumerate(all_triples):
        if ([s, p, o] not in filtered_triples):
            if s.lower() in stopWords or o.lower() in stopWords:
                continue
            elif s == p:
                continue
            if s.isdigit() or o.isdigit():
                continue
            if '%' in o or '%' in s: #= 11.96
                continue
            if (len(s) < 2) or (len(o) < 2):
                continue
            if (s.islower() and len(s) < 4) or (o.islower() and len(o) < 4):
                continue
            if s == o:
                continue            
            subj = s.strip('[,- :\'\"\n]*')
            pred = p.strip('[- :\'\"\n]*.')
            obj = o.strip('[,- :\'\"\n]*')
            
            # concat all words together that don't have syntactic purpose on their own
            for sw in ['a', 'an', 'the', 'its', 'their', 'his', 'her', 'our', 'all', 'old', 'new', 'latest', 'who', 'that', 'this', 'these', 'those']:
                subj = ' '.join(word for word in subj.split() if not word == sw)
                obj = ' '.join(word for word in obj.split()  if not word == sw)
            subj = re.sub("\s\s+", " ", subj)
            obj = re.sub("\s\s+", " ", obj)
            
            if subj and pred and obj:
                filtered_triples.append([subj, pred, obj])


    filteredEnts = get_final_entities(filtered_triples, mapper)
    return filtered_triples, filteredEnts
