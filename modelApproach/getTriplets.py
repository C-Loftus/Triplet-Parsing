from typing import Iterator
import typer
from pathlib import Path
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example

# this works so i can run this file from both
# this directory and the directory above,
# don't know how to do it more elegantly
try:
    from .scripts.rel_pipe import *
    from .scripts.rel_model import *
except:
    from scripts.rel_pipe import *
    from scripts.rel_model import *

def inferFromModel(
    trained_pipeline: Path, 
    test_data: Path, 
    stdout: Optional[bool] = typer.Option(False),
    json_output: Optional[bool] = typer.Option(False)
):

    nlp = spacy.load(trained_pipeline)
    allTriplets = []

    doc_bin = DocBin(store_user_data=True).from_disk(test_data)
    docs = doc_bin.get_docs(nlp.vocab)
    binary = True

    examples = []

    for gold in docs:
        
        pred = Doc(
            nlp.vocab,
            words=[t.text for t in gold],
            spaces=[t.whitespace_ for t in gold],
        )
        pred.ents = gold.ents
        for name, proc in nlp.pipeline:
            pred = proc(pred)
        examples.append(Example(pred, gold))


        if stdout == True:
            print(f"\n\nText: {gold.text}")

        spans = [(e.start, e.text, e.label_) for e in pred.ents]
        jsonTriplet = {
         "ent1": "",
         "label1": "",
         "ent2": "",
         "label2": "",
         "relation": ""
        }
        stdTriplet = [["", "",""], ["", "", ""]]

        for spanIndices, rel_dict in pred._.rel.items():
            gold_labels = [k for (k, v) in gold._.rel[spanIndices].items() if v == 1.0]
            if gold_labels:

                rel_dict = sorted(rel_dict.items(), key=lambda item: item[1])[0][0]                
                ENT_TEXT = 0
                ENT_LABELS = 1
                for start, text, label in spans:
                    if start == spanIndices[0]:
                        if json_output:
                            jsonTriplet["ent1"] = text
                            jsonTriplet["label1"]=label
                        else:
                            stdTriplet[ENT_TEXT][0] = text
                            stdTriplet[ENT_LABELS][0] = label
                        break

                for start, text, label in spans:
                    if start == spanIndices[1]:    
                        if json_output:
                            jsonTriplet["ent2"] = text
                            jsonTriplet["label2"]=label
                            jsonTriplet["relation"] = rel_dict
                            allTriplets.append(jsonTriplet) 
                        else:
                            stdTriplet[ENT_TEXT][2] = text
                            stdTriplet[ENT_LABELS][2] = label
                            stdTriplet[ENT_TEXT][1] = rel_dict
                            allTriplets.append(stdTriplet) 
                        break
        if stdout == True:
            print(stdTriplet)
        if binary == False:
            return allTriplets
        
    assert type(allTriplets) == list
    return allTriplets

if __name__ == "__main__":
    print(typer.run(inferFromModel))
