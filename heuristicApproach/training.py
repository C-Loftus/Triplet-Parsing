import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path

def runTrainer(outputDir: Path, trainingData, baseModel=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if baseModel is not None:
        nlp = spacy.load(baseModel)  # load existing spaCy model
    else:
        nlp = spacy.blank("en")  # create blank Language class

    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in trainingData:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if baseModel is None:
            nlp.begin_training()
        for _ in range(n_iter):
            random.shuffle(trainingData)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(trainingData, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)
        if outputDir is not None:
            if not outputDir.exists():
                outputDir.mkdir()
            nlp.to_disk(outputDir)
            print("Saved model to", outputDir)

            