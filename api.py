from typing import Optional
import requests
from nltk.corpus import stopwords
from nltk import download
import pathlib, os, time, spacy, json, ast
from pdfminer.high_level import extract_text
from modelApproach.getTriplets import inferFromModel

from   heuristicApproach.textCleaner import clean as textCleaner
from   heuristicApproach.docTagger import tagger as docTagger
from   heuristicApproach.tripletCreator import create_triples as tripletCreator
from   heuristicApproach.tripletFilter import filterTriplets as tripletFilter
import graph
import heuristicApproach.training as training

import typer

CONFIG_PATH = "config.json"

class Pipeline():
    # defaults to pipes as defined in https://arxiv.org/pdf/1909.01807.pdf
    defaultPipes = [
        textCleaner,
        docTagger,
        tripletCreator,
        tripletFilter
    ]

    # try to parse the necessary items from config, if it isn't there, use the fallback
    def _tryExceptDefault(self, approach, configArg, defaultFallBack):
        try:
            if self.conf[approach][configArg] != None:
                # parse boolean strings into bool type. This is since bool("False") still returns 
                # True in python given the nature of boolean casting
                if type(defaultFallBack) == bool:
                    return ast.literal_eval(self.conf[approach][configArg])
                return type(defaultFallBack)(self.conf[approach][configArg]) 
            else:
                raise Exception
        except:
            print(f'\'{configArg}\' missing from {CONFIG_PATH}. Falling back to \'{defaultFallBack}\'')
            return defaultFallBack

    def _initModelConfig(self):
        self.trainedModelPath = self._tryExceptDefault("modelApproach", "trainedModelPath", ".\\modelApproach\\training\\model-best")
        self.inputDataPath = self._tryExceptDefault("modelApproach", "inputDataPath", ".\\modelApproach\\training\\model-best")
        return self

    def _initHeuristicConfig(self):
        self.dumpHeuristicCSV= self._tryExceptDefault("heuristicApproach","dumpHeuristicCSV", False)
        self.spacyModel= self._tryExceptDefault("heuristicApproach", "spacyModel","en_core_web_sm")

        nlp = spacy.load(self.spacyModel)
        allTypes = nlp.get_pipe("ner").labels
        # If it is already up to date nothing will be downloaded
        download('stopwords', quiet=True) 

        self.stopWords = sorted(list(set(self.conf["heuristicApproach"]["additionalStopWords"] + list(stopwords.words('english')))))

        return self 
        
    def __init__(self):

        with open(CONFIG_PATH, 'r') as j:
            try:
                self.conf = json.loads(j.read())
            except:
                raise Exception(f'configuration file not defined at {CONFIG_PATH}')

        try: 
            self.approach = self.conf["approach"]
        except:
            raise Exception('ERROR: Approach not defined in config file. Please specify if you want a model or heuristic approach')

        if self.approach == "modelApproach":
            self = self._initModelConfig()
        elif self.approach == "heuristicApproach":
            self = self._initHeuristicConfig()
            
        self.enableFilter= self._tryExceptDefault("filter", "enableFilter",False)
        self.onlyTheseEnts = self._tryExceptDefault("filter", "onlyTheseEnts", [])


    def __str__(self):
        return f'Pipeline Functions: {[pipe.__name__ for pipe in self.pipes]}'

    def _cleanPDF(self, pdf: str):
        return pdf.replace('\\n', '').replace('•', '').replace("\\x0c", "").replace("...", " ")

    # go through and remove entities that  aren't a valid type you want in the final schema
    def _filterEntities(self, triplets, entities) -> tuple[list, list]:
        if self.enableFilter != True:
            return triplets, entities

        filteredTrip = []
        filteredEnt = []
        for zipped in zip(triplets, entities):
            for ents in zipped[1]:
                if ents in self.onlyTheseEnts:
                    filteredTrip.append(zipped[0])
                    filteredEnt.append(zipped[1])
                    break
        return filteredTrip, filteredEnt
    
    def runFromPDF(self, path: str) -> tuple[list, list]:
        pdfText = extract_text(path)
        pdfText = (repr(pdfText))
        text = self._cleanPDF(pdfText)
        return self.run(text)

    def run(self, text: str) -> tuple[list, list]:
        if self.approach == "heuristicApproach":
            df_out, corefs = docTagger(self.spacyModel, textCleaner(text))

            # dumpCSV dumps the intermediate dataframe for debugging purposes
            if self.dumpHeuristicCSV == True:
                df_out.to_csv('heuristicDumpAtTime-{}.csv'.format(str(time.strftime("%H-%M-%S", time.localtime()))))
            
            base_triplets, entitiesTypes = tripletCreator(df_out, corefs)
            triplets, entities = tripletFilter(self.stopWords, base_triplets, entitiesTypes)

        elif self.approach == "modelApproach":
            parallelArrays = inferFromModel(self.trainedModelPath, self.inputDataPath, False, False)
            triplets = [array[0] for array in parallelArrays]
            entities = [array[1] for array in parallelArrays]

        if self.enableFilter:
            triplets, entities = self._filterEntities(triplets, entities)
        
        return triplets, entities

    def plotGraph(self, triplets: list) -> None:
        g = graph.tripletGraph(triplets)
        g.plot()
    # text representation
    def printGraph(self, triplets: list):
        print(graph.tripletGraph(triplets))

    def _testWikipedia(self, articleName: str):
        response = requests.get(
            'https://en.wikipedia.org/w/api.php',
            params={
                'action': 'query',
                'format': 'json',
                'titles': articleName,
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
            },
            # infosys network disrupts ssl so verify is off to allow api request
             verify=False
        ).json()
        page = next(iter(response['query']['pages'].values()))
        text = page["extract"]
        return self.run(text)

    def train(self, trainingData):
        if self.approach != "heuristicApproach":
            raise Exception("this training interface only  works for the heuristic approach. To train the model you should use the transformer training functionality built in to the model approach code.")

        trainedModelPath = pathlib.Path(__file__).parent.resolve().joinpath("trainedModels")
        if not pathlib.Path.exists(trainedModelPath):
            pathlib.Path.mkdir(trainedModelPath)
        
        modelName = "{}-{}".format(("baseModel", str(self.spacyModel))[bool(self.spacyModel != None)], str(time.strftime("%Y-%m-%d:%M"))) 

        trainedModelPath = os.path.join(trainedModelPath, modelName)

        training.runTrainer(trainedModelPath, trainingData, baseModel=self.spacyModel)

def cli_runner(
    stdout: Optional[bool] = typer.Option(False, help="Print triplets and entities to stdout."),
    test_article: Optional[str] = typer.Option(None, help="Test triplets on a Wikipedia article"),
    raw_text: Optional[str] = typer.Option(None, help="Create triplets from raw text from stdin"),
    config_path: str = typer.Option(CONFIG_PATH),
    pdf: Optional[str] = typer.Option(None , help="Takes in a path. Parses triplets from a PDF. Must use heuristics."),
    graph: Optional[bool] = typer.Option(False, help="Graph the triplets and entities that were parsed from the approach.")
):
    global CONFIG_PATH
    CONFIG_PATH = config_path
    p = Pipeline()
    triplets, entities = [], []
    if test_article != None: 
        triplets, entities = (p._testWikipedia(test_article))
    elif pdf != None:
        triplets, entities = (p.runFromPDF(pdf))
    else:
        triplets, entities = p.run(raw_text)

    
    if stdout == True:
        for t, e in zip(triplets, entities):
            print(t,e)

    if graph == True:
        p.plotGraph(triplets)

if __name__ == "__main__":
    typer.run(cli_runner)
