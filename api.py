import inspect, requests
from typing import Tuple
from nltk.corpus import stopwords
from nltk import download
import sys, pathlib, os, time, spacy, json
from pdfminer.high_level import extract_text
from modelApproach.getTriplets import inferFromModel

from   heuristicApproach.textCleaner import clean as textCleaner
from   heuristicApproach.docTagger import tagger as docTagger
from   heuristicApproach.tripletCreator import create_triples as tripletCreator
from   heuristicApproach.tripletFilter import filterTriplets as tripletFilter
import graph
import heuristicApproach.training as training


CONFIG_PATH = "config.json"

class Pipeline():
    # defaults to pipes as defined in https://arxiv.org/pdf/1909.01807.pdf
    defaultPipes = [
        textCleaner,
        docTagger,
        tripletCreator,
        tripletFilter
    ]

    def tryExceptDefault(self, approach, configArg, defaultFallBack):
        try:
            return type(defaultFallBack)(self.conf[approach][configArg]) if self.conf[approach][configArg] != None else defaultFallBack
        except:
            return defaultFallBack

    def _initModelConfig(self):
        self.trainedModelpath = self.tryExceptDefault("modelApproach", "trainedModelpath", ".\\modelApproach\\training\\model-best")
        self.inputData = self.tryExceptDefault("modelApproach", "inputData", ".\\modelApproach\\training\\model-best")
        return self

    def _initHeuristicConfig(self):
        self.enableFilter= self.tryExceptDefault("heuristicApproach", "enableFilter",False)
        self.dumpHeuristicCSV= self.tryExceptDefault("heuristicApproach","dumpHeuristicCSV",False)
        self.spacyModel= self.tryExceptDefault("heuristicApproach", "spacyModel","en_core_web_sm")

        nlp = spacy.load(self.spacyModel)
        allTypes = nlp.get_pipe("ner").labels
        # If it is already up to date nothing will be downloaded
        download('stopwords', quiet=True) 

        self.stopWords = sorted(list(set(self.conf["heuristicApproach"]["additionalStopWords"] + list(stopwords.words('english')))))

        return self 
        
    # pipes should be a list of FunctionType
    def __init__(self):

        with open(CONFIG_PATH, 'r') as j:
            try:
                self.conf = json.loads(j.read())
            except:
                print(f'configuration file not defined at {CONFIG_PATH}')
                exit()

        try: 
            self.approach = self.conf["approach"]
        except:
            print('ERROR: Approach not defined in config file. Please specify if you want a model or heuristic approach')
            exit()

        if self.approach == "modelApproach":
            self = self._initModelConfig()
        elif self.approach == "heuristicApproach":
            self = self._initHeuristicConfig()


    def __str__(self):
        return f'Pipeline Functions: {[pipe.__name__ for pipe in self.pipes]}'

    def _cleanPDF(self, pdf: str):
        return pdf.replace('\\n', '').replace('•', '').replace("\\x0c", "").replace("...", " ")

    # go through and remove entities that  aren't a valid type you want in the final schema
    def _filterResults(self, triplets, entities):
        filteredTuples = (filter(lambda x: (x[1] in self.validTypes), zip(triplets, entities)))
        trip, ent = Tuple(zip*(filteredTuples))
        return trip, ent
        
    def runFromPDF(self, path: str):
        pdfText = extract_text(path)
        pdfText = (repr(pdfText))
        text = self._cleanPDF(pdfText)
        return self.run(text)

# dumpCSV dumps the intermediate dataframe for debugging purposes
    def run(self, text: str):

        if self.approach == "heuristicApproach":
            df_out, corefs = docTagger(self.spacyModel, textCleaner(text))
            if self.dumpHeuristicCSV == True:
                df_out.to_csv('out-{}.csv'.format(str(time.strftime("%Y-%m-%d:%M"))))
            base_triplets, entitiesTypes = tripletCreator(df_out, corefs)

            if self.enableFilter:
                triplets, entities = tripletFilter(self.stopWords, base_triplets, entitiesTypes)
            else:
                triplets, entities = base_triplets, entitiesTypes
        elif self.approach == "modelApproach":
            inferFromModel(self.trainedModelpath, self.inputData)
            
        return triplets, entities


    def plotGraph(self, triplets: list):
        g = graph.tripletGraph(triplets)
        print(g)
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
        trainedModelPath = pathlib.Path(__file__).parent.resolve().joinpath("trainedModels")
        if not pathlib.Path.exists(trainedModelPath):
            pathlib.Path.mkdir(trainedModelPath)
        
        modelName = "{}-{}".format(("baseModel", str(self.spacyModel))[bool(self.spacyModel != None)], str(time.strftime("%Y-%m-%d:%M"))) 

        trainedModelPath = os.path.join(trainedModelPath, modelName)

        training.runTrainer(trainedModelPath, trainingData, baseModel=self.spacyModel)

if __name__ == "__main__":

    p = Pipeline()
    
    if len(sys.argv) > 1:
        input_str = (sys.argv[1])
    else: 
        # put some random article name here if you want to
        # show it by default without cli args
        input_str = "Paper"
    res = (p._testWikipedia(input_str))
    # res = p.run("test")
    print(res)