## Explanation of this Program
This repository contains my code for triplet parsing.  To run this code an interface is provided through `api.py`.  All configuration is done through  the configuration file  `config.json`. 

There are two main ways of parsing triplets. First, there is a heuristic based rules approach. This is contained within the folder  `heuristicApproach`.  The code in this folder uses a for stage pipeline to parse the text, chunk relevant phrases, create triplets, and then filter out any poor quality ones. The pros for this approach is that it works without any data. As a downside,  some of the rules might not perform well on very specific texts. If you want custom entity types, this will need to be trained.

The other approach is contained within the folder `modelApproach`.  This approach uses a model which is trained for entity relationship extraction. The pros to this approach are that it doesn't rely on any heuristic rules or parts of speech. It runs entirely based on how the model was trained. The downside is that you will need data for both entity types and their relations. This may be harder to obtain and not as generic as the rules based approach.

## Installing
It is recommended to install dependencies using conda. This was how I built the software for use on my own computer. Once the conda cli is installed run
```
conda env create -f env.yml
```
You can also install the environment from a requirements.txt provided. 

## Running
To use my code in your own Python code, use the interface as defined in `api.py` within the `Pipeline` class. 

All features related to triplet parsing are contained with `config.json`. All features related to the api itself (where to parse triplets from, whether to print them out, etc.) are contained within the cli of `api.py`.

To run this program from the command line, specify your desired approach in config.json under the `"approach"` variable. All the other variables should have sensible defaults. However, they can be changed to your liking. `api.py` provides an a pipeline class for querying either approach. I used 'typer' as the way to build my cli. 
```
python api.py
```

## Output Format
Triplets and entities throughout this codebase are generally stored as parallel arrays, or list tuples. For instance
```
[[Russia, is, Country], [GPE, "", ""]]
```
An empty quote `""` signifies that spacy did not recognize any entity information from the text. 

## Unfinished Components
We have the modelApproach triplet model trained but we need an entity recognizer on top of it to make it more convenient. This model is only trained for relationship extraction. Without the entity recognizer the user will have to specify the types of entities he wants at runtime. So adding extra components to the pipeline will make the model more user friendly.

## Legal
All software that I built upon or referenced in this project was in the public domain, Creative Commons, or MIT Licensed. All of these licenses are appropriate for my project.