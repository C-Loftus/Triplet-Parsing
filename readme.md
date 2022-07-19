This repository contains my code for triplet parsing.  To run this code an interface is provided through `api.py`.  All configuration is done through  the configuration file  `config.json`. 

There are two main ways of parsing triplets. First, there is a heuristic based rules approach. This is contained within the folder  `heuristicApproach`.  The code in this folder uses a for stage pipeline to parse the text, chunk relevant phrases, create triplets, and then filter out any poor quality ones. The pros for this approach is that it works without any data. As a downside,  some of the rules might not perform well on very specific texts. If you want custom entity types, this will need to be trained.

The other approach is contained within the folder `modelApproach`.  This approach uses a model which is trained for entity relationship extraction. The pros to this approach are that it doesn't rely on any heuristic rules or parts of speech. It runs entirely based on how the model was trained. The downside is that you will need data for both entity types and their relations. This may be harder to obtain and not as generic as the rules based approach.

To run this program specify your desired approach in config.json under the `"approach"` variable. All the other variables should have sensible defaults. However, they can be changed to your liking. Then run
```
python api.py
```