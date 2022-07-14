import networkx as nx
import matplotlib.pyplot as plt
import pandas 

'''
 this file provides functionality for graphing the triplets produced
 from the run method in api.py
'''

class tripletGraph():

    def __init__(self, triplets):
        pairs = pandas.DataFrame(triplets, columns=['subject', 'relation', 'object'])
        self.knowledge_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object',
                create_using=nx.MultiDiGraph())
        self.node_deg = nx.degree(self.knowledge_graph)
        self.layout = nx.spring_layout(self.knowledge_graph, k=10, iterations=100)
        self.labels= dict(zip(list(zip(pairs.subject, pairs.object)), pairs['relation'].tolist()))

    def plot(self, x=200, y=200):
        plt.figure(num=None, figsize=(x, y), dpi=80)
        nx.draw_networkx(
            self.knowledge_graph,
            node_size=1000,
            arrowsize=8,
            linewidths=8,
            font_size=9,
            pos=self.layout,
            edge_color='red',
            edgecolors='blue',
            node_color='white',
            )
        nx.draw_networkx_edge_labels(self.knowledge_graph, pos=self.layout, edge_labels=self.labels, rotate=False,
                                 font_color='green')
        plt.axis('on')
        plt.show()

    def __str__(self):
        result = ""
        for line in self.labels:
            s = "[{}, {}, {}]\n".format(line[0], self.labels[line], line[1])
            result += (s)
        return result

    #  used for finding linked entities with the query
    def bfs(self, query):
        try:
            result = (nx.bfs_tree(self.knowledge_graph, source=query, depth_limit=1))
            relation = self.labels
            return [(line[0], relation[line], line[1]) for line in result.edges()]
        except nx.exception.NetworkXError or KeyError:
            print("Query {} not in graph, returning nothing...".format(query))

# for ent in doc.ents:
# ent.text ent.label_
if __name__ == "__main__":
    from api import Pipeline
    p = Pipeline()
    # res = (p._testWikipedia("Satoshi Nakamoto"))

    res = [['Satoshi Nakamoto', 'in', 'Japan'], ['Satoshi Nakamoto', 'in', 'United States'], ['Satoshi Nakamoto', 'in', 'Europe'], ['One person, Australian computer scientist Craig Steven', 'in', 'Japan'], ['One person, Australian computer scientist Craig Steven', 'in', 'United States'], ['One person, Australian computer scientist Craig Steven', 'in', 'Europe'], ["Nakamoto's name", 'in', 'Japan'], ["Nakamoto's name", 'in', 'United States'], ["Nakamoto's name", 'in', 'Europe'], ['Japanese', 'in', 'Japan'], ['Japanese', 'in', 'United States'], ['Japanese', 'in', 'Europe'], ['Japan', 'in', 'United States'], ['Japan', 'in', 'Europe'], ['Satoshi Nakamoto', 'is', 'name'], ['name', 'used by', 'presumed pseudonymous person'], ['name', 'used by', 'persons'], ['presumed pseudonymous person', 'developed', 'bitcoin'], ['persons', 'developed', 'bitcoin'], ['bitcoin', 'authored', 'bitcoin white paper'], ['part of implementation', 'devised', 'first blockchain database'], ['Satoshi Nakamoto', 'devised', 'first blockchain database'], ['development of bitcoin', 'until', 'December 2010'], ['widespread speculation', 'about', "Satoshi Nakamoto's true identity"], ["Satoshi Nakamoto's true identity", 'with', 'variety of people'], ['variety of people', 'posited as', 'person'], ['variety of people', 'posited as', 'persons'], ['person', 'behind', 'name'], ['persons', 'behind', 'name'], ['Satoshi Nakamoto', 'though', 'One person, Australian computer scientist Craig Steven'], ['proof', 'for', 'claim'], ["Nakamoto's name", 'is', 'Japanese'], ["Nakamoto's name", 'is', 'One person, Australian computer scientist Craig Steven'], ['One person, Australian computer scientist Craig Steven', 'was', 'man'], ['man', 'living in', 'Japan'], ['Japan', 'of', 'speculation'], ['speculation', 'has involved', 'software and/or cryptography experts'], ['software and/or cryptography experts', 'in', 'United States'], ['software and/or cryptography experts', 'in', 'Europe']]
    t = tripletGraph(res)
    print(t.bfs("Satoshi Nakamoto"))
