#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Miguel
#
# Created:     20/02/2016
# Copyright:   (c) Miguel 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import json
import networkx as nx

def main():
    myGraph=Graph('2.5.1.json')

    print myGraph.adj
    myGraph.pruneByDeg(2)
    print myGraph.adj
    myGraph.unPruneByDeg()
    print myGraph.adj

class Graph():
    """Class to hold graph data"""

    def __init__(self,filename):
        """Constructor"""
        self.adj_pruned={}
        #load adjacency list
        f=open(filename,'r')
        self.adj=json.load(f)
        #convert to int
        for key,value in self.adj.iteritems():
            self.adj[int(key)]=map(int,value)
            self.adj.pop(key)
        #create nxgraph
        self.nxgraph=nx.Graph(self.adj)

    def getAdj(self,node):
        """returns adjacency of node"""
        return self.adj[node]

    def getDeg(self,node):
        """returns degree of node"""
        return len(self.adj[node])

    def getEigen(self):
        """returns eigenvalues of graph"""
        pass

    def pruneByDeg(self,degree):
        """prunes graph based on <degree, stores pruned nodes"""
        temp_dict=dict(self.adj)
        for key,value in temp_dict.iteritems():
            if len(value)<degree:
                pop_val=self.adj.pop(key)
                self.adj_pruned[key]=pop_val
        self.nxgraph=nx.Graph(self.adj)
        return

    def unPruneByDeg(self):
        """undoes pruneByDeg"""
        for key,value in self.adj_pruned.iteritems():
            self.adj[key]=value

        self.adj_pruned=[]
        return

    def getSeeds(self):
        """return seed nodes of graph"""
        return

    def competeSeeds(list_seeds):
        """Competes each seeding against each other.
        Returns winner as index"""
        return

if __name__ == '__main__':
    main()
