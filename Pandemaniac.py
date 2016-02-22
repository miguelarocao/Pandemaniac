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
import numpy as np
import random
import pprint

def main():
    myGraph=Graph('2.5.1.json')
    seeds=myGraph.getSeeds("MaxDegree",1)
    myGraph.outputSeeds('output',seeds)

    Graph.newOutputFile("output2")
    for i in range(myGraph.numRounds):
        seeds=myGraph.getSeeds("MaxDegree",0.5)
        myGraph.outputSeeds('output2',seeds,False)


    myGraph.checkOutputFile("output")
    myGraph.checkOutputFile("output2")

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
        #get info from filename
        self.numPlayers=int(filename.split('.')[0])
        self.numSeeds=int(filename.split('.')[1])
        self.numRounds=50

    def getSeeds(self,mode,arguments):
        """Generate seeds based on mode"""
        if mode=="MaxDegree":
            #print "Generating seeds by maximum degree."
            return self.genSeedsMaxDegree(arguments)
        else:
            raise("Invalid input to getSeeds()!")
        return seeds

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

    def outputSeeds(self,filename,seeds,repeat=True):
        """Prints seed to output file. Seeds is list to output.
        repeat=True will repeat the seeds numRounds times. Use if intending to only generate seeds once.
        repeat=False will append the seeds to an existing file. Use if intending to generate multiple sets of seeds.
        Note: If appending to existing file, make sure it's empty!"""

        out=[str(seed)+"\n" for seed in seeds]

        if repeat:
            #print "Outputting repeated seeds to "+filename
            with open (filename,'w') as f:
                for i in range(self.numRounds):
                    if i==(self.numRounds-1):
                        #no extra new line at the end
                        out[-1]=out[-1][:2]
                    f.writelines(out)
        else:
            #print "Outputting non-repeated seeds to "+filename
            with open(filename,'a') as f:
                f.writelines(out)

        f.close()

    def checkOutputFile(self,filename):
        """Checks that output file is of the right format."""
        count=0
        with open(filename,'r') as f:
            lines=f.readlines()
            for line in lines:
                try:
                    int(line)
                except ValueError:
                    print str(filename)+": Invalid output file! Non-integer ouput at line "+str(count)
                    return False
                count+=1

        if count!=(self.numRounds*self.numSeeds):
            print str(filename)+": Invalid output file! Incorrect number of seeds output."
            print "\t Expected: "+str(self.numRounds*self.numSeeds)+" Found: "+str(count)
            return False

        print str(filename)+": Valid output file"
        return True

    ###SEED GENERATION METHODS

    def genSeedsMaxDegree(self,p):
        """Generate seeds based on maximum degree.
        Optional input argument sets randomization. 0<p<1"""

        numMax=int(self.numSeeds/(1.0*p))

        seeds=[None]*numMax
        deg=[0]*numMax

        for key,value in self.adj.iteritems():
            #fill seeds
            curr_deg=len(value)
            for j in range(numMax):
                if curr_deg>deg[j]:
                    deg.insert(j,curr_deg)
                    seeds.insert(j,key)
                    break

            seeds=seeds[:numMax]
            deg=deg[:numMax]

        #shuffle
        if p!=1:
            random.shuffle(seeds)

        return seeds[:self.numSeeds]

    ###PUBLIC METHODS
    @staticmethod
    def competeSeeds(list_seeds):
        """Competes each seeding against each other.
        Returns winner as index.
        Uses provided simulator."""
        return

    @staticmethod
    def newOutputFile(filename):
        """Creates/clears new output file with specified filename.
        For use with outputSeeds when repeat=False"""

        f=open(filename,'w')
        f.close()
        return
if __name__ == '__main__':
    main()
