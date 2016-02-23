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
import sim
import matplotlib.pyplot as plt
import itertools

def main():
    '''
    Graph.newOutputFile("output2")
    for i in range(myGraph.numRounds):
        seeds=myGraph.getSeeds("MaxDegree",0.5)
        myGraph.outputSeeds('output2',seeds,False)


    myGraph.checkOutputFile("output")
    myGraph.checkOutputFile("output2")'''

    myGraph=Graph('graphs/2.10.33.json')
    max_seeds=myGraph.getSeeds("MaxDegree",1)
    killer_seeds=myGraph.getSeeds("DegreeKiller")
    print max_seeds
    print killer_seeds
    #myGraph.outputSeeds('output',seeds)
    print myGraph.simulateSeeds({"HighDeg":max_seeds,"Killer":killer_seeds},True)
    #print myGraph.competeSeeds([seeds,seeds2,seeds3,seeds4])
    #Graph.plotResults('graphs/2.10.12.json','past_games/2.10.12-EngineersAtNetwork.json',0)

class Graph():
    """Class to hold graph data"""

    def __init__(self,filename):
        """Constructor"""
        self.adj_pruned={}
        #load adjacency list
        f=open(filename,'r')
        self.adj_old=json.load(f)
        self.adj = dict()
        #convert to int
        for key,value in self.adj_old.iteritems():
            self.adj[int(key)]=map(int,value)
            #self.adj.pop(key)
        #create nxgraph
        self.nxgraph=nx.Graph(self.adj)
        #get info from filename
        self.numPlayers=int((filename.split('/')[-1]).split('.')[0])
        self.numSeeds=int((filename.split('/')[-1]).split('.')[1])
        self.numRounds=50

    def getSeeds(self,mode,arguments=[]):
        """Generate seeds based on mode"""
        if mode=="MaxDegree":
            #print "Generating seeds by maximum degree."
            return self.genSeedsMaxDegree(arguments)
        if mode=="DegreeKiller":
            return self.genSeedsDegreeKiller()
        else:
            raise NameError("Invalid input to getSeeds()!")
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

    def competeSeeds(self,list_seeds):
        """Competes each seeding against each other.
        input: list of list of seeds
        Returns best seed. Uses provided simulator."""

        scoring=[20,15,12,9,6,4,2,1,0]

        #warning!
        if len(list_seeds)<self.numPlayers:
            print "Number of seeds is smaller than number of players! Adding random."
            for i in range(self.numPlayers-len(list_seeds)):
                list_seeds.append(list(np.random.randint(len(self.adj),size=self.numSeeds)))

        labels=map(str,range(len(list_seeds))) #since only care about being unique
        scores=[0]*len(list_seeds)
        dict_seeds=dict(itertools.izip(labels,list_seeds))

        #competes seeds in tournament style base on size of graph
        matches=itertools.combinations(dict_seeds.iteritems(),self.numPlayers)

        for match in matches:
            results=sim.run(self.adj,dict(match))[1]
            #print results
            #sort by number of seeds
            sorted_results=sorted(results,key=results.get,reverse=True)
            #print results
            #increment scores
            for i in range(len(sorted_results)):
                scores[int(sorted_results[i])]+=scoring[i]

        #print list_seeds
        #print scores
        return list_seeds[np.argmax(scores)]

    def simulateSeeds(self,dict_seeds,plot=False):
        """Competes each seeding against each other.
        dict_seeds: key=name, value=list of seeds
        graph: adjacency dictionary
        Returns seed results. Uses provided simulator."""

        if len(dict_seeds)!=self.numPlayers:
            raise AssertionError("Invalid Input! Not enough player seeds.")

        mapping,results=sim.run(self.adj,dict_seeds)

        if plot:
            self.drawGraph(mapping,dict_seeds.keys())

        #check
        if sum(results.values())!=len(self.adj):
            print "Warning: Mismatching number of nodes in results."

        return results

    def drawGraph(self,seed_mappings,strat_names,show_node_numbers=False):
        """Draws graph based on seed mappings."""

        color_list=['r','b','g','c','m','y','k']
        color_names=['Red','Blue','Green','Cyan','Purple','Yellow','White']
        node_colors=['w']*len(self.adj)
        node_labels={}
        for i in range(len(node_colors)):
            #generate labels
            node_labels[i]=str(i)
        #get colors
        for key,value in seed_mappings.iteritems():
            try:
                node_colors[key]=color_list[strat_names.index(value)]
            except ValueError:
                node_colors[key]='w'

        print "Legend: "
        for i in range(len(strat_names)):
            print "\t"+strat_names[i]+" is "+color_names[i]

        pos=nx.spring_layout(self.nxgraph)
        if show_node_numbers:
            nx.draw(self.nxgraph,pos,node_color=node_colors,node_size=80,labels=node_labels,with_labels=True)
        else:
            nx.draw(self.nxgraph,pos,node_color=node_colors,node_size=80)

        plt.show()

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

    def genSeedsDegreeKiller(self):
        """Generates seeds in order to beat maximum degeree.
           May not work well when dealing with >2 players."""

        deg_seeds=self.genSeedsMaxDegree(1)

        #get seeds options

        adj_options=[]
        #get three times as many high degree seeds
        old_num=self.numSeeds
        self.numSeeds=self.numSeeds+3
        deg_options=self.genSeedsMaxDegree(1)
        self.numSeeds=old_num
        #get nodes near high degree seeds
        for seed in deg_seeds:
            adj_options=adj_options+self.adj[seed]
        seed_options=list(set(adj_options) & set(deg_options))

        killer_sets=list(itertools.combinations(seed_options,self.numSeeds))

        check_count=0
        for killer in killer_sets:
            check_count+=1
            prefix=str(check_count)+"/"+str(len(killer_sets))
            winner=self.competeSeeds([deg_seeds,list(killer)])
            if cmp(sorted(deg_seeds),sorted(winner))==0:
                print prefix+": Max Degree Won"
            else:
                print prefix+": Killer won"
                return list(winner)

        print "Couldn't find better seeds!"
        return deg_seeds

    ###PUBLIC METHODS

    @staticmethod
    def newOutputFile(filename):
        """Creates/clears new output file with specified filename.
        For use with outputSeeds when repeat=False"""

        f=open(filename,'w')
        f.close()
        return

    @staticmethod
    def plotResults(graph_file,result_file,game_round):
        """Plots results using graph file and result file.
        Game round specifies rount out of 50 rounds"""

        if game_round>=50:
            raise AssertionError("plotResults() Error: Invalid game_round input.")

        #set up graph
        plotGraph=Graph(graph_file)

        #import result_file
        f=open(result_file,'r')
        results=json.load(f)
        seed_dict={}
        plot_seeds={}
        #for simulation
        for key,value in results.iteritems():
            seed_dict[str(key)]=map(int,value[-1])

        #to display initial seeds
        for key,value in seed_dict.iteritems():
            for node in value:
                try:
                    plot_seeds.pop(int(node))
                except KeyError:
                    plot_seeds[int(node)]=str(key)

        print seed_dict

        print "Showing initial seeds"
        plotGraph.drawGraph(plot_seeds,results.keys())
        print "Showing final results"
        plotGraph.simulateSeeds(seed_dict,True)

if __name__ == '__main__':
    main()
