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
import math
import operator
import numpy as np
import random
import pprint
import sim
import matplotlib.pyplot as plt
import itertools
import math
import time
from sklearn import cluster

#NUM_CLUSTERS = 8

def main():
    '''
    Graph.newOutputFile("output2")
    for i in range(myGraph.numRounds):
        seeds=myGraph.getSeeds("MaxDegree",0.5)
        myGraph.outputSeeds('output2',seeds,False)


    myGraph.checkOutputFile("output")
    myGraph.checkOutputFile("output2")'''

    #timer
    start=time.time()

    filename='8.35.3'
    myGraph=Graph('graphs/'+filename+'.json')

    #comparison
    temp = myGraph.numSeeds
    myGraph.numSeeds = int(temp)
    max_seeds = myGraph.getSeeds("MaxDegree",1)
    myGraph.numSeeds = temp

    #killer with advatange
    killer_seeds = myGraph.getSeeds("ClustDegreeKiller",1)

    #print "Simulation took: "+str(time.time()-start)+" seconds."

    print max_seeds
    print killer_seeds
    #myGraph.outputSeeds('output_clust',killer_seeds)

    killer_sets=list(itertools.combinations(killer_seeds,myGraph.numSeeds))

    #for killer in killer_sets:
    for i in range(myGraph.numRounds):
        killer=killer_sets[i]
        myGraph.outputSeeds('output2',killer,False)

    myGraph.checkOutputFile("output2")


    #print myGraph.simulateSeeds({"HighDeg":max_seeds,"Killer":killer_seeds},True)
    #print myGraph.competeSeeds([seeds,seeds2,seeds3,seeds4])[0]
    #killer_seeds=[4, 10, 107, 13, 110, 179, 174, 57, 59, 61] #33
    #killer_seeds=[1, 2, 6, 40, 201, 80, 58, 59, 93, 159] #32
    #Graph.simResults('graphs/'+filename+'.json','past_games/'+filename+'-EngineersAtNetwork.json',range(50),killer_seeds)


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
        self.num_clusters = 5#self.numSeeds
        self.numRounds=50

    def getSeeds(self,mode,arguments=[]):
        """Generate seeds based on mode"""
        if mode=="MaxDegree":
            #print "Generating seeds by maximum degree."
            return self.genSeedsMaxDegree(arguments,0)
        elif mode=="DegreeKiller":
            return self.genSeedsDegreeKiller(arguments)
        elif mode=="ClustDegreeKiller":
            return self.genSeedsClustDegreeKiller(arguments)
        elif mode=="BwDegree":
            return self.genSeedsMaxDegree(arguments,1)
        else:
            raise NameError("Invalid input to getSeeds()!")

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
        Returns best seed and score. Uses provided simulator."""

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
        #if sum(results.values())!=len(self.adj):
        #    print "Warning: Mismatching number of nodes in results."

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
    #if retmore then choose onthe basis of betweeness as well
    def genSeedsMaxDegree(self,p,bwness):
        """Generate seeds based on maximum degree.
        Optional input argument sets randomization. 0<p<1"""

        numSeeds = self.numSeeds

        if bwness:
            numSeeds = numSeeds*1.5

        if bwness:
            k_val = int(2000/math.sqrt(len(self.adj)))
            if k_val > len(self.adj):
                bw_node = nx.betweenness_centrality(self.nxgraph)
            else:
                bw_node = nx.betweenness_centrality(self.nxgraph, k = k_val )


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

        if bwness:
            numMax=int(self.numSeeds/(1.0*p))
            dict_bw = bw_node
            seeds_degree = seeds
            seeds = dict()
            for node in seeds_degree:
                value = dict_bw.get(node)
                key = node
                seeds[key] = value
            seeds_fin = dict(sorted(seeds.iteritems(), key=operator.itemgetter(1), reverse=True)[:numMax])
            seeds = seeds_fin.keys()


        #shuffle
        if p!=1:
            random.shuffle(seeds)

        return seeds[:self.numSeeds]

    def genSeedsDegreeKiller(self,advantage=1):
        """Generates seeds in order to beat maximum degeree.
           Advantage is a multiplier which gives enemy more seeds nodes.
           May not work well when dealing with >2 players."""

        old_num=self.numSeeds
        self.numSeeds=int(self.numSeeds*advantage)
        deg_seeds=self.genSeedsMaxDegree(1,0)
        self.numSeeds=old_num

        #get seeds options

        adj_options=[]
        #get three times as many high degree seeds
        self.numSeeds=len(deg_seeds)+3
        deg_options=self.genSeedsMaxDegree(1,0)
        self.numSeeds=old_num

        #get nodes near high degree seeds
        for seed in deg_seeds:
            adj_options=adj_options+self.adj[seed]
        seed_options=list(set(adj_options) & set(deg_options))

        killer_sets=list(itertools.combinations(seed_options,self.numSeeds))

        check_count=0
        best_score=0
        best_killer=None
        killer_found=False
        for killer in killer_sets:
            check_count+=1
            prefix=str(check_count)+"/"+str(len(killer_sets))

            #generate dict
            list_seeds=[deg_seeds,list(killer)]
            labels=map(str,range(len(list_seeds))) #since only care about being unique
            dict_seeds=dict(itertools.izip(labels,list_seeds))

            results=self.simulateSeeds(dict_seeds)
            if results['0']>results['1']:
                if check_count%10==0:
                    print prefix+": Max Degree Won with "+str(results['0'])
            else:
                print prefix+": Killer won with "+str(results['1'])
                killer_found=True

            #in case best seed can't be found
            if results['1']>best_score:
                best_score=results['1']
                best_killer=killer

            if check_count>(len(killer_sets)/2) and killer_found:
                return best_killer

        print "Best score: "+str(best_score)
        return best_killer


    def genSeedsClustDegreeKiller(self,advantage=1):
        """Generates seeds in order to beat maximum degeree.
           Advantage is a multiplier which gives enemy more seeds nodes.
           May not work well when dealing with >2 players."""

        old_num=self.numSeeds
        self.numSeeds=int(self.numSeeds*advantage)
        deg_seeds=self.genSeedsMaxDegree(1,0)
        self.numSeeds=old_num
        clust = cluster.SpectralClustering(n_clusters=self.num_clusters,
                                              eigen_solver='arpack',
                                              affinity="precomputed")

        adjacency_matrix = nx.adjacency_matrix(self.nxgraph)
        y = clust.fit_predict(adjacency_matrix)
        self.num_clusters = max(y) + 1
        max_degrees = [0]*self.num_clusters
        #max_degree_node = [0]*self.num_clusters
        avg_degrees = [0]*self.num_clusters
        sum_degrees = [0]*self.num_clusters
        counts = [0]*self.num_clusters

        seed_options = []

        for i in range(0, len(y)):
            deg = len(self.adj[i])            
            counts[y[i]] += 1
            sum_degrees[y[i]] += deg
            if max_degrees[y[i]] < deg:
                max_degrees[y[i]] = deg
                #max_degree_node[y[i]] = i
                
        for i in range(0, self.num_clusters):
            avg_degrees[i] = sum_degrees[i]/counts[i]

        second_largest_avg_deg = 0
        for i in range(0, self.num_clusters):
            #removing sparse graphs from consideration            
            #if avg_degrees[i] == np.median(avg_degrees):
                #counts[i] = 0
            #We only select the top two densest clusters
            if avg_degrees[i] != max(avg_degrees):
                if second_largest_avg_deg < avg_degrees[i]:
                    second_largest_avg_deg = avg_degrees[i]

        for i in range(0, self.num_clusters):
            #removing sparse graphs from consideration            
            if avg_degrees[i] < second_largest_avg_deg:
                counts[i] = 0

        #calculate the classes used by the degree based seeds
        deg_seeds_y = [0]*len(deg_seeds)
        for i in range(0, len(deg_seeds)):
            deg_seeds_y[i] = y[deg_seeds[i]]

        numMax = self.numSeeds + 3
        seeds=[None]*numMax
        deg=[0]*numMax
        
        #get seeds options
        for i in range(0, len(y)):
            #Select from large clusters only
            if counts[y[i]] == max(counts):
                #fill seeds
                curr_deg=len(self.adj[i])
                for j in range(numMax):
                    if curr_deg>deg[j]:
                        deg.insert(j,curr_deg)
                        seeds.insert(j,i)
                        break
    
                seeds=seeds[:numMax]
                deg=deg[:numMax]

        seed_options = seeds
        return seed_options
        
    ###PUBLIC METHODS

    @staticmethod
    def newOutputFile(filename):
        """Creates/clears new output file with specified filename.
        For use with outputSeeds when repeat=False"""

        f=open(filename,'w')
        f.close()
        return

    @staticmethod
    def simResults(graph_file,result_file,game_round,my_seeds=None):
        """Plots results using graph file and result file.
        If len(game_round)>1 no plotting will occur.
        Game round specifies rount out of 50 rounds"""

        teamname="EngineersAtNetwork"

        if len(game_round)!=1:
            print "simResult():Multi iteration, no plotting."

        if my_seeds:
            print "simResult():Using custom nodes!"

        #set up graph
        simGraph=Graph(graph_file)

        #import result_file
        f=open(result_file,'r')
        results=json.load(f)

        #dict for scoring
        score_dict={}

        #for simulation
        for r in game_round:
            seed_dict={}
            plot_seeds={}
            for key,value in results.iteritems():
                seed_dict[str(key)]=map(int,value[r])
                if my_seeds and key==teamname:
                    seed_dict[str(key)]=my_seeds
                try:
                    score_dict[str(key)]
                except KeyError:
                    score_dict[str(key)]=0

            #to display initial seeds
            for key,value in seed_dict.iteritems():
                for node in value:
                    try:
                        plot_seeds.pop(int(node))
                    except KeyError:
                        plot_seeds[int(node)]=str(key)

            if len(game_round)==1:
                print "Showing initial seeds"
                simGraph.drawGraph(plot_seeds,results.keys())
                print "Showing final results"
                print simGraph.simulateSeeds(seed_dict,True)
            else:
                output=simGraph.simulateSeeds(seed_dict,False)
                print "Round "+str(r)+" Node Totals:",
                print output
                if output[output.keys()[0]]>output[output.keys()[1]]:
                    score_dict[output.keys()[0]]+=1
                elif output[output.keys()[0]]<output[output.keys()[1]]:
                    score_dict[output.keys()[1]]+=1
                else:
                    score_dict[output.keys()[0]]+=1
                    score_dict[output.keys()[1]]+=1

        print "TOTAL WINS: ",
        print score_dict

if __name__ == '__main__':
    main()
