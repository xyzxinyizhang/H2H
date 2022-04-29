from itertools import product
from itertools import combinations
from pack import pack
import copy

class MapperBase:
    def __init__(self):
        self.MappingType = ['CNN','LSTM', 'FC']

    def getAccsAccuTimeSum(self, Gtime):
        sysTimeSum = 0
        nodes = list(Gtime.nodes)
        for node in nodes:
            sysTimeSum += (Gtime.nodes[node]['end'] - Gtime.nodes[node]['start'])
        return sysTimeSum

    def getJointDNNTrans(self, modalities, netBandwidth, dataType):
        DDR2MemBand = netBandwidth * (10 ** 9) * 8
        transLat = 0
        for modality in modalities:
            layers = list(modality.keys())
            for layer in layers:
                LayerObj = modality[layer]
                nodeType = LayerObj['LayerType']
                if nodeType == 'CNN':
                    transLat += (LayerObj['M'] * LayerObj['N'] * LayerObj['K'] * LayerObj['K']) * dataType / DDR2MemBand
                    transLat += (LayerObj['N'] * LayerObj['R'] * LayerObj['C'] * 4) * dataType / DDR2MemBand
                    transLat += (LayerObj['M'] * LayerObj['R'] * LayerObj['C']) * dataType / DDR2MemBand
                if nodeType == 'FC':
                    transLat += (LayerObj['M'] * LayerObj['N'] +LayerObj['N'] * LayerObj['K']) * dataType / DDR2MemBand
                    transLat += (LayerObj['M'] * LayerObj['K']) * dataType / DDR2MemBand
                if nodeType == 'LSTM':
                    transLat += (4*LayerObj['Hidden'] * LayerObj['Dimen'] + 4*LayerObj['Hidden'] * LayerObj['Hidden'] + LayerObj['Dimen']) * dataType / DDR2MemBand
                    transLat += LayerObj['Hidden'] * dataType / DDR2MemBand
        return transLat

    def getLayerCompLat(self, layerObject, accObject):
        accObject.getRunPara(layerObject)
        LatSec, Th = accObject.getLayerPerf(WEItrans=False, IFMtrans=False, OFMtrans=False)
        return LatSec

    def getJointDNNCompLat(self, accList, Ggraph):
        CompLat = 0
        lastNodes = []
        nodes = Ggraph.nodes
        for node in nodes:
            successors = list(Ggraph.successors(node))
            if not successors:
                lastNodes.append(node)

        for node in lastNodes:
            accName = Ggraph.nodes[node]['acc']
            accObj = self.getAccbyName(accName, accList)
            layerObject = Ggraph.nodes[node]['attri']
            getLayerCompLat = self.getLayerCompLat(layerObject, accObj)
            CompLat += getLayerCompLat
            nodeStartted = node
            while list(Ggraph.predecessors(nodeStartted)):
                nodeStartted = list(Ggraph.predecessors(nodeStartted))
                nodeStartted = nodeStartted[0]
                accName = Ggraph.nodes[nodeStartted]['acc']
                accObj = self.getAccbyName(accName, accList)
                layerObject = Ggraph.nodes[nodeStartted]['attri']
                gLayerCompLat = self.getLayerCompLat(layerObject, accObj)
                CompLat+=gLayerCompLat
        return CompLat

    def getAccbyName(self, accName, AccList):
        for Acc in AccList:
            if Acc.accName == accName:
                return Acc
    def getLayerBindVarify(self, timeGraph, accGraph):
        nodes = list(timeGraph.nodes)
        for node in nodes:
            if timeGraph.nodes[node]['bind'] != accGraph.nodes[node]['bind']:
                print('The layer to acc binding is wrong. The binding should be same in timeGraph and accGraph')
                exit()

    def getLayerTimeVarify(self, timeGraph, accGraph):
        nodes = list(timeGraph.nodes)
        for node in nodes:
            if timeGraph.nodes[node]['start'] != accGraph.nodes[node]['start'] or timeGraph.nodes[node]['end'] != accGraph.nodes[node]['end']:
                print('The layer time in last tune between Gacc and Gtime is wrong. The time should be same in timeGraph and accGraph')
                exit()


    def getLayerWeight(self, layerObj):
        if layerObj['LayerType'] == 'CNN':
            LayerWeight = layerObj['K'] * layerObj['K'] * layerObj['N'] * layerObj['M']
        elif layerObj['LayerType'] == 'LSTM':
            LayerWeight = layerObj['Hidden'] * 4 * (layerObj['Hidden'] + layerObj['Dimen'])
        elif layerObj['LayerType'] == 'FC':
            LayerWeight = layerObj['M'] * layerObj['N']
        else:
            print('This layer type is not supported yet, ask **** to add')
            print('function is getLayerWeight in class MapperBase')
            exit()
        return LayerWeight


    def getGraphSize(self, graph):
        graphNodes = graph.nodes
        return len(list(graphNodes))

    def getBindLayerinGraphNum(self, graph):
        graphNodes = graph.nodes
        BindLayer = 0
        for node in list(graphNodes):
            if graph.nodes[node]['bind'] == True:
                BindLayer+=1
        return BindLayer

    def getSmallBindLayerinGraph(self, graph):
        graphNodes = graph.nodes
        BindLayerEnd = float('inf')
        smallestNode = None
        for node in list(graphNodes):
            if graph.nodes[node]['end'] <= BindLayerEnd and graph.nodes[node]['bind']==True:
                BindLayerEnd =graph.nodes[node]['end']
                smallestNode = node
        return smallestNode

    def getSmallEndLayerinGraph(self, graph):
        graphNodes = graph.nodes
        BindLayerEnd = float('inf')
        smallestNode = None
        for node in list(graphNodes):
            if graph.nodes[node]['end'] < BindLayerEnd:
                BindLayerEnd =graph.nodes[node]['end']
                smallestNode = node
        return smallestNode

    def getNoPredesorNode(self, Gmap):
        nodes = list(Gmap.nodes)
        cnnlayers2Map = []
        lstmlayers2Map = []
        fclayers2Map = []
        for node in nodes:
            if not Gmap.pred[node]:
                info = Gmap.nodes[node]['attri']
                if info['LayerType'] == 'CNN':
                    cnnlayers2Map.append(node)
                elif info['LayerType'] == 'LSTM':
                    lstmlayers2Map.append(node)
                elif info['LayerType'] == 'FC':
                    fclayers2Map.append(node)
                else:
                    print('This layer type is not supported yet, ask **** to add')
                    print('This function is getNoPredesorNode, class MapperBase')
                    exit()
        layers2Map = {'CNN':cnnlayers2Map, 'LSTM':lstmlayers2Map, 'FC':fclayers2Map}
        return layers2Map

    def getAccInitTotLatency(self, accList):
        AccLatency={}
        for acc in accList:
            accName = acc.getAccName()
            AccLatency[accName]=0
        return AccLatency

    def getPredEndTime(self, NodeName, timeGraph):
        preds = timeGraph.predecessors(NodeName)
        predsEnd=[0]
        if preds:
            for pred in preds:
                predsEnd.append(timeGraph.nodes[str(pred)]['end'])
        return max(predsEnd)

    def getMinDistanceMappingS1(self, layers2Map, accs2Map, AccInitTotLatency, mapGraph, timeGraph, Pgraph=None):
        distance = float('inf')
        layerTimeMapped = None

        for accs in accs2Map:
            if len(layers2Map) != len(accs):
                print('The layer number and acc permutation size do not match!!!')
                exit()

            AccInitTotLatencyTmp = copy.deepcopy(AccInitTotLatency)
            layerTime = {}

            flagPass = False
            if Pgraph:
                for i in range(len(accs)):
                    layerName = layers2Map[i]
                    acc = accs[i]

                    if  Pgraph.has_node(layerName) and  Pgraph.nodes[layerName]['bind'] == True and acc.accName != Pgraph.nodes[layerName]['acc']:
                        flagPass = True
                        break
            if flagPass == True:

                continue

            for i in range(len(accs)):
                layerName = layers2Map[i]
                acc = accs[i]
                minStartTime = self.getPredEndTime(layerName, timeGraph)
                layerInfo = mapGraph.nodes[layerName]['attri']
                acc.getRunPara(layerInfo)

                layerTime[layerName] = [acc.getAccName(), max(AccInitTotLatencyTmp[acc.getAccName()], minStartTime)]
                LatSec, Th = acc.getLayerPerf()

                AccInitTotLatencyTmp[acc.getAccName()] = max(AccInitTotLatencyTmp[acc.getAccName()],
                                                             minStartTime) + LatSec

                layerTime[layerName].append(AccInitTotLatencyTmp[acc.getAccName()])



            if max(AccInitTotLatencyTmp.values()) < distance:
                distance = max(AccInitTotLatencyTmp.values())
                AccInitTotLatencyCandidate = copy.deepcopy(AccInitTotLatencyTmp)
                layerTimeMapped = layerTime

        if layerTimeMapped == None:
            print('bug hit', layers2Map, '\n')
            print('bug hit', accs2Map)

        return layerTimeMapped, AccInitTotLatencyCandidate

    def getMinDistanceMappingS2(self, layers2Map, accs2Map, AccInitTotLatency, mapGraph, timeGraph, Pgraph=None):

        distance = float('inf')
        layerTimeMapped = None

        for layers in layers2Map:
            for accs in accs2Map:
                if len(layers) != len(accs):
                    print('The layer number and acc permutation size do not match!!!')
                    exit()

                AccInitTotLatencyTmp = copy.deepcopy(AccInitTotLatency)
                layerTime = {}

                flagPass = False

                if Pgraph:
                    for i in range(len(layers)):
                        layerName = layers[i]
                        acc = accs[i]
                        if Pgraph.has_node(layerName) and Pgraph.nodes[layerName]['bind'] == True and acc.accName != Pgraph.nodes[layerName]['acc']:
                            flagPass = True
                            break
                if flagPass == True:
                    continue

                for i in range(len(layers)):
                    layerName = layers[i]
                    acc = accs[i]
                    minStartTime = self.getPredEndTime(layerName, timeGraph)
                    layerInfo = mapGraph.nodes[layerName]['attri']
                    acc.getRunPara(layerInfo)

                    layerTime[layerName] = [acc.getAccName(), max(AccInitTotLatencyTmp[acc.getAccName()], minStartTime)]
                    LatSec, Th = acc.getLayerPerf()

                    AccInitTotLatencyTmp[acc.getAccName()] = max(AccInitTotLatencyTmp[acc.getAccName()],
                                                                 minStartTime) + LatSec

                    layerTime[layerName].append(AccInitTotLatencyTmp[acc.getAccName()])


                if max(AccInitTotLatencyTmp.values()) < distance:
                    distance = max(AccInitTotLatencyTmp.values())
                    AccInitTotLatencyCandidate = copy.deepcopy(AccInitTotLatencyTmp)
                    layerTimeMapped = layerTime

        if layerTimeMapped == None:
            print('bug hit', layers2Map, '\n')
            print('bug hit', accs2Map)

        return layerTimeMapped, AccInitTotLatencyCandidate

    def getAccCategory(self, accList):
        cnnAccs2Map = []
        lstmAccs2Map = []
        fcAccs2Map = []
        for acc in accList:
            if acc.accType == 'LSTM':
                lstmAccs2Map.append(acc)
            elif acc.accType == 'CNN':
                cnnAccs2Map.append(acc)
            elif acc.accType == 'FC':
                fcAccs2Map.append(acc)
            else:
                print('This layer type is not supported yet, ask **** to add')
                exit('This function is getAccCategory, MapperBase')

        accs2Map = {'CNN': cnnAccs2Map, 'LSTM': lstmAccs2Map, 'FC': fcAccs2Map}
        return accs2Map

    def GraphChange(self,layerTimeMapped,accLayers,mapGraph,timeGraph):

        for layerName in list(layerTimeMapped.keys()):

            accName = layerTimeMapped[layerName][0]

            accLayers[accName].append(layerName)
            mapGraph.remove_node(layerName)
            timeGraph.nodes[layerName]['acc']=accName
            timeGraph.nodes[layerName]['start']=layerTimeMapped[layerName][1]
            timeGraph.nodes[layerName]['end']=layerTimeMapped[layerName][2]
            timeGraph.nodes[layerName]['isAssigned'] = True
        return accLayers,mapGraph,timeGraph

    def getGraphValidate(self, mapGraph, timeGraph):
        GmapLen = len(list(mapGraph.nodes))
        GtimeFlag=True
        for node in list(timeGraph.nodes):
            if timeGraph.nodes[node]['isAssigned'] == False:
                GtimeFlag=False
                break
        if GmapLen==0 and GtimeFlag:
            return True
        else:
            print('GmapLen is: ', GmapLen)
            print('GtimeFlag is: ', GtimeFlag)
            print('map graph check not pass or timegraph check not pass')
            exit()



    def getMap2AccObjandGtime(self, accLayers, accList, timeGraph, bindedLayer):
        if bindedLayer:
            print('This is modality switching Knapsack problem')
        for accName in list(accLayers.keys()):
            candidateAcc = self.getAccObjbyName(accName, accList)
            candidateAccIndex = accList.index(candidateAcc)
            accObj, timeGraph = pack(candidateAcc, timeGraph, accLayers[accName], bindedLayer)
            accList[candidateAccIndex] = accObj
        return accList, timeGraph

    def getAccTimeGraph(self, accList, graphin, graphout):
        graphout.clear()
        for acc in accList:
            accLayers = acc.assiLayers
            layerIndex = 0
            lastNode = ''
            for layerName in accLayers:
                node = graphin.nodes[layerName]
                graphout.add_node(layerName, attri=node['attri'], weight=node['weight'],
                                   isLayer=node['isLayer'], isAssigned=node['isAssigned'], acc=node['acc'],
                                   bind=node['bind'], bindVisit=False, start=node['start'], end=node['end'])
                if layerIndex == 0:
                    layerIndex = layerIndex + 1
                    lastNode = layerName
                else:
                    graphout.add_edge(lastNode, layerName)
                    layerIndex = layerIndex + 1
                    lastNode = layerName
        return graphout

    def getDependCheck(self, node1, node2):
        flag = False
        n1m = node1.split("L")
        n2m = node2.split("L")
        n1l = int(n1m[1])
        n2l = int(n2m[1])
        if n1l+1 == n2l:
            flag = True
        return flag

    def getModalCheck(self, node1, node2):
        flag = False
        n1m = node1.split("L")
        n2m = node2.split("L")
        if n1m[0] == n2m[0]:
            flag = True
        return flag

    def getAccMaxElapsed(self, graph):
        timeMax = 0
        for node in list(graph.nodes):
            if graph.nodes[node]['end'] > timeMax :
                timeMax = graph.nodes[node]['end']
        return timeMax

    def getAccObjbyName(self, accName, accList):
        for acc in accList:
            if acc.accName == accName:
                return acc

    def getGraphTimePrint(self, graph, layers):
        for layer in layers:
            print(layer, graph.nodes[layer]['start'],graph.nodes[layer]['end'])

    def getPerRep(self, accs, layerNum):
        permu=[]
        for roll in product(accs, repeat=layerNum):
            permu.append(list(roll))
        return permu

    def getComb(self, layers, accNum):
        comb = list(combinations(layers, accNum))
        return comb


if __name__ == "__main__":
    print('This class storages graph helper funciton.')








