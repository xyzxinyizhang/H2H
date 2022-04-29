import sys
import pickle
sys.path.append('\\Accelerator')
sys.path.append('\\DNN')

from cnn_acc_1 import cnn_acc_1
from ExampleDNN import DNN
import copy
from MapperH2H import MapperInit

class MapperSwitch(MapperInit):
    def __init__(self):
        super(MapperSwitch,self).__init__()
        self.PGtime = None
        self.PGacc = None
        self.Pmodalities = []
        self.PAccList = []
        self.PAccLayers = {}

    def getLastMapper(self,LastMapper):
        self.PGtime = copy.deepcopy(LastMapper.GtimeLastTune)
        self.PGacc = copy.deepcopy(LastMapper.GaccLastTune)
        self.Pmodalities = copy.deepcopy(LastMapper.modalities)
        self.PAccList = copy.deepcopy(LastMapper.AccListLastTune)
        self.PAccLayers = copy.deepcopy(LastMapper.AccLayersLastTune)


    def getBindedLayerCheck(self, layers2Map):

        flag = False
        layerTypes = list(layers2Map.keys())
        for layerType in layerTypes:
            layers = layers2Map[layerType]
            stopFlag = False
            for layer in layers:
                if self.PGacc.nodes[layer]['bind'] == True:
                    flag = True
                    stopFlag = True
                    break
            if stopFlag == True:
                break
        return flag

    def getBindedLayers(self, layers2Map, Pgraph):
        bindLayers = []
        unbindLayers = copy.deepcopy(layers2Map)
        layerTypes = list(layers2Map.keys())
        for layerType in layerTypes:
            layers = layers2Map[layerType]
            for layer in layers:
                if Pgraph.nodes[layer]['bind'] == True:
                    bindLayers.append(layer)
                    unbindLayers[layerType].remove(layer)
        return bindLayers,unbindLayers

    def getBindedLayersInGraph(self, graph, Pgraph):
        bindLayers = []
        allnodes = list(graph.nodes())
        for node in allnodes:
            if Pgraph.has_node(node) and Pgraph.nodes[node]['bind'] == True:
                bindLayers.append(node)
        return bindLayers

    def getBindLayerMapping(self, AccInitTotLatency,  mapGraph, timeGraph, bindLayers, Pgraph):
        AccInitTotLatencyTmp = copy.deepcopy(AccInitTotLatency)
        layerTime = {}

        for layer in bindLayers:
            layerName = layer
            accName = Pgraph.nodes[layerName]['acc']
            acc = self.getAccObjbyName(accName, self.AccList)
            minStartTime =self.getPredEndTime(layerName, timeGraph)
            layerInfo = mapGraph.nodes[layerName]['attri']
            acc.getRunPara(layerInfo)

            layerTime[layerName] = [acc.getAccName(), max(AccInitTotLatencyTmp[acc.getAccName()],minStartTime)]
            LatCycle, LatSec, Th = acc.getLayerPerf()
            AccInitTotLatencyTmp[acc.getAccName()] = max(AccInitTotLatencyTmp[acc.getAccName()],minStartTime) + LatCycle
            layerTime[layerName].append(AccInitTotLatencyTmp[acc.getAccName()])

        AccInitTotLatencyCandidate = copy.deepcopy(AccInitTotLatencyTmp)
        layerTimeMapped = layerTime

        if layerTimeMapped==None:
            print(bindLayers)
            print('bug hit')
        return  layerTimeMapped, AccInitTotLatencyCandidate

    def getMapping(self):
        self.getLayerBindVarify(self.PGtime, self.PGacc)
        AccInitTotLatency = self.getAccInitTotLatency(self.AccList)
        while self.getGraphSize(self.Gmap) > 0:
            layers2Map = self.getNoPredesorNode(self.Gmap)
            accs2Map = self.getAccCategory(self.AccList)
            for type in self.MappingType:
                currentTypeLayers = layers2Map[type]
                if currentTypeLayers:
                    if len(accs2Map[type]) > len(layers2Map[type]):
                        Accs2MapTmp = self.getPerRep(accs2Map[type], len(layers2Map[type]))
                        layers2MapTmp = layers2Map[type]
                        layerTimeMapped, AccInitTotLatency = self.getMinDistanceMappingS1(layers2MapTmp, Accs2MapTmp,
                                                                                        AccInitTotLatency,
                                                                                        self.Gmap, self.Gtime,
                                                                                       self.PGacc)
                        self.AccLayers, self.Gmap, self.Gtime = self.GraphChange(layerTimeMapped, self.AccLayers,
                                                                                 self.Gmap, self.Gtime)
                    elif len(accs2Map[type]) <= len(layers2Map[type]):
                        LayersComb = self.getComb(layers2Map[type],len(accs2Map[type]))

                        Accs2MapTmp = self.getPerRep(accs2Map[type], len(accs2Map[type]))
                        layers2MapTmp = LayersComb
                        layerTimeMapped, AccInitTotLatency = self.getMinDistanceMappingS2(layers2MapTmp, Accs2MapTmp,
                                                                                        AccInitTotLatency,
                                                                                        self.Gmap, self.Gtime,
                                                                                        self.PGacc)
                        self.AccLayers, self.Gmap, self.Gtime = self.GraphChange(layerTimeMapped, self.AccLayers,
                                                                                 self.Gmap, self.Gtime)


                else:
                    continue

        if not self.getGraphValidate(self.Gmap, self.Gtime):
            print('the graph Gmap is not cleaned or Gtime is not built correctly')
            exit()
        self.GtimeReserve = copy.deepcopy(self.Gtime)
        self.AccLayersReserve = copy.deepcopy(self.AccLayers)
        self.AccListReserve = copy.deepcopy(self.AccList)
        print(self.AccLayers)
        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gtime))
        power = self.getSysPower(self.Gtime)
        print('after step1, the system time is: ', self.PlatformElapsed)
        print('after step1, the system energy is: ', power)

        return


    def getBindedTime(self):
        allBindLayers = self.getBindedLayersInGraph(self.Gacc, self.PGacc)
        self.AccList, self.Gtime = self.getMap2AccObjandGtime(self.AccLayers, self.AccList, self.Gtime, allBindLayers)
        self.Gacc = self.getAccTimeGraph(self.AccList, self.Gtime, self.Gacc)
        self.GaccReserve = copy.deepcopy(self.Gacc)
        self.getLayerBindVarify(self.Gtime, self.Gacc)
        GaccTmp=copy.deepcopy(self.Gacc)
        while self.getBindLayerinGraphNum(GaccTmp) > 0:
            node = self.getSmallBindLayerinGraph(GaccTmp)
            layerInfo = self.Gacc.nodes[node]['attri']
            acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
            acc.getRunPara(layerInfo)
            LatSec, Th = acc.getLayerPerf(WEItrans=False)

            self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start']+LatSec
            self.Gtime.nodes[node]['end'] = self.Gtime.nodes[node]['start'] + LatSec

            HWsuccessor = self.Gacc.successors(node)
            SWsuccessor = self.Gtime.successors(node)
            for HWsu in list(HWsuccessor):
                self.getNodeShortenBinding(HWsu)
            for SWsu in list(SWsuccessor):
                self.getNodeShortenBinding(SWsu)


            GaccTmp.remove_node(node)



        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gacc))
        print('after step2, the system time is: ', self.PlatformElapsed)
        power = self.getSysPower(self.Gacc)
        print('after step2, the system energy is: ', power)

        return

if __name__ == "__main__":
    acc1 = cnn_acc_1()
    AccsList = [acc1]
    # the first round mapper
    SwitchMapper = MapperSwitch()

    InitMapper = open('H2HMapper.pkl', 'rb')
    InitMapper = pickle.load(InitMapper)

    SwitchMapper.getLastMapper(InitMapper)
    SwitchMapper.getModalityLayers(DNN)
    SwitchMapper.getAccList(AccsList)
    SwitchMapper.getMapping()
    SwitchMapper.getBindedTime()
    SwitchMapper.getIfmOfmTrans()
    SwitchMapper.getHomoNeighbor()
    with open("SwitchedMapper.pkl", "wb") as file:
        pickle.dump(SwitchMapper, file, True)



