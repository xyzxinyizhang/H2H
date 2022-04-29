import sys
import pickle
import time
sys.path.append('\\Accelerator')
sys.path.append('\\DNN')

from cnn_acc_1 import cnn_acc_1
from ExampleDNN import DNN
import networkx as nx
import matplotlib.pyplot as plt
import copy
from MapperBase import MapperBase

class MapperInit(MapperBase):
    def __init__(self):
        super(MapperInit,self).__init__()
        self.Gmap = nx.DiGraph()
        self.Gtime = nx.DiGraph()
        self.GtimeLastTune = nx.DiGraph()
        self.GtimeLastTuneTmp = nx.DiGraph()
        self.Gacc = nx.DiGraph()
        self.GaccLastTune = nx.DiGraph()
        self.GaccLastTuneTmp = nx.DiGraph()
        self.modalities=[]
        self.AccList = []
        self.AccListLastTune = []
        self.AccListLastTuneTmp = []
        self.AccLayers = {}
        self.AccLayersLastTune = {}
        self.AccLayersLastTuneTmp = {}
        self.GtimeReserve = nx.DiGraph()
        self.GtimeReserveTmp = nx.DiGraph()
        self.GaccReserve = nx.DiGraph()
        self.GaccReserveTmp = nx.DiGraph()
        self.AccListReserve = []
        self.AccListReserveTmp = []
        self.AccLayersReserve = {}
        self.AccLayersReserveTmp = {}
        self.PlatformElapsed = float('inf')
        self.NetBand = 0
        self.Dtype = 0

    def getSysPower(self, timeGraph):
        allNodes = list(timeGraph.nodes)
        sysPower = 0
        for node in allNodes:
            accName = timeGraph.nodes[node]['acc']
            start = timeGraph.nodes[node]['start']
            end = timeGraph.nodes[node]['end']
            Acc = self.getAccbyName(accName, self.AccList)
            sysPower = sysPower + (end - start)* Acc.power
        return sysPower

    def getAccList(self, AccList):
        self.AccList = AccList
        self.getAccLayersInit()

    def getAccLayersInit(self):
        for acc in self.AccList:
            accName = acc.getAccName()
            self.AccLayers[accName]=[]

    def getModalityLayers(self, LayerList):
        self.modalities=LayerList
        self.getLayers2Graph()
        self.Gtime=copy.deepcopy(self.Gmap)

    def getLayers2Graph(self):
        self.Gmap = nx.DiGraph()
        for modality in self.modalities:
            layers = list(modality.keys())
            layerIndex = 0
            lastNode = ''
            for layer in layers:
                LayerObj = modality[layer]
                nodeName = LayerObj['LayerName']
                LayerWeight = self.getLayerWeight(LayerObj)
                self.Gmap.add_node(nodeName, attri=LayerObj, weight = LayerWeight, isLayer=True, isAssigned = False, acc = None, bind = False, start=0, end=0)
                if layerIndex == 0:
                    layerIndex = layerIndex + 1
                    lastNode = nodeName
                else:
                    self.Gmap.add_edge(lastNode, nodeName)
                    layerIndex = layerIndex + 1
                    lastNode = nodeName

    def getGraphVisual(self, Gname):
        if Gname == 'Gmap':
            nx.draw(self.Gmap, with_labels=True)
            plt.show()
        elif Gname == 'Gtime':
            nx.draw(self.Gtime, with_labels=True)
            plt.show()
        elif Gname == 'Gacc':
            nx.draw(self.Gacc, with_labels=True)
            plt.show()

    def getAccLayersAssign(self):
        touchedAcc = list(self.AccLayers.keys())
        for accName in touchedAcc:
            for acc in self.AccList:
                if acc.accName == accName:
                    acc.assiLayers = self.AccLayers[accName]

    def getAccLayersBind(self):
        for acc in self.AccList:
            if acc.assiLayers:
                accOccupy = 0
                for layer in acc.bindLayer:
                    layerInfo = self.getLayerInModality(layer)
                    acc.getRunPara(layerInfo)
                    if acc.accType == 'CNN':
                        layerWeiElement = acc.K*acc.K*acc.N*acc.M
                    elif acc.accType == 'LSTM':
                        layerWeiElement = acc.Hidden*4*(acc.Hidden+acc.Dimen)
                    elif acc.accType == 'FC':
                        layerWeiElement = acc.M*acc.N
                    accOccupy = accOccupy + layerWeiElement

                layerLeft = copy.deepcopy(acc.assiLayers)
                for layer in acc.bindLayer:
                    layerLeft.remove(layer)
                for layer in layerLeft:
                    layerInfo = self.getLayerInModality(layer)
                    acc.getRunPara(layerInfo)
                    if acc.accType == 'CNN':
                        layerWeiElement = acc.K*acc.K*acc.N*acc.M
                    elif acc.accType == 'LSTM':
                        layerWeiElement = acc.Hidden*4*(acc.Hidden+acc.Dimen)
                    elif acc.accType == 'FC':
                        layerWeiElement = acc.M*acc.N
                    if accOccupy + layerWeiElement <= acc.extraWeightElement:
                        acc.bindLayer.append(layer)
                        accOccupy = accOccupy + layerWeiElement
                    elif accOccupy + layerWeiElement > acc.extraWeightElement:
                        break

    def getLayerInModality(self, layer):
        for modality in self.modalities:
            if layer in list(modality.keys()):
                layerInfo = modality[layer]
                return layerInfo


    def getMapping(self):
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
                                                                                          self.Gmap, self.Gtime)

                        self.AccLayers, self.Gmap, self.Gtime = self.GraphChange(layerTimeMapped, self.AccLayers,
                                                                                 self.Gmap, self.Gtime)
                    elif len(accs2Map[type]) <= len(layers2Map[type]):
                        LayersComb = self.getComb(layers2Map[type], len(accs2Map[type]))

                        Accs2MapTmp = self.getPerRep(accs2Map[type], len(accs2Map[type]))
                        layers2MapTmp = LayersComb
                        layerTimeMapped, AccInitTotLatency = self.getMinDistanceMappingS2(layers2MapTmp, Accs2MapTmp,
                                                                                          AccInitTotLatency,
                                                                                          self.Gmap, self.Gtime)

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

        print('after step1, the system time time is: ', self.PlatformElapsed)
        print('after step1, the system energy is: ', power)
        return

    def getNodeShortenBinding(self, node):
        HWpreds = list(self.Gacc.predecessors(node))
        SWpreds = list(self.Gtime.predecessors(node))

        preds = HWpreds + SWpreds
        maxend = 0
        for pred in preds:
            if self.Gacc.nodes[pred]['end'] > maxend:
                maxend = self.Gacc.nodes[pred]['end']

        if maxend < self.Gacc.nodes[node]['start']:
            latency = self.Gacc.nodes[node]['end'] - self.Gacc.nodes[node]['start']
            self.Gacc.nodes[node]['start'] = maxend
            self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + latency
            self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
            self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
            HWsuccessor = self.Gacc.successors(node)
            SWsuccessor = self.Gtime.successors(node)
            for HWsu in list(HWsuccessor):
                self.getNodeShortenBinding(HWsu)
            for SWsu in list(SWsuccessor):
                self.getNodeShortenBinding(SWsu)
        else:
            return

    def getNodeShortenBindingLastTune(self, node):
        HWpreds = list(self.GaccLastTuneTmp.predecessors(node))
        SWpreds = list(self.GtimeLastTuneTmp.predecessors(node))

        preds = HWpreds + SWpreds
        maxend = 0
        for pred in preds:
            if self.GaccLastTuneTmp.nodes[pred]['end'] > maxend:
                maxend = self.GaccLastTuneTmp.nodes[pred]['end']

        if maxend < self.GaccLastTuneTmp.nodes[node]['start']:
            latency = self.GaccLastTuneTmp.nodes[node]['end'] - self.GaccLastTuneTmp.nodes[node]['start']
            self.GaccLastTuneTmp.nodes[node]['start'] = maxend
            self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + latency
            self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
            self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
            HWsuccessor = self.GaccLastTuneTmp.successors(node)
            for HWsu in list(HWsuccessor):
                self.getNodeShortenBindingLastTune(HWsu)
        else:
            return

    def getHelperGaccTime(self):
        for acc in self.AccList:
            layers = list(acc.assiLayers)
            print('timeline of acc:', acc.accName)
            self.getGraphTimePrint(self.Gacc, layers)
        print('end Gacc graph time show************************************************************')

    def getKnapsack(self, BindLayers= None):
        self.AccList, self.Gtime = self.getMap2AccObjandGtime(self.AccLayers, self.AccList, self.Gtime, BindLayers)
        self.Gacc = self.getAccTimeGraph(self.AccList, self.Gtime, self.Gacc)
        self.GaccReserve= copy.deepcopy(self.Gacc)

    def getBindedTime(self):
        self.getLayerBindVarify(self.Gtime, self.Gacc)
        GaccTmp=copy.deepcopy(self.Gacc)
        while self.getBindLayerinGraphNum(GaccTmp) > 0:
            node = self.getSmallBindLayerinGraph(GaccTmp)
            layerInfo = self.Gacc.nodes[node]['attri']
            acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
            acc.getRunPara(layerInfo)
            LatSec, Th = acc.getLayerPerf(WEItrans=False)
            self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start']+ LatSec
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

    def getBindedTimeLastTune(self):
        self.AccListLastTuneTmp, self.GtimeLastTuneTmp = self.getMap2AccObjandGtime(self.AccLayersLastTuneTmp, self.AccListReserve, self.GtimeLastTuneTmp, None)
        self.getLayerTimeVarify(self.GtimeLastTuneTmp, self.GaccLastTuneTmp)
        self.GaccLastTuneTmp = self.getAccTimeGraph(self.AccListLastTuneTmp, self.GtimeLastTuneTmp, self.GaccLastTuneTmp)
        self.getLayerBindVarify(self.GtimeLastTuneTmp, self.GaccLastTuneTmp)
        self.getLayerTimeVarify(self.GtimeLastTuneTmp, self.GaccLastTuneTmp)
        GaccTmp=copy.deepcopy(self.GaccLastTuneTmp)
        while self.getBindLayerinGraphNum(GaccTmp) > 0:
            node = self.getSmallBindLayerinGraph(GaccTmp)
            layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
            acc = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
            acc.getRunPara(layerInfo)
            LatSec, Th = acc.getLayerPerf(WEItrans=False)
            self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start']+ LatSec
            self.GtimeLastTuneTmp.nodes[node]['end'] = self.GtimeLastTuneTmp.nodes[node]['start'] + LatSec
            HWsuccessor = self.GaccLastTuneTmp.successors(node)
            SWsuccessor = self.GtimeLastTuneTmp.successors(node)
            for HWsu in list(HWsuccessor):
                self.getNodeShortenBindingLastTune(HWsu)
            for SWsu in list(SWsuccessor):
                self.getNodeShortenBindingLastTune(SWsu)

            GaccTmp.remove_node(node)

        return

    def getGraphUpdatViaModal(self, node):
        preds = list(self.Gacc.predecessors(node))
        successs = list(self.Gacc.successors(node))
        if not preds and not successs:
            return

        if not preds and successs:
            success = successs[0]
            if self.getModalCheck(node,success) and self.getDependCheck(node,success):
                accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                self.getGraphUpdatViaModal(success)

            elif self.getModalCheck(node,success) and not self.getDependCheck(node,success):
                self.getGraphUpdatViaModal(success)

            elif not self.getModalCheck(node, success):
                self.getGraphUpdatViaModal(success)

        if preds and not successs:
            pred=preds[0]
            if self.getModalCheck(pred, node) and self.getDependCheck(pred, node):
                accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                self.Gacc.nodes[node]['start'] = self.Gacc.nodes[pred]['end']
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                return

            if self.getModalCheck(pred, node) and not self.getDependCheck(pred, node):
                return

            elif not self.getModalCheck(pred, node):
                return

        if preds and successs:
            pred=preds[0]
            success = successs[0]
            if self.getModalCheck(node,success) and self.getModalCheck(pred,node) and \
                    self.getDependCheck(node,success) and self.getDependCheck(pred,node):
                accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=False)
                self.Gacc.nodes[node]['start'] = self.Gacc.nodes[pred]['end']
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                self.getGraphUpdatViaModal(success)

            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                    not self.getDependCheck(node, success) and self.getDependCheck(pred, node):
                accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                self.Gacc.nodes[node]['start'] = self.Gacc.nodes[pred]['end']
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                self.getGraphUpdatViaModal(success)

            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 self.getDependCheck(node, success) and not self.getDependCheck(pred, node):
                accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                self.getGraphUpdatViaModal(success)

            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 not self.getDependCheck(node, success) and not self.getDependCheck(pred, node):
                self.getGraphUpdatViaModal(success)

            elif self.getModalCheck(node, success) and not self.getModalCheck(pred, node) and \
                    self.getDependCheck(node, success):
                accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                self.getGraphUpdatViaModal(success)

            elif self.getModalCheck(node, success) and not self.getModalCheck(pred, node) and \
                    not self.getDependCheck(node, success):
                self.getGraphUpdatViaModal(success)

            elif not self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 self.getDependCheck(pred, node):
                accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                self.Gacc.nodes[node]['start'] = self.Gacc.nodes[pred]['end']
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                self.getGraphUpdatViaModal(success)

            elif not self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 not self.getDependCheck(pred, node):
                self.getGraphUpdatViaModal(success)

            elif not self.getModalCheck(node, success) and not self.getModalCheck(pred, node):
                self.getGraphUpdatViaModal(success)


    def getIfmOfmTrans(self):
        initNode = []
        for node in list(self.Gacc.nodes):
            if len(list(self.Gacc.predecessors(node)))==0:
                initNode.append(node)

        for node in initNode:
            self.getGraphUpdatViaModal(node)

        GaccTmp = copy.deepcopy(self.Gacc)
        for node in initNode:
            layerOrder = node.split("L")
            if layerOrder[1] == '1':
                GaccTmp.remove_node(node)
            else:
                continue

        while self.getGraphSize(GaccTmp) > 0:
            node = self.getSmallEndLayerinGraph(GaccTmp)
            HWpreds = list(self.Gacc.predecessors(node))
            SWpreds = list(self.Gtime.predecessors(node))
            preds = list(set(HWpreds+SWpreds))
            maxend = 0
            for pred in preds:
                if self.Gacc.nodes[pred]['end'] > maxend:
                    maxend=self.Gacc.nodes[pred]['end']

            if maxend < self.Gacc.nodes[node]['start']:
                latency = self.Gacc.nodes[node]['end']-self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['start'] = maxend
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + latency
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']

                HWsuccessor = self.Gacc.successors(node)
                SWsuccessor = self.Gtime.successors(node)
                for HWsu in list(HWsuccessor):
                    self.getNodeShortenBinding(HWsu)
                for SWsu in list(SWsuccessor):
                    self.getNodeShortenBinding(SWsu)

            GaccTmp.remove_node(node)

        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gacc))
        print('after step3, the system time is: ', self.PlatformElapsed)
        power = self.getSysPower(self.Gacc)
        print('after step3, the system energy is: ', power)


        return

    def getNodeExtend(self, node):
        HWpreds = list(self.GaccLastTuneTmp.predecessors(node))
        SWpreds = list(self.GtimeLastTuneTmp.predecessors(node))
        earlyStart = 0
        for HWpred in HWpreds:
            if self.GaccLastTuneTmp.nodes[HWpred]['end'] > earlyStart:
                earlyStart = self.GaccLastTuneTmp.nodes[HWpred]['end']
        for SWpred in SWpreds:
            if self.GtimeLastTuneTmp.nodes[SWpred]['end'] > earlyStart:
                earlyStart = self.GtimeLastTuneTmp.nodes[SWpred]['end']


        length = self.GaccLastTuneTmp.nodes[node]['end']-self.GaccLastTuneTmp.nodes[node]['start']
        self.GaccLastTuneTmp.nodes[node]['start'] = earlyStart
        self.GaccLastTuneTmp.nodes[node]['end'] = earlyStart + length

        self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
        self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']

        HWsuccessor = self.GaccLastTuneTmp.successors(node)
        for HWsu in list(HWsuccessor):
            self.getNodeExtend(HWsu)

        if not list(HWsuccessor) :
            return


    def getTunePerf(self,node,neighbor, relation):
        self.GtimeLastTuneTmp = copy.deepcopy(self.GtimeReserve)
        self.GaccLastTuneTmp = copy.deepcopy(self.GaccReserve)
        self.AccLayersLastTuneTmp = copy.deepcopy(self.AccLayersReserve)
        if relation == 'pred':
            nodeAcc = self.GtimeLastTuneTmp.nodes[node]['acc']
            predAcc = self.GtimeLastTuneTmp.nodes[neighbor]['acc']
            self.GtimeLastTuneTmp.nodes[node]['acc'] = predAcc
            self.GaccLastTuneTmp.nodes[node]['acc'] = predAcc
            self.AccLayersLastTuneTmp[nodeAcc].remove(node)
            predAccPos =  self.AccLayersLastTuneTmp[predAcc].index(neighbor)+1
            self.AccLayersLastTuneTmp[predAcc].insert(predAccPos, node)
            predSuccOnAcc = list(self.GaccLastTuneTmp.successors(neighbor))
            nodesuccOnAcc = list(self.GaccLastTuneTmp.successors(node))
            nodepredOnAcc = list(self.GaccLastTuneTmp.predecessors(node))

            if predSuccOnAcc:
                predSuccOnAcc= predSuccOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(neighbor,predSuccOnAcc)
                self.GaccLastTuneTmp.add_edge(neighbor, node)
                self.GaccLastTuneTmp.add_edge(node, predSuccOnAcc)
            elif not predSuccOnAcc:
                self.GaccLastTuneTmp.add_edge(neighbor, node)

            if nodesuccOnAcc and nodepredOnAcc:
                nodesuccOnAcc = nodesuccOnAcc[0]
                nodepredOnAcc = nodepredOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(nodepredOnAcc, node)
                self.GaccLastTuneTmp.remove_edge(node, nodesuccOnAcc)
                self.GaccLastTuneTmp.add_edge(nodepredOnAcc, nodesuccOnAcc)
            elif not nodesuccOnAcc and nodepredOnAcc:
                nodepredOnAcc = nodepredOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(nodepredOnAcc, node)
            elif nodesuccOnAcc and not nodepredOnAcc:
                nodesuccOnAcc = nodesuccOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(node, nodesuccOnAcc)
            elif not nodesuccOnAcc and not nodepredOnAcc:
                pass

            self.getNodeExtend(node)
            if nodesuccOnAcc:
                self.getNodeShortenBindingLastTune(nodesuccOnAcc)

        elif relation == 'success':
            nodeAcc = self.GtimeLastTuneTmp.nodes[node]['acc']
            successAcc = self.GtimeLastTuneTmp.nodes[neighbor]['acc']
            self.GtimeLastTuneTmp.nodes[node]['acc'] = successAcc
            self.GaccLastTuneTmp.nodes[node]['acc'] = successAcc
            self.AccLayersLastTuneTmp[nodeAcc].remove(node)
            successAccPos = self.AccLayersLastTuneTmp[successAcc].index(neighbor)
            self.AccLayersLastTuneTmp[successAcc].insert(successAccPos, node)

            succPredOnAcc = list(self.GaccLastTuneTmp.predecessors(neighbor))
            nodesuccOnAcc = list(self.GaccLastTuneTmp.successors(node))
            nodepredOnAcc = list(self.GaccLastTuneTmp.predecessors(node))
            if succPredOnAcc:
                succPredOnAcc = succPredOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(succPredOnAcc, neighbor)
                self.GaccLastTuneTmp.add_edge(succPredOnAcc, node)
                self.GaccLastTuneTmp.add_edge(node, neighbor)
            elif not succPredOnAcc:
                self.GaccLastTuneTmp.add_edge(node, neighbor)

            if nodesuccOnAcc and nodepredOnAcc:
                nodesuccOnAcc = nodesuccOnAcc[0]
                nodepredOnAcc = nodepredOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(nodepredOnAcc, node)
                self.GaccLastTuneTmp.remove_edge(node, nodesuccOnAcc)
                self.GaccLastTuneTmp.add_edge(nodepredOnAcc, nodesuccOnAcc)
            elif not nodesuccOnAcc and nodepredOnAcc:
                nodepredOnAcc = nodepredOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(nodepredOnAcc, node)
            elif nodesuccOnAcc and not nodepredOnAcc:
                nodesuccOnAcc = nodesuccOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(node, nodesuccOnAcc)
            elif not nodesuccOnAcc and not nodepredOnAcc:
                pass

            self.getNodeExtend(node)
            if nodesuccOnAcc:
                self.getNodeShortenBindingLastTune(nodesuccOnAcc)

        self.GaccReserveTmp = self.GaccLastTuneTmp
        self.GtimeReserveTmp = self.GtimeLastTuneTmp
        self.AccLayersReserveTmp = self.AccLayersLastTuneTmp

        self.getBindedTimeLastTune()
        self.getIfmOfmTransLastTune()


    def getModalTune(self, node):
        preds = list(self.GtimeReserve.predecessors(node))
        successs = list(self.GtimeReserve.successors(node))
        for pred in preds:
            if self.GtimeReserve.nodes[node]['acc'] != self.GtimeReserve.nodes[pred]['acc'] \
                    and self.GtimeReserve.nodes[node]['attri']['LayerType'] == self.GtimeReserve.nodes[pred]['attri']['LayerType']:
                self.getTunePerf(node,pred,'pred')
                if self.PlatformElapsed > self.getAccMaxElapsed(self.GaccLastTuneTmp):
                    self.GaccReserve = copy.deepcopy(self.GaccReserveTmp)
                    self.GtimeReserve = copy.deepcopy(self.GtimeReserveTmp)
                    self.AccLayersReserve = copy.deepcopy(self.AccLayersReserveTmp)
                    self.GaccLastTune = copy.deepcopy(self.GaccLastTuneTmp)
                    self.GtimeLastTune = copy.deepcopy(self.GtimeLastTuneTmp)
                    self.AccLayersLastTune = copy.deepcopy(self.AccLayersLastTuneTmp)
                    self.AccListLastTune = copy.deepcopy(self.AccListLastTuneTmp) # acclist is newly generated in each iteration
                    self.PlatformElapsed = self.getAccMaxElapsed(self.GaccLastTuneTmp)
        for success in successs:

            if self.GtimeReserve.nodes[node]['acc'] != self.GtimeReserve.nodes[success]['acc'] \
                    and self.GtimeReserve.nodes[node]['attri']['LayerType'] == self.GtimeReserve.nodes[success]['attri']['LayerType']:

                self.getTunePerf(node,success, 'success')
                if self.PlatformElapsed  > self.getAccMaxElapsed(self.GaccLastTuneTmp):

                    self.GaccReserve = copy.deepcopy(self.GaccReserveTmp)
                    self.GtimeReserve = copy.deepcopy(self.GtimeReserveTmp)
                    self.AccLayersReserve = copy.deepcopy(self.AccLayersReserveTmp)

                    self.GaccLastTune = copy.deepcopy(self.GaccLastTuneTmp)
                    self.GtimeLastTune = copy.deepcopy(self.GtimeLastTuneTmp)
                    self.AccLayersLastTune = copy.deepcopy(self.AccLayersLastTuneTmp)
                    self.AccListLastTune = copy.deepcopy(self.AccListLastTuneTmp)
                    self.PlatformElapsed = self.getAccMaxElapsed(self.GaccLastTuneTmp)
        if successs:
            for success in successs:
                self.getModalTune(success)
        else:
            return


    def getHomoNeighbor(self):
        lastNode = []
        for node in list(self.Gtime.nodes):
            if len(list(self.Gtime.successors(node))) == 0:
                lastNode.append(node)
        lastNode = sorted(lastNode, key=lambda x: self.Gtime.nodes[x]['end'], reverse=True)
        initNode = copy.deepcopy(lastNode)
        for i in range(len(initNode)):
            element = initNode[i]
            element = element.split("L")
            element = element[0]+'L1'
            initNode[i] = element

        self.GaccLastTune = copy.deepcopy(self.Gacc)
        self.GtimeLastTune = copy.deepcopy(self.Gtime)
        self.AccLayersLastTune = copy.deepcopy(self.AccLayers)
        self.AccListLastTune = copy.deepcopy(self.AccList)

        initNodeRepeat = []
        for i in range(2):
            initNodeRepeat = initNodeRepeat + initNode

        start = time.time()
        for node in initNodeRepeat:
            self.getModalTune(node)
        end = time.time()
        timeElapse = end-start


        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.GaccLastTune))
        print('after step4, the system time is: ', self.PlatformElapsed)
        power = self.getSysPower(self.GaccLastTune)
        print('after step4, the system energy is: ', power)
        print('the step4 search time is : ', timeElapse)


    def getIfmOfmTransLastTune(self):
        initNode = []
        for node in list(self.GaccLastTuneTmp.nodes):
            if len(list(self.GaccLastTuneTmp.predecessors(node)))==0:
                initNode.append(node)
        for node in initNode:
            self.getGraphUpdatViaModalLastTune(node)

        GaccTmp = copy.deepcopy(self.GaccLastTuneTmp)
        for node in initNode:
            layerOrder = node.split("L")
            if layerOrder[1] == '1':
                GaccTmp.remove_node(node)
            else:
                continue

        while self.getGraphSize(GaccTmp) > 0:
            node = self.getSmallEndLayerinGraph(GaccTmp)
            HWpreds = list(self.GaccLastTuneTmp.predecessors(node))
            SWpreds = list(self.GtimeLastTuneTmp.predecessors(node))
            preds = HWpreds+SWpreds
            maxend = 0
            for pred in preds:
                if self.GaccLastTuneTmp.nodes[pred]['end'] > maxend:
                    maxend=self.GaccLastTuneTmp.nodes[pred]['end']

            if maxend < self.GaccLastTuneTmp.nodes[node]['start']:
                latency = self.GaccLastTuneTmp.nodes[node]['end']-self.GaccLastTuneTmp.nodes[node]['start']
                self.GaccLastTuneTmp.nodes[node]['start'] = maxend
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + latency
                self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']

                HWsuccessor = self.GaccLastTuneTmp.successors(node)
                SWsuccessor = self.GtimeLastTuneTmp.successors(node)
                for HWsu in list(HWsuccessor):
                    self.getNodeShortenBindingLastTune(HWsu)
                for SWsu in list(SWsuccessor):
                    self.getNodeShortenBindingLastTune(SWsu)


            GaccTmp.remove_node(node)



        return

    def getGraphUpdatViaModalLastTune(self, node):
        preds = list(self.GaccLastTuneTmp.predecessors(node))
        successs = list(self.GaccLastTuneTmp.successors(node))

        if not preds and not successs:
            return

        if not preds and successs:
            success = successs[0]
            if self.getModalCheck(node,success) and self.getDependCheck(node,success):
                accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                self.getGraphUpdatViaModalLastTune(success)

            elif self.getModalCheck(node,success) and not self.getDependCheck(node,success):
                self.getGraphUpdatViaModalLastTune(success)

            elif not self.getModalCheck(node, success):
                self.getGraphUpdatViaModalLastTune(success)


        if preds and not successs:
            pred=preds[0]
            if self.getModalCheck(pred, node) and self.getDependCheck(pred, node):
                accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                self.GaccLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[pred]['end']
                self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']

                return

            if self.getModalCheck(pred, node) and not self.getDependCheck(pred, node):

                return


            elif not self.getModalCheck(pred, node):

                return

        if preds and successs:
            pred=preds[0]
            success = successs[0]

            if self.getModalCheck(node,success) and self.getModalCheck(pred,node) and \
                    self.getDependCheck(node,success) and self.getDependCheck(pred,node):
                accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=False)
                self.GaccLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[pred]['end']
                self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']

                self.getGraphUpdatViaModalLastTune(success)


            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                    not self.getDependCheck(node, success) and self.getDependCheck(pred, node):
                accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                self.GaccLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[pred]['end']
                self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']

                self.getGraphUpdatViaModalLastTune(success)


            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 self.getDependCheck(node, success) and not self.getDependCheck(pred, node):
                accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']

                self.getGraphUpdatViaModalLastTune(success)


            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 not self.getDependCheck(node, success) and not self.getDependCheck(pred, node):
                self.getGraphUpdatViaModalLastTune(success)



            elif self.getModalCheck(node, success) and not self.getModalCheck(pred, node) and \
                    self.getDependCheck(node, success):
                accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']

                self.getGraphUpdatViaModalLastTune(success)


            elif self.getModalCheck(node, success) and not self.getModalCheck(pred, node) and \
                    not self.getDependCheck(node, success):

                self.getGraphUpdatViaModalLastTune(success)


            elif not self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 self.getDependCheck(pred, node):
                accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                self.GaccLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[pred]['end']
                self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']

                self.getGraphUpdatViaModalLastTune(success)


            elif not self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 not self.getDependCheck(pred, node):

                self.getGraphUpdatViaModalLastTune(success)


            elif not self.getModalCheck(node, success) and not self.getModalCheck(pred, node):

                self.getGraphUpdatViaModalLastTune(success)


if __name__ == "__main__":

    acc1 = cnn_acc_1()
    AccsList = [acc1]

    print('Modality case example*************************************************************************************************************************')
    InitMapper = MapperInit()

    InitMapper.getModalityLayers(DNN)
    # add corss-layer dependency
    # e.g. InitMapper.Gmap.add_edge('M2L12', 'M1L13')
    InitMapper.getAccList(AccsList)
    InitMapper.getMapping()
    InitMapper.getKnapsack()
    InitMapper.getBindedTime()
    InitMapper.getIfmOfmTrans()
    InitMapper.getHomoNeighbor()
    with open("H2HMapper.pkl", "wb") as file:
        pickle.dump(InitMapper, file, True)



