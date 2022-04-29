from math import *

class cnn_acc_1:
    # This is the performance model of "Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks"
    # FPGA board VC707
    # R_wei is the ratio of DDR space that is reserved for weights
    # Bandwidth is the bandwidth between private memory and accelerator
    # R_wei = 0.5 half of the DDR is reserved for weight locality optimization
    def __init__(self, throughput = 61.62, W_mem=1.25, S_mem = 1, R_wei = 0.5, Dtype=32, Facc=100, Bandwidth=512, Tm=64, Tn=7, Tr=13, Tc=13, Tr_=13, Tc_=13, power=18.61):
        # the parameters that need to be transferred in
        self.M = None
        self.N = None
        self.R = None
        self.C = None
        self.K = None
        self.S = None
        self.P = None
        # the parameters that is fixed for an accelerator
        self.W_mem = W_mem
        self.S_mem = S_mem
        self.R_wei = R_wei
        self.Dtype = Dtype
        self.Facc = Facc
        self.Bandwidth = Bandwidth
        self.Tm = Tm
        self.Tn = Tn
        self.Tr = Tr
        self.Tc = Tc
        self.Tr_ = Tr_
        self.Tc_ = Tc_
        self.isPingPong = True
        self.accName = 'cnn_acc_1'
        self.accType = 'CNN'
        # represent the number of extra weight element
        self.power = power
        self.extraWeightElement = self.S_mem * self.R_wei * 8 * (10**9)/ self.Dtype
        self.assiLayers = []
        self.bindLayer = []
        # Gop/s
        self.Th = throughput

    def getExtraWeightElement(self):
        self.extraWeightElement = self.S_mem * self.R_wei * 8 * (10 ** 9) / self.Dtype

    def getPower(self):
        return self.power

    def getRunPara(self, Para):
        self.M = Para['M']
        self.N = Para['N']
        self.R = Para['R']
        self.C = Para['C']
        self.K = Para['K']
        self.S = Para['S']
        self.P = Para['P']
        self.LayerName = Para['LayerName']

    def getAccName(self):
        return self.accName

    def getOps(self):
        return 2*self.R * self.C * self.M * self.K * self.K * self.N

    def getTileOps(self):
        return 2 * self.Tr * self.Tc * self.Tm * self.K * self.K * self.Tn

    def getCompPerf(self):
        ops = self.getTileOps()/(10**9)
        tComp = ops/self.Th
        return tComp

    def getS_mem2AccPerfPerTile(self):
        S_mem2AccBand = self.Bandwidth * self.Facc * (10 ** 6)
        tWeight = (self.Tm * self.Tn * self.K * self.K) \
                  * self.Dtype / S_mem2AccBand
        tIfm = (self.Tn * self.Tr_ * self.Tc_) \
               * self.Dtype / S_mem2AccBand
        tDataIn = tWeight + tIfm
        tDataOut = (self.Tm * self.Tr * self.Tc) * self.Dtype / S_mem2AccBand
        return tDataIn, tDataOut

    def getDataInPerf(self, WEItrans, IFMtrans, OFMtrans):
        S_mem2MemBand = self.W_mem * (10**9) * 8
        if WEItrans==True and IFMtrans==True:
            tWeight = (self.M * self.N * self.K * self.K) \
                  * self.Dtype / S_mem2MemBand
            tIfm = (self.N * self.R * 2 * self.C * 2) \
                  * self.Dtype / S_mem2MemBand
            tDataIn = tWeight + tIfm
        if WEItrans==False and IFMtrans==True:
            tDataIn = ( self.N * self.R * 2 * self.C * 2) * self.Dtype / S_mem2MemBand
        if WEItrans==True and IFMtrans==False:
            tDataIn = (self.M * self.N * self.K * self.K) \
                  * self.Dtype / S_mem2MemBand
        if WEItrans==False and IFMtrans==False:
            tDataIn = 0

        if OFMtrans==True:
            tDataOut = (self.M * self.R * self.C ) \
                  * self.Dtype / S_mem2MemBand
        if OFMtrans == False:
            tDataOut = 0
        return tDataIn, tDataOut

    def getLayerPerf(self, WEItrans=True,IFMtrans=True, OFMtrans=True):
        tDataInfromMem, tDataOutfromMem = self.getDataInPerf(WEItrans, IFMtrans, OFMtrans)
        # process a layer tile by tile in this paper with loop tiling optimization
        tDataIn, tDataOut = self.getS_mem2AccPerfPerTile()
        tComp = self.getCompPerf()
        Lat1 = tDataIn + tComp
        Lat2 = ceil(self.N / self.Tn) * Lat1 + tDataOut
        LatSec = ceil(self.R / self.Tr) * ceil(self.C / self.Tc) * ceil(self.M / self.Tm) * Lat2
        LatSec = LatSec + tDataInfromMem + tDataOutfromMem
        return LatSec, self.Th

