'''This is basic version of Knapsack problem from several open-source project'''
'''The optimized Knapsack solver in H2H is not revealed in this program'''

import numpy as np
import math

bias = 1000

def pack(AccObject, Gtime, AssignedLayers, bindedLayers):
    AccObject.getExtraWeightElement()
    C = math.ceil(AccObject.extraWeightElement/bias)
    AccObject.assiLayers = AssignedLayers
    AccObject.bindLayer = []

    if bindedLayers:
        for layer in AssignedLayers:
            if layer in bindedLayers:
                AccObject.bindLayer.append(layer)
                bindLayerWei = math.ceil(Gtime.nodes[layer]['weight'] / bias)
                Gtime.nodes[layer]['bind'] = True
                C = C - bindLayerWei

        for element in  AccObject.bindLayer:
            AssignedLayers.remove(element)

    num = len(AssignedLayers)
    v = []
    price = []
    for i in range(len(AssignedLayers)):
        layerWei = math.ceil(Gtime.nodes[AssignedLayers[i]]['weight'] / bias)
        v.append(layerWei)
        price.append(layerWei)

    sum = np.zeros((num + 1, C + 1))
    for jew in range(num + 1):
        for c in range(C + 1):
            if (jew == 0):
                sum[jew][c] = 0
            else:
                sum[jew][c] = sum[jew - 1][c]
            if (jew > 0 and c >= v[jew - 1]):

                sum[jew][c] = max(sum[jew - 1][c], sum[jew - 1][c - v[jew - 1]] + price[jew - 1])


    pack = np.zeros((num))
    volume = C

    for jew in range(num, -1, -1):
        if (sum[jew][volume] > sum[jew - 1][volume]):
            pack[jew - 1] = 1
            volume = volume - v[jew - 1]

    pack = [int(x) for x in pack]

    for i in range(len(pack)):
        if pack[i]==1:
            Gtime.nodes[AssignedLayers[i]]['bind'] = True
            AccObject.bindLayer.append(AssignedLayers[i])
        else:
            continue

    return AccObject, Gtime





