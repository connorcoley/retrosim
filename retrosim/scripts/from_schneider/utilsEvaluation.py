#
#  Copyright (c) 2016, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
#
#     * Redistributions of source code must retain the above copyright 
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following 
#       disclaimer in the documentation and/or other materials provided 
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc. 
#       nor the names of its contributors may be used to endorse or promote 
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Created by Nadine Schneider, July 2016


from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import time

import itertools
import random

from rdkit import Chem
from rdkit.Chem import AllChem

#import the rxn role assignment functionality from the rdkit contrib directory
import sys
from rdkit.Chem import RDConfig
sys.path.append(RDConfig.RDContribDir)

from RxnRoleAssignment import identifyReactants, utils


def _normalizeRXNAndCalcReactantSetByMapping(rxn):
    utils.removeAgentsAndTransferToReactants(rxn)    
    numReacts = rxn.GetNumReactantTemplates()
    reactsByMapping=set()
    for i in range(numReacts):
        react = rxn.GetReactantTemplate(i)
        for a in react.GetAtoms():
            if a.HasProp('molAtomMapNumber'):
                reactsByMapping.add(i)  
                break
    return reactsByMapping

def _identifyReactantsBaseline(rxn):
    reactants = rxn.GetReactants()
    uniqueReactants,reactantSmiles = utils.uniqueMolecules(reactants)
    uniqueRfinal=sorted(set(uniqueReactants.values()))
    numReactants = len(uniqueRfinal)
    numAtms = [reactants[r].GetNumAtoms() for r in uniqueRfinal]
    numAtmsProd=0
    for tmpl in rxn.GetProducts():
        numAtmsProd+=tmpl.GetNumAtoms()
    possibleReactants=[]
    for i in range(1,numReactants+1):
        for x in itertools.combinations(range(numReactants),i):
            sampleNumAtoms = [numAtms[j] for j in x]
            if len(sampleNumAtoms) >= 5 and np.median(sampleNumAtoms) <= 2:
                continue
            numAtmsSample=np.sum(sampleNumAtoms)
            if abs(numAtmsSample-numAtmsProd) <= numAtmsProd*0.3:
                possibleReactants.append(x)
    finalReactants=defaultdict(list)
    for s in possibleReactants:
        tmp=[uniqueRfinal[r] for r in s]
        finalReactants[len(tmp)].append(tmp)
    if len(finalReactants):
        minimalSol = sorted(finalReactants)[0] 
        return finalReactants[minimalSol]
    else:
        return []


def calcNumDifferences(dataSet, idRXNCls, idSmiles, idSetReactants, recalcReactantSet=False, useBaseline=False):
    numDifferences = defaultdict(list)
    count=0
    timings=[]
    numSols=[]

    random.seed(42)
    
    for row in dataSet.itertuples():
        cls = row[idRXNCls]
        smi = row[idSmiles]
        mapping = row[idSetReactants]
        rxn = AllChem.ReactionFromSmarts(str(smi).split()[0], useSmiles=True)
        if recalcReactantSet:
            mapping = _normalizeRXNAndCalcReactantSetByMapping(rxn)
        starttime = time.time()
        if useBaseline:
            res = _identifyReactantsBaseline(rxn)
        else:
            res = identifyReactants.identifyReactants(rxn)
        timings.append(time.time()-starttime)

        min_diffs=100
        if useBaseline:
            random.shuffle(res)
            if len(res):
                reacts = set(res[0])
            else:
                reacts = set()
            min_diffs = len(reacts.symmetric_difference(mapping))
        else:
            diff1=set()
            reacts=set()
            for r in res[0]:
                reacts = set(r)
                if len(res[1]):
                    reacts |= set(res[1])
                diffs = reacts.symmetric_difference(mapping)
                if len(diffs) < min_diffs:
                    min_diffs = len(diffs)
                    diff1 = mapping.difference(reacts)
            numSols.append((len(res[0]),min_diffs))
            
        numDifferences[cls].append([min_diffs,smi])
        count+=1
        if not count%10000:
            print("done", count)
    return numDifferences, timings, numSols

##### DRAWING #######

def setPlotProperties(ax):
    ax.yaxis.grid(b=True, which='major', color='gray', linestyle=':',linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left() 
    
def plotNumberSolutions(numSols):
    solutions=[0,0,0,0,0]
    numReactions=0
    for s in np.array(numSols)[:,0]:
        numReactions+=1
        solutions[s]+=1
    solutions = [(x/float(numReactions))*100 for x in solutions]
    print(solutions)

    fig, ax = plt.subplots(dpi=300, figsize=(4,3))

    bins = np.arange(0,5,1)
    b = ax.bar(bins, solutions, color='#003b6f')
    width = b[0].get_width()*0.5
    ax.set_xticks(bins + width)
    ax.set_xticklabels(bins)
    ax.set_yticks(np.arange(0,120,20))

    ax.set_ylabel('% reactions',fontsize=12)
    ax.set_xlabel('# solutions',fontsize=12)

    setPlotProperties(ax)
    plt.tight_layout()
    return fig
    
def plotNumberDifferences(numDifferences):
    differences=[0,0,0,0,0]
    numReactions=0
    for k,v in numDifferences.items():
        numReactions+=len(v)
        for j in v:
            if j[0] < 5:
                differences[j[0]]+=1

    dd = [float(i)/numReactions for i in differences]
    print(dd)

    fig, ax = plt.subplots(dpi=300, figsize=(4,3))

    bins = np.arange(0,5,1)
    b = ax.bar(bins, [x*100 for x in dd], color='#003b6f')
    width = b[0].get_width()*0.5
    ax.set_xticks(bins + width)
    ax.set_xticklabels(bins)
    ax.set_yticks(np.arange(0,120,20))

    ax.set_ylabel('% reactions',fontsize=12)
    ax.set_xlabel('# differences',fontsize=12)

    setPlotProperties(ax)
    plt.tight_layout()
    return fig
    
def plotNumberDifferencesRxnCls(numDifferences, rxnCls):
    fig, axes = plt.subplots(2,5, dpi=300, figsize=(16,5), sharey=True, sharex=True)

    count = 0
    bins = np.arange(0,5,1)
    row=0
    xlabel=''
    for k,v in sorted(numDifferences.items(),key=lambda x: int(x[0])):  
        led=str(len(v))
        vals=[x[0] for x in v]
        values = [(x/float(led))*100 for x in Counter(vals).values()]
        if len(values) < 5:
            values+=[0]*(5-len(values))
        if len(values) > 5:
            values=values[:5]
        ax = axes[row,count]    
        b = ax.bar(bins, values, color='#003b6f')
        width = b[0].get_width()*0.5
        ax.set_xticks(bins + width)
        ax.set_xticklabels(bins)
        ax.set_yticks(np.arange(0,140,20))
        ax.set_ylim(0,100)
        ttl = ax.set_title(rxnCls[str(k)],fontsize=12)
        ttl.set_position([.5, 1.05])
        if count == 0:
            ax.set_ylabel('% reactions',fontsize=12)
        ax.set_xlabel(xlabel,fontsize=12)
        setPlotProperties(ax)

        count+=1
        if count==5:
            row=1
            count=0
            xlabel='# differences'

    plt.tight_layout()
    return fig
