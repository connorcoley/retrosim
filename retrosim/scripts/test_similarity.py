from __future__ import print_function

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(4)

import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit import DataStructs
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import sys

from retrosim.utils.generate_retro_templates import process_an_example
from retrosim.data.get_data import get_data_df, split_data_df

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
import os

SCRIPT_ROOT = os.path.dirname(__file__)
PROJ_ROOT = os.path.dirname(SCRIPT_ROOT)

############### DEFINITIONS FOR VALIDATION SEARCH ########################
all_getfp_labels = ['Morgan2noFeat', 'Morgan3noFeat', 'Morgan2Feat', 'Morgan3Feat']
all_similarity_labels = ['Tanimoto', 'Dice', 'TverskyA', 'TverskyB',]
all_dataset = ['val']

############### DEFINITIONS FOR FINAL EVALUATION #########################
all_getfp_labels = ['Morgan2Feat']
all_similarity_labels = ['Tanimoto']
all_dataset = ['test']


def ranks_to_acc(found_at_rank, fid=None):
    def fprint(txt):
        print(txt)
        if fid is not None:
            fid.write(txt + '\n')
            
    tot = float(len(found_at_rank))
    fprint('{:>8} \t {:>8}'.format('top-n', 'accuracy'))
    accs = []
    for n in [1, 3, 5, 10, 20, 50]:
        accs.append(sum([r <= n for r in found_at_rank]) / tot)
        fprint('{:>8} \t {:>8}'.format(n, accs[-1]))
    return accs


if __name__ == '__main__':
    data = get_data_df(os.path.join(PROJ_ROOT, 'data', 'data_processed.csv'))
    split_data_df(data) # 80/10/10 within each class

    for getfp_label in all_getfp_labels:
        if getfp_label == 'Morgan2noFeat':
            getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
        elif getfp_label == 'Morgan3noFeat':
            getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 3, useFeatures=False)
        elif getfp_label == 'Morgan2Feat':
            getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=True)
        elif getfp_label == 'Morgan3Feat':
            getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 3, useFeatures=True)
        else:
            raise ValueError('Unknown getfp label')

        for similarity_label in all_similarity_labels:
            if similarity_label == 'Tanimoto':
                similarity_metric = DataStructs.BulkTanimotoSimilarity
            elif similarity_label == 'Dice':
                similarity_metric = DataStructs.BulkDiceSimilarity
            elif similarity_label == 'TverskyA': # weighted towards punishing onlyA
                def similarity_metric(x, y):
                    return DataStructs.BulkTverskySimilarity(x, y, 1.5, 1.0)
            elif similarity_label == 'TverskyB': # weighted towards punishing onlyB
                def similarity_metric(x, y):
                    return DataStructs.BulkTverskySimilarity(x, y, 1.0, 1.5)
            else:
                raise ValueError('Unknown similarity label')

    	    for dataset in all_dataset:
            
                for class_ in ['all', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10][::-1]:
                    label = '{}_class{}_fp{}_sim{}'.format(
                        dataset,
                        class_,
                        getfp_label,
                        similarity_label,
                    )
                    print('#'*80)
                    print(label)
                    print('#'*80)

                    if os.path.isfile(os.path.join(SCRIPT_ROOT, 'out', '{}.txt'.format(label))):
                        print('Have done setting {} already'.format(label))
                        continue


                    ### Only get new FPs if necessary - is a little slow
                    try:
                        if prev_FP != getfp:
                            raise NameError
                    except NameError:
                        print('Getting FPs using {}'.format(getfp_label))
                        all_fps = []
                        for smi in tqdm(data['prod_smiles']):
                            all_fps.append(getfp(smi))
                        data['prod_fp'] = all_fps
                        prev_FP = getfp


                    ### Get the training data subset of the full data
                    if class_ != 'all':
                        datasub = data.loc[data['class'] == class_].loc[data['dataset'] == 'train']
                    else:
                        datasub = data.loc[data['dataset'] == 'train']
                    fps = list(datasub['prod_fp'])
                    print('Size of knowledge base: {}'.format(len(fps)))


                    ### Get the validation or test data
                    if class_ != 'all':
                        datasub_val = data.loc[data['class'] == class_].loc[data['dataset'] == dataset]
                    else:
                        datasub_val = data.loc[data['dataset'] == dataset]


                    ### Define function to process one example based on index, ix
                    def do_one_rdchiral(ix, debug=False):
                        jx_cache = {}
                        ex = Chem.MolFromSmiles(datasub_val['prod_smiles'][ix])
                        rct = rdchiralReactants(datasub_val['prod_smiles'][ix])
                        fp = datasub_val['prod_fp'][ix]
                        
                        sims = similarity_metric(fp, [fp_ for fp_ in datasub['prod_fp']])
                        js = np.argsort(sims)[::-1]

                        prec_goal = Chem.MolFromSmiles(datasub_val['rxn_smiles'][ix].split('>')[0])
                        [a.ClearProp('molAtomMapNumber') for a in prec_goal.GetAtoms()]
                        prec_goal = Chem.MolToSmiles(prec_goal, True)
                        # Sometimes stereochem takes another canonicalization...
                        prec_goal = Chem.MolToSmiles(Chem.MolFromSmiles(prec_goal), True)
                        
                        # Get probability of precursors
                        probs = {}
                        
                        for j in js[:100]: # limit to 100 for speed
                            jx = datasub.index[j]
                            
                            if jx in jx_cache:
                                (rxn, template, rcts_ref_fp) = jx_cache[jx]
                            else:
                                retro_canonical = process_an_example(datasub['rxn_smiles'][jx], super_general=True)
                                if retro_canonical is None:
                                    continue
                                template = '(' + retro_canonical.replace('>>', ')>>')
                                rcts_ref_fp = getfp(datasub['rxn_smiles'][jx].split('>')[0])
                                rxn = rdchiralReaction(template)
                                jx_cache[jx] = (rxn, template, rcts_ref_fp)
                            try:
                                outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
                            except Exception as e:
                                print(template)
                                print(datasub_val['rxn_smiles'][ix])
                                raise(e)
                                outcomes = []
                            
                            for precursors in outcomes:
                                precursors_fp = getfp(precursors)
                                precursors_sim = similarity_metric(precursors_fp, [rcts_ref_fp])[0]
                                if precursors in probs:
                                    probs[precursors] = max(probs[precursors], precursors_sim * sims[j])
                                else:
                                    probs[precursors] = precursors_sim * sims[j]
                        
                        testlimit = 50
                        found_rank = 9999
                        for r, (prec, prob) in enumerate(sorted(probs.iteritems(), key=lambda x:x[1], reverse=True)[:testlimit]):
                            if prec == prec_goal:
                                found_rank = min(found_rank, r + 1)
                        if found_rank == 9999:
                            print('## not found ##')
                            print(datasub_val['rxn_smiles'][ix])
                            print(prec_goal)

                        return found_rank

                    ### Run all validation/testing examples in parallel
                    inputs = list(datasub_val.index)
                    found_at_rank = Parallel(n_jobs=num_cores, verbose=5)(delayed(do_one_rdchiral)(i) for i in inputs)
                    
                    ### Save to individual file
                    with open(os.path.join(SCRIPT_ROOT, 'out', '{}.txt'.format(label)), 'w') as fid:
                        accs = ranks_to_acc(found_at_rank, fid=fid)

                    ### Save to global results file
                    if not os.path.isfile(os.path.join(SCRIPT_ROOT, 'out', 'results.txt')):
                        with open(os.path.join('out', 'results.txt'), 'w') as fid2:
                            fid2.write('\t'.join(['{:>16}'.format(x) for x in [
                                'dataset', 'class_', 'n_in_class', 'getfp_label', 'similarity_label',
                                'top-1 acc', 'top-3 acc', 'top-5 acc', 'top-10 acc',
                                'top-20 acc', 'top-50 acc']]) + '\n')
                    with open(os.path.join(SCRIPT_ROOT, 'out', 'results.txt'), 'a') as fid2:
                        fid2.write('\t'.join(['{:>16}'.format(x) for x in [
                            dataset, class_, len(found_at_rank), getfp_label, similarity_label,
                        ] + accs]) + '\n')



