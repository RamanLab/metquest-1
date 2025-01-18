# -*- coding: utf-8 -*-

from __future__ import absolute_import

import math
import itertools
import time
from numpy import prod
from networkx import get_node_attributes
from metquest.guided_bfs import forward_pass
from metquest.generate_partitions import generate_partitions


def find_pathways(G, seed_mets_input, path_len_cutoff, *args):
    """
    This function tries to identify pathways between a set of seed and
    target metabolites of a given size cut-off.

    Parameters
    ----------
    G : NetworkX DiGraph Object
        Bipartite graph of the metabolic network
    seed_mets_input : set
        Set of seed metabolites including the source
    path_len_cutoff : int
        Maximum size of the pathways
    *args
        Used to decide if a particular combination has to be evaluated or not.
        i.e., if the number of pathways produced for two different metabolites
        are higher, for instance, if in the reaction A + B -> C,
        A has 2000 pathways and B has 5 pathways, then C will have 10000 pathways
        at the maximum. If this pathway cutoff (maxnumpath) is 1000,
        this combination will not be evaluated, provided C has been
        already found.
        By default, it is set to 1000

    Returns
    -------
    pathway_table : dict
        Dictionary of dictionary containing the pathways of different sizes
        identified for every metabolite. This will have only the acyclic/
        branched pathways.
    cyclic_pathways : dict
        Dictionary of dictionary containing cyclic pathways of different sizes
        identified for every metabolite.
    scope : set
        Set of metabolites which can be synthesised
    """

    global succ, pred, lower_bound_metabolite, maxnumpath, seedmets, \
        pathway_table, cyclic_pathways
    pathway_table = {}
    cyclic_pathways = {}
    tic = time.perf_counter()
    #  Setting the cutoff for maximum number of pathways
    if args:
        for maxnumpath_input in args:
            maxnumpath = maxnumpath_input
    else:
        maxnumpath = 1000
    # DiGraph definitions successors and predecessors
    succ = G.successors
    pred = G.predecessors
    seedmets = seed_mets_input
    # Removing reactions whose reactants are more than 5
    node_attributes = get_node_attributes(G, 'bipartite')
    node_attributes_inverted_dict = {}
    # 0 metabolites , 1 are reactions
    for keys, values in node_attributes.items():
        node_attributes_inverted_dict[values] = \
            node_attributes_inverted_dict.get(values, [])
        node_attributes_inverted_dict[values].append(keys)
    # If the number of metabolites, apart from the ones provided
    # in seed are greater than 5, such reactions are removed
    for rxnstoremove in node_attributes_inverted_dict[1]:
        if len(set(pred(rxnstoremove)) - seedmets) >= 5:
            G.remove_node(rxnstoremove)
    # Performing guided BFS on directed graph by calling forward_pass
    lower_bound_metabolite, status_dict, scope = forward_pass(G, seedmets)
    # Sorting the keys (reactions) in the status dictionary,
    # since dictionary keys are not good to iterate over.
    # There could be differences in the order of insertion of
    # dictionary keys. Although, this does not matter with the
    # algorithm implementation.
    rxns_to_visit = list(status_dict.keys())
    rxns_to_visit.sort()
    # For seed metabolites, the pathway table is initialised to 0
    for seedmetabs in list(seedmets):
        pathway_table[seedmetabs] = {0: ''}
    # Status dict consists of all the reactions that can be
    # visited from the seed metabolites
    for rxns in rxns_to_visit:
        if set(pred(rxns)).issubset(seedmets):
            # Initialisation of dictionary with the
            # metabolites produced with one rxn
            for metssucc in succ(rxns):
                if metssucc not in pathway_table:
                    pathway_table[metssucc] = {1: []}
            # Filling table with one reaction that produced metabolite
            for metssucc in succ(rxns):
                # Since we don't want pathways generating seed metabolites
                if metssucc not in seedmets:
                    pathway_table[metssucc][1].append(set([rxns]))

    # For filling values from the second column
    for currentcolumnidx in range(2, path_len_cutoff+1):
        for rxns in status_dict:  # rxns_to_visit:
            # To eliminate seed metabolites, whose column value
            # is always 0 - so that more partitions are not generated.
            mets_needed = list(set(pred(rxns)) - seedmets)
            # To only go over reactions whose inputs are not
            # seed metabolites. There could be reactions whose inputs
            # are only seed mets, eg atp + h2o
            if mets_needed:
                for val in range(currentcolumnidx-1,
                                 len(mets_needed)*(currentcolumnidx-1)+1):
                    if val <= len(mets_needed)*(currentcolumnidx-2):
                        _first_round_calculations(
                            mets_needed, currentcolumnidx, rxns, val)
                    else:
                        _second_round_calculations(
                            mets_needed, currentcolumnidx, rxns, val)
    toc = time.perf_counter()
    timetaken = toc - tic
    print('Time taken', timetaken)
    return pathway_table, cyclic_pathways, scope


def _first_round_calculations(mets_needed, currentcolumnidx, rxns, val):
    """
    This function takes as input metabolites required by the reaction,
    current column index (pathway length) which we are trying to fill,
    the current reaction which we are considering and the sum that we
    need to generate. This function tries to calculate the upper limit
    on the number of inputs that can take the value of (k-1) and generate
    partitions which have not been previously generated. For more
    details, please refer main manuscript (Algorithm 1 - Lines 14-19)

    Parameters
    ----------
    mets_needed : list
        List of metabolites a reaction requires
    currentcolumnidx : int
        An integer denoting the current column which is evaluated
    rxns : str
        Current reaction which is evaluated
    val : int
        Maximum sum that is to be generated

    Returns
    -------
    None
    """

    # Line 15 in the algorithm - Upper limit on the number of
    # inputs that can take the value of (k-1)
    optimized_val = int(math.floor(val/((currentcolumnidx-1))))
    for currentval in range(1, optimized_val+1):
        # GENERATING COMBINATIONS
        # This will give combinations of  metabolites whose size is
        # defined by currentval
        for currentmetcomb in itertools.combinations(mets_needed, currentval):
            temp_rxn_list = []
            number_of_pathways_found = {}
            onemetnotfound = ''
            for metabolites in list(currentmetcomb):
                if metabolites in pathway_table:
                    # To ensure that the current iteration uses metabs
                    # generated only till the previous iteration
                    if currentcolumnidx - 1 in pathway_table[metabolites]:
                        temp_rxn_list.append([[rxns]])
                        number_of_pathways_found[metabolites] = \
                            len(pathway_table[metabolites][currentcolumnidx-1])
                    else:
                        # Because one of the metabolites is not found,
                        # hence breaks out of the second loop
                        onemetnotfound = 'Y'
                        break
            if onemetnotfound != 'Y':
                other_mets_not_in_comb = list(set(mets_needed) - set(currentmetcomb))
                if set(currentmetcomb).issubset(pathway_table):
                    for mets in currentmetcomb:
                        # Assigning the value of columnindex -1
                        # to the metabolites in current metabolite combination
                        temp_rxn_list.append(pathway_table[mets][currentcolumnidx-1])

                    first_discovery_step = []
                    # This will give values of the lower bound of
                    # metabolites which are not involved in combination
                    for varmet in list(other_mets_not_in_comb):
                        first_discovery_step.append(min(lower_bound_metabolite[varmet]))
                    all_partitions = generate_partitions(val-((currentcolumnidx-1)*currentval),
                                                         first_discovery_step, currentcolumnidx - 1)
                    for partitions in all_partitions:
                        _find_all_rxn_combination_firstround(rxns, partitions, other_mets_not_in_comb,
                                                             temp_rxn_list, number_of_pathways_found,
                                                             currentcolumnidx)


def _find_all_rxn_combination_firstround(rxns, partitions, other_mets_not_in_comb,
                                        temp_rxn_list, number_of_pathways_found, currentcolumnidx):
    """
    This function fetches the pathways from the table corresponding to
    the entries in partition generated.

    Parameters
    ----------
    rxns : str
        Current reaction which is evaluated
    paritions : tuple
        Combinations of numbers that would geenrate the required sum
    other_mets_not_in_comb : list
        other metabolites participating in the reaction for which a
        number from the partition has not been assigned yet
    temp_rxn_list : list
        a list consisting of all pathways that generated the metabolite
        for which the value of (k-1) is assigned
    number_of_pathways_found : dict
        number of pathways found for the metabolite evaluated
    currentcolumnidx : int
         value of the current column index (pathway length)

    Returns
    -------
    None
    """
    # To check if all the other metabolites required by that reaction
    # can be generated using the partitions.
    counter = 0
    more_pathways_found = ''
    for varmetidx in range(len(other_mets_not_in_comb)):
        if other_mets_not_in_comb[varmetidx] in pathway_table:
            # checking if the pathway table consists of values from the
            # current partition for the input metabolite
            if partitions[varmetidx] in pathway_table[other_mets_not_in_comb[varmetidx]]:
                counter += 1
                number_of_pathways_found[other_mets_not_in_comb[varmetidx]] = \
                    len(pathway_table[other_mets_not_in_comb[varmetidx]][partitions[varmetidx]])
    if counter == len(other_mets_not_in_comb):
        if all([prod(list(number_of_pathways_found.values())) > maxnumpath,
                set(succ(rxns)).issubset(set(pathway_table))]):
            more_pathways_found = 'Y'
        else:
            # Deep copy of the reaction list, because temp_rxn_list_current
            # varies with every iteration to evaluate other partitions
            temp_rxn_list_current = temp_rxn_list[:]
            for varmetidx in range(len(other_mets_not_in_comb)):
                temp_rxn_list_current.append(
                    pathway_table[other_mets_not_in_comb[varmetidx]][partitions[varmetidx]])
            _populate_table(rxns, temp_rxn_list_current, currentcolumnidx)


def _populate_table(rxns, temp_rxn_list_current, currentcolumnidx):
    """
    This function fills in the entry in the main pathway table. It also
    evaluates the pathways and identifies if its cyclic. If the
    pathway is a cyclic pathway, it is stored separately, and is not
    used in the subsequent calculations.

    Parameters
    ----------
    rxns : str
        Current reaction which is evaluated
    paritions : tuple
        Combinations of numbers that would geenrate the required sum
    temp_rxn_list_current : list of lists
        a list of lists consisting of all the alternate pathways
        producing the metabolites required by the reaction
    currentcolumnidx : int
         value of the current column index (pathway length)

    Returns
    -------
    None
    """
    #  Temprxnlist consists of all combinations of pathways
    #  producing all the input metabolites
    for rxnunion in itertools.product(*temp_rxn_list_current):
        reaction_combntn = set([])
        for rxnentry in rxnunion:
            for individualele in rxnentry:
                reaction_combntn.add(individualele)
        metabs_in_cycle = set([])

        for currentrxn in reaction_combntn:
            for inputmetab in pred(currentrxn):
                if inputmetab not in seedmets:
                    metabs_in_cycle.add(inputmetab)
        for succmets in succ(rxns):
            if succmets not in seedmets:
                if len(reaction_combntn) >= currentcolumnidx:
                    if succmets in pathway_table:
                        if succmets in metabs_in_cycle:
                            if succmets in cyclic_pathways:
                                # multiple reaction_combntn sets may exist the metabolite in a column k, check whether cmbnt exists before appending
                                if not any(set(reaction_combntn) == existing_set for existing_set in cyclic_pathways[succmets].get(len(reaction_combntn), [])):
                                    cyclic_pathways[succmets].setdefault(len(reaction_combntn), []).append(set(reaction_combntn))
                                # change from .update to .setdefault method
                                # cyclic_pathways[succmets].update(
                                #     {len(reaction_combntn): [list(reaction_combntn)]})
                            else:
                                cyclic_pathways[succmets] = \
                                    {len(reaction_combntn): [list(reaction_combntn)]}
                        else:
                            if len(reaction_combntn) in pathway_table[succmets]:
                                try:
                                    if pathway_table[succmets][len(reaction_combntn)]. \
                                                                index(reaction_combntn):
                                        pass  # Because this entry is already in the pathway_table
                                except ValueError:
                                    pathway_table[succmets][len(reaction_combntn)].append(
                                        reaction_combntn)
                            else:
                                pathway_table[succmets].update(
                                    {len(reaction_combntn): [reaction_combntn]})

                    else:
                        pathway_table[succmets] = {len(reaction_combntn): [reaction_combntn]}

def _second_round_calculations(mets_needed, currentcolumnidx, rxns, val):
    """
    This function takes into account all the metabolites required by the
    reaction and based on the partition of numbers, fetches the values
    from the pathway table. Since there can be multiple alternate
    routes to produce the same metabolite, this function makes a list
    of all possible pathways (of the given size) that produces this
    metabolite of interest.

    Parameters
    ----------
    mets_needed : list
        List of metabolites a reaction requires
    currentcolumnidx : int
        An integer denoting the current column which is evaluated
    rxns : str
        Current reaction which is evaluated
    val : int
        Maximum sum that is to be generated

    Returns
    -------
    None
    """
    first_discovery_step = []
    for predmets in mets_needed:
        first_discovery_step.append(min(lower_bound_metabolite[predmets]))
    all_partitions = generate_partitions(val, first_discovery_step, currentcolumnidx-1)
    for partitions in all_partitions:
        temp_rxn_list = []
        number_of_pathways_found = {}
        more_pathways_found = ''
        counter_new = 0
        temp_rxn_list.append([[rxns]])
        for item in range(len(mets_needed)):
            if mets_needed[item] in pathway_table:
                if partitions[item] in pathway_table[mets_needed[item]]:
                    number_of_pathways_found[mets_needed[item]] = \
                        len(pathway_table[mets_needed[item]][partitions[item]])
                    counter_new += 1
        if counter_new == len(mets_needed):
            if all([prod(list(number_of_pathways_found.values())) > maxnumpath,
                    set(succ(rxns)).issubset(set(pathway_table))]):
                more_pathways_found = 'NA'
            else:
                for item in range(len(mets_needed)):
                    temp_rxn_list.append(pathway_table[mets_needed[item]][partitions[item]])
                _populate_table(rxns, temp_rxn_list, currentcolumnidx)
