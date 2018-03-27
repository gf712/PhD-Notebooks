#! /usr/bin/env python3

import mdtraj as md
from glob import glob
import numpy as np
from scipy import spatial
from tqdm import tqdm
import warnings
import subprocess
import threading
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import requests
from multiprocessing import Manager


def calculate_ca_distance(file, threshold=0.4):

    """
    Function to calculate the C_alpha distance between
    amino acids that form salt bridges.
    It is assumed that only the NZ and NH1 atoms of lysine
    and argenine, respectively, can participate in this type
    of interaction with OD1 and OE1 atoms of aspartate and
    glutamate, within the given threshold

    Args:
        file (str): path to pdb file
        threshold (float): maximum distance (in nm) between oposite
            charged atoms to form a salt bridge

    Returns:
        list: distance in nm between C_alpha of residues interacting
            via a salt bridge

    Raises:
        ValueError: Undefined errors used for debugging

    Warning:
        Emits a warning when there is an issue with PDB file
        and it cannot be parsed by mdtraj and returns an empty list
    """

    try:

        try :
            pdb = md.load_pdb(file)

        except:
            warnings.warn("Error opening {}".format(file))
            return list()

        # extract atoms from residues that can form salt bridges
        lys_atoms = [pdb.top.atom(x).residue.atoms for x in pdb.top.select('resname LYS and name CA')]
        arg_atoms = [pdb.top.atom(x).residue.atoms for x in pdb.top.select('resname ARG and name CA')]
        asp_atoms = [pdb.top.atom(x).residue.atoms for x in pdb.top.select('resname ASP and name CA')]
        glu_atoms = [pdb.top.atom(x).residue.atoms for x in pdb.top.select('resname GLU and name CA')]
        
        # filter list by atoms of interest -> C_alpha and charged atoms
        lys = [list(filter(lambda x: x.name == 'CA' or x.name == 'NZ' , t))  for t in lys_atoms]
        arg = [list(filter(lambda x: x.name == 'CA' or x.name == 'NH1' , t)) for t in arg_atoms]
        asp = [list(filter(lambda x: x.name == 'CA' or x.name == 'OD1' , t)) for t in asp_atoms]
        glu = [list(filter(lambda x: x.name == 'CA' or x.name == 'OE1' , t)) for t in glu_atoms]
        
        # filter list of atoms to ensure that we always have a pair -> C_alpha, charged atom
        # if not this will introduce noise to the processed data
        lys_result = list(filter(lambda x: len(x) == 2, lys))
        arg_result = list(filter(lambda x: len(x) == 2, arg))
        asp_result = list(filter(lambda x: len(x) == 2, asp))
        glu_result = list(filter(lambda x: len(x) == 2, glu))
        
        # extract CA and charged atoms of residues that have atoms that can form salt bridges
        lys_ca = [[atom.index for atom in res if atom.name == 'CA'][0] for res in lys_result]
        arg_ca = [[atom.index for atom in res if atom.name == 'CA'][0] for res in arg_result]
        asp_ca = [[atom.index for atom in res if atom.name == 'CA'][0] for res in asp_result]
        glu_ca = [[atom.index for atom in res if atom.name == 'CA'][0] for res in glu_result]
        
        lys_charged = [[atom.index for atom in res if atom.name == 'NZ'][0]  for res in lys_result]
        arg_charged = [[atom.index for atom in res if atom.name == 'NH1'][0] for res in arg_result]
        asp_charged = [[atom.index for atom in res if atom.name == 'OD1'][0] for res in asp_result]
        glu_charged = [[atom.index for atom in res if atom.name == 'OE1'][0] for res in glu_result]

        # extract coordinates of relevant atoms
        traj = pdb.xyz[0]

        pos = np.concatenate((lys_charged, arg_charged)).astype(int) # enforce int type
        neg = np.concatenate((asp_charged, glu_charged)).astype(int)

        pos_ca = np.concatenate((lys_ca, arg_ca)).astype(int)
        neg_ca = np.concatenate((asp_ca, glu_ca)).astype(int)

        pos_traj = traj[pos]
        neg_traj = traj[neg]

        pos_ca_traj = traj[pos_ca]
        neg_ca_traj = traj[neg_ca]

        # calculate a distance matrix
        # rows: positively charged atoms
        # columns: negatively charged atoms
        dists = spatial.distance.cdist(pos_traj, neg_traj)

        # get distance between C_alpha of residues which form salt bridges
        ca_dist = list()

        for pair in zip(*np.where(dists < threshold)):

            pos_i, neg_i = pair

            pos_ca_i = pos_ca_traj[pos_i]
            neg_ca_i = neg_ca_traj[neg_i]

            ca_dist.append(spatial.distance.euclidean(pos_ca_i, neg_ca_i))

        return ca_dist

    except:
        raise ValueError("Error with file: {}".format(file))


def worker(ca_dists, completed, name=None, file=None):
    """
    Worker that executes all the steps required to obtain
    the C_alpha distances.

    Params:
        ca_dists (multiprocessing.Manager.dict): stores all the C_alpha distances
        name (str): name of PDB to store in ca_dists.
            Default is None. In this case the file name is
            used to name the entry.
        file (str): path to PDB file

    Returns:
        Nothing.
    """


    if file is not None and name is None:
        # use local pdb copy and use name as dict key
        PDB_id = file.split('/')[-1].split('.')[0]
    else:
        # download copy by identifier and use ID as dict key
        if len(name) > 0:
            PDB_id = name

        else:
            #  skip
            completed.value += 1
            return

    # download PDB if it is not a local file
    if file is None:
        file = './Data/{}'.format(PDB_id)
        try:

            output = subprocess.check_output(['wget', '-O', file,
                                              'https://files.rcsb.org/download/{}.pdb'.format(PDB_id)],
                                              stderr=subprocess.STDOUT)

        except subprocess.CalledProcessError as exc:
            print("Status : FAIL", PDB_id, exc.returncode, exc.output)
    else:
        # this is just for sanity checks
        # creates an empty file with the same name as the file
        # being processed
        output = subprocess.check_output(['touch', './Data/{}'.format(PDB_id)])

    try:
        ca_dists[PDB_id] = calculate_ca_distance(file, threshold=0.4)
    except Exception as e:
        print("Status : FAIL", e.error)

    # clean up when done
    subprocess.check_output(['rm', './Data/{}'.format(PDB_id)])

    sys.stdout.flush()

    completed.value += 1


def worker_group(ca_dists, completed, names=None, files=None):

    """
    Groups up work, so that one process can perform more
    than one job at a time and be more efficient
    """

    if names is not None:
        for name in names:
            worker(ca_dists, completed, name=name)
    if files is not None:
        for file in files:
            worker(ca_dists, completed, file=file)


def split_list(a, n=4):

    """
    Splits list into n iterators.
    """
    iter_lengths = [int(len(a) / n)] * n
    # add remainder to last process
    # iter_lengths[-1] += iter_lengths[-1] + len(a) % n

    iterators = []

    i = 0

    for x in iter_lengths:
        if isinstance(a[x], list):
            iterators.append([a[x][0] for x in range(i, i+x) if len(a[x]) == 1])
        else:
            iterators.append([a[x] for x in range(i, i+x)])
        i += x

    current_length = sum(iter_lengths)
    remainder = len(a) % n

    i = 0

    while remainder > 0:
        iterators[i].append(a[current_length + remainder - 1])
        remainder -= 1
        i += 1

    return iterators


def main(threshold, directory=None, strategy = '2*jobs', n_jobs = -1):

    """
    Main function to run workers.

    Params:
        threshold (float): Salt bridge length threshold
        directory (str): path to directory
        strategy (str): how to dispatch the jobs
        n_jobs (int): number of processes to use
    """

    if directory is None:

        print('Retrieving PDB IDs.')

        # download name of all PDBs on rcsb
        import re

        pattern = re.compile(r'^(\w{4}) ; .')

        try:
            output = subprocess.check_output(['wget', '-O', 'author.idx',
                                              'ftp://ftp.wwpdb.org/pub/pdb/derived_data/index/author.idx'],
                                              stderr=subprocess.STDOUT)

        except subprocess.CalledProcessError as exc:
            print("Status : FAIL", PDB_id, exc.returncode, exc.output)

        with open('./author.idx', 'r') as f:
            ids = [pattern.findall(x) for x in f.readlines()]

    else:
        ids = glob(directory)

    print('Preparing to process {} PDB(s).'.format(len(ids)))

    if n_jobs < 0:
        # use all cpus
        import psutil
        n_jobs = psutil.cpu_count() + n_jobs + 1
    #elif n_jobs < -1:
    #    raise ValueError("Invalid value for n_jobs.")
    manager = Manager()
    ca_dist = manager.dict()
    completed = manager.Value('i', 0)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:

        print('Spawning processes...')

        if strategy == 'individual':
            jobs = split_list(ids, len(ids))
        elif strategy == '2*jobs':
            jobs = split_list(ids, n_jobs*2)

        # Create processes
        if directory is None:
            # using PDBs downloaded from rcsb
            futures = {executor.submit(worker_group, ca_dist, completed, j, None) for j in jobs}
        else:
            # using local PDB files
            futures = {executor.submit(worker_group, ca_dist, completed, None, j) for j in jobs}

        print('Spawned {} processes'.format(len(futures)))

        pbar = tqdm(total=len(ids))
        while completed.value < len(ids):
            pbar.update(completed.value - pbar.n)

    with open('salt_bridge_result.json', 'w') as fp:
        json.dump(ca_dist.copy(), fp)


if __name__ == '__main__':

    # disable warnings
    warnings.simplefilter("ignore")

    main(0.4, '/acrm/data/pdb/*.ent', n_jobs=-5)

    print('DONE!')
