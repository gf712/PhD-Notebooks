#!usr/bin/env python3

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

from os.path import basename
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(usage="""{} """.
                                     format(sys.argv[0]),
                                     epilog="Calculate the energy of a PDB file structure using "
                                            "any forcefield provided by openmm.")

    parser.add_argument("-i", "--input", type=str,
                        help="""PDB file""")

    parser.add_argument("-f", "--forcefield", type=str,
                        help="""Forcefield for parameterisation of PDB structure""")

    return parser.parse_args()


def main(args):

    """
    Function to calculate potential energy of PDB file using openmm.
    This program is needed due to a (probable) memory leak in the addHydrogen method

    :param args: input
                 forcefield
    :return:
    """

    # parse PDB
    pdb = PDBFile(args.input)

    # load forcefield
    forcefield = ForceField(args.forcefield)

    # create model with PDB
    modeller = Modeller(pdb.topology, pdb.positions)

    # add hydrogens
    modeller.addHydrogens(forcefield)

    # parameterise structure
    system = forcefield.createSystem(modeller.topology)

    # this doesn't matter since we don't perform an actual simulation
    integrator = VerletIntegrator(0.001 * picoseconds)

    # create simulation object
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # get the energy of the system in the first (and only) snapshot
    state = simulation.context.getState(getEnergy=True)

    # retrieve potential energy
    potential = state.getPotentialEnergy()

    name = basename(args.input)

    # write out result (to catch with other programs)
    sys.stdout.write('{} {:.2f} {}'.format(name, potential._value, potential.unit))


if __name__ == "__main__":

    args = parse_args()

    main(args)
