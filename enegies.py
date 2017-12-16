#!usr/bin/env python3

from glob import glob
from os.path import basename
from natsort import natsorted
from subprocess import run, PIPE


def main():

    energies = {}

    for file in natsorted(glob('models/*.pdb')):

        # solves memory leak -> OS garbage collector is called after each energy calc
        # no need to find what is happening
        # issue seems to be in addHydrogens method of modeller
        output = run(["python", "calc_energy.py", "-i", file, "-f", 'amber10.xm'], stdout=PIPE)

        energies[basename(file)] = output.stdout.decode()

        print(energies[basename(file)])

    with open('all_energies_amber_10.csv', 'w') as f:
        f.write('file,energy,unit\n')
        for file in natsorted(glob('models/*.pdb')):
            f.write('{},{},{}\n'.format(basename(file), *energies[basename(file)].split()[1:]))


if __name__ == '__main__':
    main()
