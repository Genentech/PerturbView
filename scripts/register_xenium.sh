#!/bin/bash
#SBATCH --cpus-per-task 4
#SBATCH -p defq
#SBATCH -t 0-02:00
#SBATCH --qos=short
#SBATCH --mem=128G

#SBATCH --chdir=/gnet/is1/p01/shares/ctg-microscopy/EL/perturbview_test/
#SBATCH --mail-type=END
#SBATCH --mail-user=lubecke@gene.com
#SBATCH -o wsireg.out
#SBATCH -e wsireg.err

source perturbview_tissue activate
cd $1

python ~/perturbview_tissue/scripts/register_xenium.py $1 $2 -o $3
