#!/bin/bash
#SBATCH --cpus-per-task 4
#SBATCH -p defq
#SBATCH -t 0-02:00
#SBATCH --qos=short
#SBATCH --mem=64G

#SBATCH --chdir=/gnet/is1/p01/shares/ctg-microscopy/EL/perturbview_test/
#SBATCH --mail-type=END
#SBATCH --mail-user=lubecke@gene.com
#SBATCH -o wsireg.out
#SBATCH -e wsireg.err

source perturbview_tissue activate
python ~/perturbview_tissue/scripts/call_bases.py $1 $2 -o $3 -s $4 -r $5 -d $6 -t /gstore/scratch/u/lubecke
