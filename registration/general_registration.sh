#!/bin/bash
#SBATCH -t 0-2:00
#SBATCH --qos=short
#SBATCH --mem=200G
#SBATCH -c 4
#SBATCH -N 1-1
#SBATCH -p defq

#SBATCH --chdir=/gstore/scratch/u/lubecke
#SBATCH --mail-type=END
#SBATCH --mail-user=lubecke@gene.com
#SBATCH -o wsireg.out
#SBATCH -e wsireg.err

source perturbview_tissue activate
cd $1

python ~/perturbview_tissue/registration_cdf_to_HE.py $2 $3 $4 -o $5
