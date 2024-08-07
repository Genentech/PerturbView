#!/bin/bash
#SBATCH --cpus-per-task 12
#SBATCH -p defq
#SBATCH -t 0-02:00
#SBATCH --qos=short
#SBATCH --mem=16G

#SBATCH --chdir=/gnet/is1/p01/shares/ctg-microscopy/EL/perturbview_test/
#SBATCH --mail-type=END
#SBATCH --mail-user=lubecke@gene.com
#SBATCH -o wsireg.out
#SBATCH -e wsireg.err

source $HOME/.bashrc
conda activate perturbview_tissue

python $HOME/perturbview_tissue/scripts/stitch_images.py $1 -f $2 -o $3
