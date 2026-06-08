#!/bin/bash
#SBATCH --job-name=build_dalia
#SBATCH --output=../logs/build_%j.out
#SBATCH --error=../logs/build_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread

# 1. Charger l'environnement
source ~/.bashrc

echo "================================================="
echo "Démarrage de la construction du conteneur Apptainer"
echo "Nœud de calcul : $SLURMD_NODENAME"
echo "Date : $(date)"
echo "================================================="

cd /lustre/work/${PROJECT_GROUP}/${USER}/stage/code/uncertainty-DLM/

# ========================================================
# FIX CRUCIAL : Désactiver les montages hôtes pendant le build
# ========================================================
# On vide temporairement ces variables pour que fakeroot ne plante pas
export APPTAINER_BIND=""
export APPTAINER_MOUNTS=""
export SINGULARITY_BIND=""

# 2. Construction du conteneur avec droits "fakeroot"
apptainer build --fakeroot my_project.sif dalia_env.def

echo "================================================="
echo "Construction terminée. Extraction du fichier lock..."
echo "================================================="

# ========================================================
# RESTAURATION : Réactiver le montage pour l'extraction
# ========================================================
# On remet le bind pour que le conteneur puisse écrire le lockfile dans Lustre
export APPTAINER_BIND="/lustre/work"

# 3. Extraction automatique des dépendances gelées
if [ -f "my_project.sif" ]; then
    apptainer exec my_project.sif cat /opt/requirements_arm64_freeze.txt > requirements_arm64_freeze.txt
    echo "Fichier requirements_arm64_freeze.txt généré avec succès !"
else
    echo "ERREUR : Le fichier my_project.sif n'a pas été généré."
fi

echo "Fin du job : $(date)"
