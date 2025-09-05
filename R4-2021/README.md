# R4 2021 - Introduction au Reservoir Computing

## Installation

Ce tutoriel utilise une version en développement de [Reservoirpy, disponible ici.](https://github.com/reservoirpy/reservoirpy/tree/v0.2.5).

Téléchargez cette version depuis GitHub, puis installez là à l'aide de la commande:

```bash
pip install -e /racine/du/projet/reservoirpy
```

Installez ensuite les dépendences contenues dans le fichier `requirements.txt`;

```bash
pip install -r requirements.txt
```

Lancez ensuite Jupyter dans le dossier contenant le notebook:

```bash
jupyter notebook
```

Ouvrez le notebook depuis la page ouverte sur votre navigateur. Si aucune page ne s'est ouverte sur votre navigateur, ouvrez un des les liens indiqués par le résultat de la commande `jupyter notebook`.

## Données

Les 1 à 3 utilisent des jeux de données inclus dans ReservoirPy. Les chapitres 4 et 5 nécessitent des données externes, disponibles à la demande.

Les données sur la prédiction de chute du robot (du chapitre 4) sont [disponibles ici](https://doi.org/10.5281/zenodo.5900966): 
- Grégoire Passault. (2022). Humanoid push recovery logs (simulation) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5900966
