# PFIA 2021 - Tutoriel *Reservoir Computing : théorie, intuitions et applications avec ReservoirPy*

Lundi 28 juin 2021, Bordeaux


## Avant de commencer

Téléchargez l'ensemble de ce repository sur votre ordinateur.

Si vous disposez de Python 3.6 à 3.8 sur votre ordinateur:

### 1. Créez un nouvel environnement virtuel, et installez les dépendances

A l'aide de `conda`, `virtualenv` ou `pipenv`, créez un nouvel environnement virtuel à la racine du dossier du tutoriel. Installez les dépendances incluses dans le fichier `requirements.txt`. Par exemple, en utilisant `pip`:

```bash
pip install -r ./requirements.txt
```

Assurez vous d'avoir bien installé:
- reservoirpy==0.2.4
- jupyter==1.0.0
- matplotlib==3.4.2
- pandas==1.2.4
- scikit-learn==0.24.2
- librosa==0.8.1

### 2. Ouvrez le notebook

Nous recommandons l'utilisation de Jupyter. Dans un terminal (ou une console Conda) à la racine du dossier, avec l'environnement virtuel contenant Jupyter:

```bash
jupyter notebook
```

Une nouvelle fenêtre s'ouvre alors dans votre navigateur. Ouvrez le notebook `PFIA_2021_RC_Tutorial.ipynb`.

### 3. Ouvrez les slides

Si vous ne pouvez pas/ne souhaitez pas suivre le tutoriel à l'aide du notebook, nous mettons à votre disposition les slides de cette présentation. Ces slides sont une représentation épurée du contenu du notebook.

En vous assurant que le fichier `PFIA_2021_RC_Tutorial.slides.html` est bien dans le même dossier que le dosser `/static`, ouvrez le ficher. Une nouvelle fenêtre s'ouvre dans votre navigateur. 

Nous recommandons d'activer Javascript dans le navigateur pour un expérience optimale.


## Données de chant de canaris pour le chapitre 4 du tutoriel

Si vous le souhaitez, les données de chant de canari utilisées dans le chapitre 4 du tutoriel sont disponibles sur [Zenodo, en cliquant sur ce lien](https://zenodo.org/record/4736597).
Placez les dossiers `audio` et `annotations` obtenus dans un dossier `canary-data` à la racine du dossier du tutoriel.

**Attention, ces données sont assez lourdes ! (plusieurs Go).**

