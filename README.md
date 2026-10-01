# NEXUS

Projet NEXUS : notebooks d'analyse et interface web locale (`Nexus_site`).

## Contenu

- `Nexus.ipynb`, `Interface.ipynb`, `Analyse_questionnaire_Nexus.ipynb`, `Nexus_Extensions_NER_RAG.ipynb` : notebooks du projet.
- `Nexus_site/` : application Flask (interface locale RAG + NER + Ollama).

## Modèles (hébergés sur Hugging Face)

Les modèles sont trop volumineux pour GitHub (> 100 Mo), ils sont donc hébergés sur le Hub :
**https://huggingface.co/salmatag/nexus-models**

Téléchargement automatique au bon emplacement :

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="salmatag/nexus-models",
    local_dir="Nexus_site/models",
)
```

Contenu : `biomedbert_ner_bc5cdr` (NER biomédical) et `nexus_pubmed_big` (NexusLM).

## Installation

Voir `Nexus_site/README.md`.
