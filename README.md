# 📊 A Human–AI Collaborative Framework for Open Coding in Thematic Analysis

## Project Description

> The objective is to create an `Augmented AI Tool` for people who is in charge of data driven desitions, this way instead of reading all of the transcripts in an interview, or all of the comments down the business page in google maps he uses this tool and see the bad and the good 

>We aim to use AI to augment the human capacity in the decision making progress not to replace it.

---

## 📂 Repository Structure
- `data/`: Raw and processed datasets
- - `Optuna Logs.txt`: Here we keep the logs of about 30 runs on optuna we did to see what are the best Hyperparameters on the models
- - `palabras_50_sin_acentos.csv`: This is our dataset we use for the Dendogram-notebook
- `notebooks/`: Jupyter Notebooks for data validation and analysis
- - `Dendogram-notebook.ipynb`: This is the initial notebook, in here we use **Sentence-Transformers** to vectorize words, than we use **UMAP** to pass them into a 2D space, then we apply **Agglomerative Clustering**, we use **Optuna** to optimize hyperparameters, we then make the dendogram and we use an local LLM in **Ollama** to ask for the labels in the intersections. 
- `Containerfile`: This is a containerfile for python and libraries.
- `Modelfile`: This is a file that stores the LLM we created **cal-state_words** (only answers with 1 word to any prompt).
- `Ollama.Dockerfile`: This is another containerfile for ollama and creating the cal-state_words LLM using the Modelfile
- `docker-compose.yml`: Template to compose both containerfiles
- `requirements.txt`: File with all the python libraries needed
- `README.md`: This file


---

##  Installation Guide

###  Requirements
- Docker or Podman (or any WSL to run the container)
- All dependencies listed in `requirements.txt`

### 📦 Setup Instructions

1. **Clone the Repository**
   ```bas
   git clone https://github.com/HiramZ04/Cal-State-UMAP_Clustering.git
   cd <your-repo-folder>
2. **Build the Container using Docker or Podman**

   ```bash
   # Using Docker:
   docker compose build
   docker compose up -d

   # To Stop:
   docker compose down
   ```

   ```bash
   # Using Podman:
   podman compose build
   podman compose up -d
   
   # To Stop:
   podman compose down
   ```


## Contributors

We want to thank the following individuals who have contributed to this project:

| Name | GitHub Username |
|---|---|
| Jesus A. Beltran - Advisor | [3eltran23](https://github.com/3eltran23) |
| Hiram Zuniga | [HiramZ04](https://github.com/HiramZ04)  |
| Sebastian Soto | [SebastianSoto17](https://github.com/SebastianSoto17)  |
| Diego Hernandez | [wushang549](https://github.com/wushang549)  |

