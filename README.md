# UMAP + Clustering con Optuna (Silhouette)


puede correr este contenedor con Podman o Docker:
git clone el repo en el podman 
podman build . -t {Un nombre que quiera del contenedor}






Este proyecto documenta nuestro proceso para la creación de una herramienta donde queremos **darle labels a textos grandes con control de granularidad**.  
Está pensado para personas involucradas en **toma de decisiones basada en datos**, que buscan descubrir temas, categorías o patrones dentro de textos extensos sin necesidad de etiquetas previas.

El proyecto usa un dataset de prueba y lo **vectoriza con Sentence-Transformers**, luego bajamos a **2D con UMAP**, aplicamos **clustering con AgglomerativeClustering**, y finalmente **optimizamos los hiperparámetros con Optuna**.  
Los logs de las ejecuciones que hicimos para el proceso de Optuna están en:  
`Optuna Logs.txt`.

Después de obtener los mejores clusters, generamos el **dendrograma**, pero faltan los **labels en las intersecciones**.  
Esos labels se los pedimos a un **LLM local**, creado con **Ollama**, llamado `cal-state-words`.

> Para correr la parte del notebook donde usamos el LLM para generar labels de las intersecciones del dendrograma necesitas tener **Ollama** instalado y cualquier modelo compatible.  
> Nosotros usamos `cal-state-words`, pero puedes usar el que quieras (por ejemplo, `gemma3:1b`).  
> Solo asegúrate de tener Ollama corriendo localmente.
>
> ```bash
> pip install ollama
> ollama pull gemma3:1b
> ```

---

### Archivos relevantes
- Notebook principal: `Optuna_SilhoutteScore_C7 (1).ipynb`  
- Logs de los estudios: `Optuna Logs.txt`  
- Dataset de ejemplo: `palabras_50_sin_acentos.csv`  
- Archivo de dependencias: `requirements.txt`  
- README

---

## Crear y activar el environment (Windows)

Para garantizar que el proyecto se ejecute igual en cualquier computadora, usamos un **entorno virtual (venv)** junto con el `requirements.txt`.

```powershell
# 1) Crear y activar entorno fijando python 3.10
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Actualizar pip e instalar dependencias del requirements.txt
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Registrar kernel para Jupyter
python -m ipykernel install --user --name calstate-umap --display-name "CalState UMAP (venv)"

# 4) Abrir Jupyter Lab o en Notebook 
jupyter lab
# o
jupyter notebook
