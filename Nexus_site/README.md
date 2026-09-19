# Nexus_site

Local web interface for the Nexus TER project.

This prototype brings together the main components of the project:

- NexusLM PubMed Big, the language model built from scratch;
- a local RAG module to retrieve relevant passages from uploaded documents;
- a BiomedBERT NER module to detect biomedical entities;
- Ollama to generate a more readable local answer from the retrieved context.

The application runs locally with Flask. Uploaded documents stay on the user's machine.

## 1. Requirements

Before running the project, install:

- Python 3.10 or 3.11;
- pip;
- Ollama, only if you want to test the final local answer generation;
- the Ollama model `llama3.2`.

Python 3.14 is not recommended for this project because some packages, such as TensorFlow, h5py or torch, may not be available or fully compatible.

## 2. Project Structure

```text
Nexus_site/
|-- app.py
|-- README.md
|-- requirements.txt
|-- templates/
|   `-- index.html
|-- static/
|   |-- css/
|   |   `-- app.css
|   |-- js/
|   |   `-- app.js
|   `-- img/
|       |-- nexus_jellyfish_favicon.png
|       `-- nexus_jellyfish_logo.png
`-- models/
    |-- nexus_pubmed_big/
    |   |-- config.json
    |   |-- metadata.json
    |   |-- model.weights.h5
    |   `-- vocab_pubmed_big.pkl
    `-- biomedbert_ner_bc5cdr/
        |-- config.json
        |-- model.safetensors
        |-- special_tokens_map.json
        |-- tokenizer.json
        `-- tokenizer_config.json
```

## 3. Python Setup

Open a terminal in the `Nexus_site` folder.

### How to Find the Folder Path on Windows

If you do not know the exact path to the project folder:

1. Open the folder `Nexus_site` in Windows File Explorer.
2. Click in the address bar at the top of the window.
3. Copy the full path that appears.
4. In PowerShell, type `cd`, add a space, then paste the path between quotes.

Example on Windows:

```powershell
cd "C:\path\to\Nexus_site"
```

Another simple method is to type `cd ` in PowerShell, then drag and drop the `Nexus_site` folder into the terminal window. PowerShell will automatically insert the correct path.

```powershell
cd "path\to\Nexus_site"
```

On macOS or Linux, open a terminal and use:

```bash
cd /path/to/Nexus_site
```

You can also type `cd `, drag and drop the `Nexus_site` folder into the terminal, then press Enter.

## 4. Install Python Dependencies

### Windows PowerShell

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\activate
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

### macOS / Linux

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

If `python3` is the only available command on your machine, use `python3 -m pip` instead of `python -m pip`.

## 5. Ollama Setup

Install Ollama, then download the model used by the application:

```powershell
ollama pull llama3.2
```

Start Ollama:

Windows PowerShell:

```powershell
ollama serve
```

macOS / Linux:

```bash
ollama serve
```

If Ollama is already running in the background, this command may say that the port is already in use. In that case, you can continue.

## 6. Run the Application

From the `Nexus_site` folder, run:

Windows:

```powershell
python app.py
```

macOS / Linux:

```bash
python app.py
```

or, depending on your installation:

```bash
python3 app.py
```

Then open this address in a browser:

```text
http://127.0.0.1:5000
```

## 7. Quick Test

### Test NexusLM only

1. Disable `RAG`, `NER` and `Ollama` in the interface.
2. Enter an English prompt, for example:

```text
the patient received
```

This shows the small NexusLM model generating text token by token from its PubMed training.

### Test the full document prototype

1. Upload a biomedical PDF or text file.
2. Enable `RAG`, `NER` and `Ollama`.
3. Ask a question in English, for example:

```text
What is metformin used for?
```

The RAG panel shows the retrieved passages, the NER panel shows detected biomedical entities, and Ollama produces a local answer based on the retrieved context.

## 8. Main Files

`app.py` contains the Flask back-end. It loads NexusLM, the vocabulary, the RAG module, the NER module and the API routes.

`templates/index.html` contains the HTML structure of the web page.

`static/css/app.css` contains the visual styling.

`static/js/app.js` contains the browser-side interactions.

`models/nexus_pubmed_big` contains the NexusLM PubMed Big model.

`models/biomedbert_ner_bc5cdr` contains the biomedical NER model.

## 9. Common Issues

### ModuleNotFoundError

If Python reports a missing package, make sure the virtual environment is activated, then run:

```powershell
python -m pip install -r requirements.txt
```

### h5py, TensorFlow or torch installation issue

Check the Python version:

```powershell
python --version
```

Use Python 3.10 or 3.11 if possible.

### Ollama does not answer

Check that the model is installed:

```powershell
ollama list
```

Check that Ollama is running:

```powershell
ollama serve
```

### Port 5000 already in use

Stop the previous Flask server with `Ctrl + C`, then run:

```powershell
python app.py
```

## 10. Important Note

Nexus is an experimental prototype. It is not a validated medical tool and must not be used for diagnosis or medical decision-making. The answers should be understood as a technical demonstration of NexusLM, local RAG, biomedical NER and Ollama running on a local machine.
