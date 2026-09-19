from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
import urllib.error
import urllib.request
import h5py
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as exc:
    TfidfVectorizer = None
    cosine_similarity = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


try:
    import tensorflow as tf
    from tensorflow.keras import layers
except Exception as exc:
    tf = None
    layers = None
    TENSORFLOW_IMPORT_ERROR = exc
else:
    TENSORFLOW_IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent
NEXUS_MODEL_DIR = BASE_DIR / "models" / "nexus_pubmed_big"


MODEL_KERAS_PATH = NEXUS_MODEL_DIR / "model_nexus_pubmed.keras"


CONFIG_PATH = NEXUS_MODEL_DIR / "config.json"
WEIGHTS_PATH = NEXUS_MODEL_DIR / "model.weights.h5"


VOCAB_PATH = NEXUS_MODEL_DIR / "vocab_pubmed_big.pkl"


DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_MAX_TOKENS = 25
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_K = 3
UNK_ID = 1
PAD_ID = 0


OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))


if tf is not None:

    @tf.keras.utils.register_keras_serializable(package="Custom")
    class PositionalEmbedding(layers.Layer):

        def __init__(self, sequence_length: int, vocab_size: int, embed_dim: int, **kwargs):
            super().__init__(**kwargs)
            self.sequence_length = int(sequence_length)
            self.vocab_size = int(vocab_size)
            self.embed_dim = int(embed_dim)
            self.token_embeddings = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embed_dim,
                name="token_embeddings",
            )
            self.position_embeddings = layers.Embedding(
                input_dim=self.sequence_length,
                output_dim=self.embed_dim,
                name="position_embeddings",
            )

        def call(self, inputs):
            length = tf.shape(inputs)[-1]
            positions = tf.range(start=0, limit=length, delta=1)
            return self.token_embeddings(inputs) + self.position_embeddings(positions)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "sequence_length": self.sequence_length,
                    "vocab_size": self.vocab_size,
                    "embed_dim": self.embed_dim,
                }
            )
            return config


    @tf.keras.utils.register_keras_serializable(package="Custom")
    class MultiHeadAttentionCustom(layers.Layer):

        def __init__(self, embed_dim: int, num_heads: int, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = int(embed_dim)
            self.num_heads = int(num_heads)
            if self.embed_dim % self.num_heads != 0:
                raise ValueError("embed_dim doit être divisible par num_heads.")
            self.projection_dim = self.embed_dim // self.num_heads

            self.Wq = layers.Dense(self.embed_dim, use_bias=False, name="Wq")
            self.Wk = layers.Dense(self.embed_dim, use_bias=False, name="Wk")
            self.Wv = layers.Dense(self.embed_dim, use_bias=False, name="Wv")
            self.Wo = layers.Dense(self.embed_dim, use_bias=False, name="Wo")

        def _separate_heads(self, x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            q = self._separate_heads(self.Wq(inputs), batch_size)
            k = self._separate_heads(self.Wk(inputs), batch_size)
            v = self._separate_heads(self.Wv(inputs), batch_size)

            scores = tf.matmul(q, k, transpose_b=True)
            scores = scores / tf.math.sqrt(tf.cast(self.projection_dim, tf.float32))


            seq_len = tf.shape(inputs)[1]
            causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            causal_mask = tf.reshape(causal_mask, (1, 1, seq_len, seq_len))
            scores = scores + (1.0 - causal_mask) * -1e9

            weights = tf.nn.softmax(scores, axis=-1)
            attention = tf.matmul(weights, v)
            attention = tf.transpose(attention, perm=[0, 2, 1, 3])
            concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
            return self.Wo(concat_attention)

        def get_config(self):
            config = super().get_config()
            config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
            return config


    @tf.keras.utils.register_keras_serializable(package="Custom")
    class TransformerDecoderBlock(layers.Layer):

        def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = int(embed_dim)
            self.num_heads = int(num_heads)
            self.ff_dim = int(ff_dim)

            self.att = MultiHeadAttentionCustom(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                name="att",
            )
            self.ffn = tf.keras.Sequential(
                [
                    layers.Dense(self.ff_dim, activation="relu"),
                    layers.Dense(self.embed_dim),
                ],
                name="ffn",
            )
            self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
            self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
            self.dropout1 = layers.Dropout(0.1, name="dropout1")
            self.dropout2 = layers.Dropout(0.1, name="dropout2")

        def call(self, inputs, training=False):
            attn_output = self.att(inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.norm1(inputs + attn_output)

            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.norm2(out1 + ffn_output)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "embed_dim": self.embed_dim,
                    "num_heads": self.num_heads,
                    "ff_dim": self.ff_dim,
                }
            )
            return config


    @tf.keras.utils.register_keras_serializable(package="Custom")
    class NexusLM(tf.keras.Model):

        def __init__(
            self,
            sequence_length: int,
            vocab_size: int,
            embed_dim: int,
            num_heads: int,
            ff_dim: int,
            num_layers: int,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.sequence_length = int(sequence_length)
            self.vocab_size = int(vocab_size)
            self.embed_dim = int(embed_dim)
            self.num_heads = int(num_heads)
            self.ff_dim = int(ff_dim)
            self.num_layers = int(num_layers)

            self.pos_embedding = PositionalEmbedding(
                sequence_length=self.sequence_length,
                vocab_size=self.vocab_size,
                embed_dim=self.embed_dim,
                name="positional_embedding",
            )
            self.decoder_blocks = [
                TransformerDecoderBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.ff_dim,
                    name=("transformer_decoder_block" if i == 0 else f"transformer_decoder_block_{i}"),
                )
                for i in range(self.num_layers)
            ]
            self.layernorm = layers.LayerNormalization(epsilon=1e-6, name="layernorm")
            self.out = layers.Dense(self.vocab_size, name="dense")

        def call(self, inputs, training=False):
            x = self.pos_embedding(inputs)
            for block in self.decoder_blocks:
                x = block(x, training=training)
            x = self.layernorm(x)
            return self.out(x)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "sequence_length": self.sequence_length,
                    "vocab_size": self.vocab_size,
                    "embed_dim": self.embed_dim,
                    "num_heads": self.num_heads,
                    "ff_dim": self.ff_dim,
                    "num_layers": self.num_layers,
                }
            )
            return config


app = Flask(__name__)

MODEL = None
W2I: Dict[str, int] = {}
I2W: Dict[int, str] = {}
MODEL_INFO: Dict[str, Any] = {}
LOAD_ERROR = ""


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z']+", text)


def normalize_i2w(i2w: Dict[Any, Any]) -> Dict[int, str]:
    return {int(k): str(v) for k, v in i2w.items()}


def load_vocab(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Vocabulaire introuvable : {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, tuple) and len(data) >= 2:
        w2i, i2w = data[0], data[1]
    elif isinstance(data, dict) and "w2i" in data and "i2w" in data:
        w2i, i2w = data["w2i"], data["i2w"]
    elif isinstance(data, dict) and "word_to_index" in data and "index_to_word" in data:
        w2i, i2w = data["word_to_index"], data["index_to_word"]
    else:
        raise ValueError(
            "Format de vocabulaire non reconnu. Attendu : (w2i, i2w) ou {'w2i': ..., 'i2w': ...}."
        )

    return dict(w2i), normalize_i2w(i2w)


def read_vocab_file(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    return load_vocab(path)


def detect_config_from_weights(weights_path: Path) -> Dict[str, Any]:
    if not weights_path.exists():
        return {}

    with h5py.File(weights_path, "r") as f:
        token_shape = f["layers/nexus_lm/layers/positional_embedding/token_embeddings/vars/0"].shape
        pos_shape = f["layers/nexus_lm/layers/positional_embedding/position_embeddings/vars/0"].shape
        ffn_shape = f["layers/nexus_lm/decoder_blocks/transformer_decoder_block/ffn/layers/dense/vars/0"].shape

        decoder_root = f["layers/nexus_lm/decoder_blocks"]
        num_layers = sum(1 for key in decoder_root.keys() if key.startswith("transformer_decoder_block"))

    vocab_size, embed_dim = int(token_shape[0]), int(token_shape[1])
    sequence_length = int(pos_shape[0])
    ff_dim = int(ffn_shape[1])


    num_heads = 4 if embed_dim % 4 == 0 else 1

    return {
        "sequence_length": sequence_length,
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "ff_dim": ff_dim,
        "num_layers": int(num_layers),
    }


def find_matching_vocab(expected_vocab_size: int) -> Tuple[Path, Dict[str, int], Dict[int, str]]:
    candidates = [VOCAB_PATH] + sorted(NEXUS_MODEL_DIR.glob("vocab*.pkl"))
    seen = set()
    for path in candidates:
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        try:
            w2i, i2w = read_vocab_file(path)
        except Exception:
            continue
        if len(w2i) == int(expected_vocab_size):
            return path, w2i, i2w


    w2i, i2w = read_vocab_file(VOCAB_PATH)
    return VOCAB_PATH, w2i, i2w


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-6)
    logits = logits.astype(np.float64)
    logits = logits / temperature
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def clean_probabilities(probs: np.ndarray, encoded_context: List[int], repetition_penalty: float = 1.18) -> np.ndarray:
    probs = probs.astype(np.float64).copy()

    for bad_id in (PAD_ID, UNK_ID):
        if 0 <= bad_id < len(probs):
            probs[bad_id] = 0.0


    for idx, token in I2W.items():
        if 0 <= idx < len(probs) and token.startswith("<") and token.endswith(">"):
            probs[idx] = 0.0

    recent_ids = encoded_context[-8:]
    for token_id in recent_ids:
        if 0 <= token_id < len(probs):
            probs[token_id] = probs[token_id] / repetition_penalty

    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / total
    return probs


def top_k_sample(probs: np.ndarray, top_k: int) -> Tuple[int, List[Dict[str, Any]]]:
    top_k = max(1, min(int(top_k), len(probs)))
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]
    top_probs = top_probs / np.sum(top_probs)

    chosen_id = int(np.random.choice(top_indices, p=top_probs))

    alternatives = []
    for idx in top_indices:
        idx_int = int(idx)
        alternatives.append(
            {
                "id": idx_int,
                "token": I2W.get(idx_int, f"<id:{idx_int}>"),
                "prob": float(probs[idx_int]),
                "chosen": idx_int == chosen_id,
            }
        )
    return chosen_id, alternatives


def read_generation_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {
            "sequence_length": DEFAULT_SEQUENCE_LENGTH,
            "vocab_size": len(W2I) if W2I else 30000,
            "embed_dim": 128,
            "num_heads": 4,
            "ff_dim": 256,
            "num_layers": 2,
        }

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)


    for layer in data.get("config", {}).get("layers", []):
        layer_config = layer.get("config", {})
        if layer.get("class_name") == "NexusLM" or layer_config.get("vocab_size"):
            return {
                "sequence_length": int(layer_config.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)),
                "vocab_size": int(layer_config.get("vocab_size", len(W2I) or 30000)),
                "embed_dim": int(layer_config.get("embed_dim", 128)),
                "num_heads": int(layer_config.get("num_heads", 4)),
                "ff_dim": int(layer_config.get("ff_dim", 256)),
                "num_layers": int(layer_config.get("num_layers", 2)),
            }

    return {
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "vocab_size": len(W2I) if W2I else 30000,
        "embed_dim": 128,
        "num_heads": 4,
        "ff_dim": 256,
        "num_layers": 2,
    }


def build_model_from_config(model_config: Dict[str, Any]):
    if tf is None:
        raise RuntimeError(f"TensorFlow n'est pas disponible : {TENSORFLOW_IMPORT_ERROR}")

    sequence_length = int(model_config["sequence_length"])
    inputs = tf.keras.Input(shape=(sequence_length,), dtype="int32", name="input_layer_3")
    nexus = NexusLM(
        sequence_length=sequence_length,
        vocab_size=int(model_config["vocab_size"]),
        embed_dim=int(model_config["embed_dim"]),
        num_heads=int(model_config["num_heads"]),
        ff_dim=int(model_config["ff_dim"]),
        num_layers=int(model_config["num_layers"]),
        name="nexus_lm",
    )
    logits = nexus(inputs)
    last_token_logits = logits[:, -1, :]
    model = tf.keras.Model(inputs=inputs, outputs=last_token_logits, name="Nexus_PubMed")


    _ = model(np.zeros((1, sequence_length), dtype=np.int32), training=False)
    return model


def load_nexus() -> Dict[str, Any]:
    global MODEL, W2I, I2W, MODEL_INFO, LOAD_ERROR

    LOAD_ERROR = ""


    config_from_weights = detect_config_from_weights(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else {}
    config = config_from_weights or read_generation_config()


    vocab_path_used, W2I, I2W = find_matching_vocab(int(config["vocab_size"]))

    if MODEL_KERAS_PATH.exists():

        custom_objects = {
            "NexusLM": NexusLM,
            "PositionalEmbedding": PositionalEmbedding,
            "TransformerDecoderBlock": TransformerDecoderBlock,
            "MultiHeadAttentionCustom": MultiHeadAttentionCustom,
        }
        MODEL = tf.keras.models.load_model(MODEL_KERAS_PATH, custom_objects=custom_objects, compile=False)
        source = str(MODEL_KERAS_PATH.name)
    else:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"config.json introuvable : {CONFIG_PATH}")
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"model.weights.h5 introuvable : {WEIGHTS_PATH}")

        MODEL = build_model_from_config(config)
        MODEL.load_weights(str(WEIGHTS_PATH))
        source = f"{CONFIG_PATH.name} + {WEIGHTS_PATH.name}"

    MODEL_INFO = {
        "loaded": True,
        "source": source,
        "vocab_path": str(vocab_path_used.name),
        "vocab_size_file": len(W2I),
        **config,
    }
    return MODEL_INFO


def ensure_model_loaded() -> None:
    global LOAD_ERROR
    if MODEL is not None:
        return
    try:
        load_nexus()
    except Exception as exc:
        LOAD_ERROR = str(exc)
        raise


def read_uploaded_files(files) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []

    for file_storage in files:
        filename = file_storage.filename or "fichier"
        suffix = Path(filename).suffix.lower()
        raw = file_storage.read()

        text = ""
        if suffix in {".txt", ".md", ".csv", ".json", ".py", ".html", ".xml"}:
            text = raw.decode("utf-8", errors="ignore")
        elif suffix == ".pdf":
            import io

            pdf_errors = []


            try:
                from pypdf import PdfReader

                reader = PdfReader(io.BytesIO(raw))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as exc:
                pdf_errors.append(f"pypdf: {exc}")
                try:
                    from PyPDF2 import PdfReader

                    reader = PdfReader(io.BytesIO(raw))
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                except Exception as exc2:
                    pdf_errors.append(f"PyPDF2: {exc2}")
                    text = ""

            if not text.strip():
                text = (
                    "[PDF reçu, mais le texte n'a pas pu être extrait. "
                    "Installe pypdf avec `pip install pypdf`, ou utilise un PDF contenant du texte sélectionnable.]"
                )
        else:
            text = "[Fichier reçu, mais ce type n'est pas lu automatiquement.]"

        if text.strip():
            documents.append({"filename": filename, "text": text})

    return documents


def split_text_into_chunks(text: str, chunk_size: int = 900, overlap: int = 160) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def rechercher_passages_notebook(question: str, documents: List[str], top_k: int = 2) -> List[Tuple[str, float, int]]:
    if TfidfVectorizer is None or cosine_similarity is None:
        raise RuntimeError(
            "scikit-learn est nécessaire pour le RAG du notebook. "
            "Installe-le avec : pip install scikit-learn"
        )

    clean_documents = [doc for doc in documents if doc and doc.strip()]
    if not clean_documents or not question.strip():
        return []

    vectorizer = TfidfVectorizer()
    X_docs = vectorizer.fit_transform(clean_documents)
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, X_docs)[0]
    indices = scores.argsort()[::-1][:max(1, int(top_k))]

    resultats: List[Tuple[str, float, int]] = []
    for i in indices:
        resultats.append((clean_documents[int(i)], float(scores[int(i)]), int(i)))
    return resultats


def simple_rag_retrieve(query: str, documents: List[Dict[str, str]], top_n: int = 3) -> List[Dict[str, Any]]:
    passages: List[Dict[str, Any]] = []
    flat_documents: List[str] = []

    for doc in documents:
        chunks = split_text_into_chunks(doc["text"], chunk_size=850, overlap=120)
        for idx, chunk in enumerate(chunks):
            passages.append({
                "filename": doc["filename"],
                "chunk_id": idx + 1,
                "text": chunk,
            })
            flat_documents.append(chunk)

    if not passages:
        return []

    resultats = rechercher_passages_notebook(query, flat_documents, top_k=top_n)
    retrieved: List[Dict[str, Any]] = []
    for passage_text, score, flat_index in resultats:
        item = dict(passages[flat_index])
        item["score"] = score
        item["method"] = "TfidfVectorizer + cosine_similarity"
        item["text"] = passage_text
        retrieved.append(item)

    return retrieved

def build_rag_prompt(user_prompt: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return user_prompt

    context = "\n\n".join(
        f"[PASSAGE {i+1} | {chunk['filename']} | score={chunk['score']:.3f}]\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    )


    return f"""relevant biomedical context:
{context}

question: {user_prompt}
answer:"""


def extract_uploaded_text(files) -> str:
    docs = read_uploaded_files(files)
    return "\n".join(f"\n[FICHIER : {d['filename']}]\n{d['text'][:1000]}" for d in docs)


try:
    import torch
except Exception:
    torch = None

try:
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline as hf_pipeline,
    )
except Exception as exc:
    AutoModelForTokenClassification = None
    AutoTokenizer = None
    hf_pipeline = None
    TRANSFORMERS_IMPORT_ERROR = exc
else:
    TRANSFORMERS_IMPORT_ERROR = None

NER_MODEL = None
NER_LOAD_ERROR = ""
NER_BASE_MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
NER_DEFAULT_LOCAL_DIRS = [
    BASE_DIR / "models" / "biomedbert_ner_bc5cdr",
]


def resolve_ner_model_source() -> str:
    env_model = os.environ.get("BIOMEDBERT_NER_MODEL", "").strip()
    if env_model:
        return env_model

    env_dir = os.environ.get("BIOMEDBERT_NER_DIR", "").strip()
    if env_dir:
        return env_dir

    for candidate in NER_DEFAULT_LOCAL_DIRS:
        if (candidate / "config.json").exists():
            return str(candidate)


    return ""


def ensure_ner_loaded():
    global NER_MODEL, NER_LOAD_ERROR
    if NER_MODEL is not None:
        return NER_MODEL

    if hf_pipeline is None or AutoTokenizer is None or AutoModelForTokenClassification is None:
        NER_LOAD_ERROR = (
            "Le module transformers n'est pas installé. Installe-le avec : "
            "python -m pip install transformers torch"
        )
        raise RuntimeError(NER_LOAD_ERROR)

    model_source = resolve_ner_model_source()
    if not model_source:
        local_options = ", ".join(str(p.name) for p in NER_DEFAULT_LOCAL_DIRS)
        NER_LOAD_ERROR = (
            "Aucun dossier de modèle NER fine-tuné n'a été trouvé. "
            "Copie le dossier sauvegardé par le notebook BiomedBERT/BC5CDR dans le même dossier que le site, "
            "par exemple sous le nom 'biomedbert_ner_bc5cdr', ou indique son chemin avec : "
            "export BIOMEDBERT_NER_MODEL='/chemin/vers/biomedbert_ner_bc5cdr'. "
            f"Dossiers testés : {local_options}."
        )
        raise RuntimeError(NER_LOAD_ERROR)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)


        tokenizer.model_max_length = 512
        model = AutoModelForTokenClassification.from_pretrained(model_source)
        device = 0 if (torch is not None and torch.cuda.is_available()) else -1
        NER_MODEL = hf_pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device,
        )
        return NER_MODEL
    except Exception as exc:
        NER_LOAD_ERROR = str(exc)
        raise RuntimeError(
            "Impossible de charger le modèle NER BiomedBERT fine-tuné sur BC5CDR. "
            "Vérifie que le dossier contient bien les fichiers sauvegardés par save_pretrained "
            "(config.json, model.safetensors ou pytorch_model.bin, tokenizer...). "
            f"Source utilisée : {model_source}. Détail : {exc}"
        )


def split_text_for_ner(text: str, max_words: int = 280) -> List[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []


    words = clean.split()
    chunks: List[str] = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def run_biomedical_ner(text: str, max_chars: int = 4500) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []
    ner_model = ensure_ner_loaded()

    text_to_analyze = re.sub(r"\s+", " ", text).strip()[:max_chars]
    chunks = split_text_for_ner(text_to_analyze, max_words=280)

    raw_entities: List[Dict[str, Any]] = []
    for chunk in chunks:
        try:

            raw_entities.extend(ner_model(chunk, truncation=True, max_length=512))
        except TypeError:

            raw_entities.extend(ner_model(chunk))

    entities: List[Dict[str, Any]] = []
    for ent in raw_entities:
        word = str(ent.get("word", "")).strip()
        group = str(ent.get("entity_group", ent.get("entity", "ENTITE"))).strip()
        score = float(ent.get("score", 0.0))
        if not word:
            continue
        group = group.replace("B-", "").replace("I-", "")
        entities.append({
            "word": word,
            "entity_group": group,
            "score": score,
        })
    return entities


def summarize_entities(entities: List[Dict[str, Any]], limit: int = 30) -> List[Dict[str, Any]]:
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ent in entities:
        key = (ent["word"].lower(), ent["entity_group"])
        if key not in seen:
            seen[key] = dict(ent)
            seen[key]["count"] = 1
        else:
            seen[key]["count"] += 1
            seen[key]["score"] = max(seen[key]["score"], ent["score"])
    return sorted(seen.values(), key=lambda e: e["score"], reverse=True)[:limit]


def build_ollama_prompt(user_prompt: str, retrieved_chunks: List[Dict[str, Any]], conversation_history: str = "") -> str:
    if retrieved_chunks:
        context = "\n\n".join(
            f"[Passage {i + 1} - {chunk['filename']} - score={chunk['score']:.3f}]\n{chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        )
    else:
        context = "Aucun document pertinent n'a été fourni."

    history_block = conversation_history.strip() if conversation_history.strip() else "Aucun historique."

    return f"""Tu es un assistant documentaire biomédical local.
Réponds uniquement à partir du contexte fourni quand il existe.
Si le contexte ne contient pas l'information, indique clairement que le document ne permet pas de répondre.
Tiens compte de l'historique quand la nouvelle question fait référence à une réponse précédente.
Réponds en français de manière claire, courte et structurée.

HISTORIQUE DE DISCUSSION :
{history_block}

CONTEXTE DOCUMENTAIRE :
{context}

QUESTION ACTUELLE :
{user_prompt}

RÉPONSE :
"""


def call_ollama_local(prompt: str, model_name: str = OLLAMA_MODEL) -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request_obj = urllib.request.Request(
        OLLAMA_API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request_obj, timeout=OLLAMA_TIMEOUT) as response:
            raw = response.read().decode("utf-8", errors="ignore")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Ollama ne répond pas. Lance d'abord `ollama serve`, "
            f"vérifie que le modèle `{model_name}` est installé avec `ollama list`, "
            "puis relance la génération."
        ) from exc

    elapsed = time.perf_counter() - start
    result = json.loads(raw)
    response_text = str(result.get("response", "")).strip()

    return {
        "response": response_text,
        "model": model_name,
        "elapsed_seconds": elapsed,
        "raw": result,
    }


def ollama_response_to_steps(
    response_text: str,
    prompt_token_count: int,
    elapsed_seconds: float,
    max_tokens: int,
) -> Dict[str, Any]:
    words = re.findall(r"\S+", response_text)
    if max_tokens > 0:
        words = words[:max_tokens]

    steps: List[Dict[str, Any]] = []
    generated: List[str] = []
    total = max(1, len(words))

    for index, word in enumerate(words, start=1):
        generated.append(word)
        progress = index / total * 100
        avg_speed = index / elapsed_seconds if elapsed_seconds > 0 else 0.0
        steps.append({
            "index": index,
            "token_id": None,
            "token": word,
            "alternatives": [
                {
                    "id": None,
                    "token": word,
                    "prob": 1.0,
                    "chosen": True,
                }
            ],
            "generated_text": " ".join(generated),
            "metrics": {
                "prompt_tokens": prompt_token_count,
                "shown_tokens": index,
                "remaining_tokens": max(0, total - index),
                "current_context_size": prompt_token_count,
                "progress": progress,
                "elapsed_seconds": elapsed_seconds,
                "tokens_per_second": avg_speed,
                "temperature": 0.2,
                "top_k": 0,
            },
        })

    return {
        "prompt_tokens": prompt_token_count,
        "tokens": words,
        "steps": steps,
        "final_text": response_text,
        "stats": {
            "total_time": elapsed_seconds,
            "tokens_generated": len(words),
            "tokens_per_second": (len(words) / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
            "sequence_length": prompt_token_count,
            "model_info": {
                "loaded": True,
                "source": f"Ollama local ({OLLAMA_MODEL})",
                "vocab_path": "non utilisé",
                "vocab_size_file": "non utilisé",
            },
        },
    }


def encode_prompt(prompt: str) -> List[int]:
    tokens = tokenize(prompt)
    return [int(W2I.get(tok, UNK_ID)) for tok in tokens]


def make_input_sequence(encoded: List[int], sequence_length: int) -> np.ndarray:
    seq = encoded[-sequence_length:]
    if len(seq) < sequence_length:
        seq = [PAD_ID] * (sequence_length - len(seq)) + seq
    return np.array([seq], dtype=np.int32)


def generate_steps(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
) -> Dict[str, Any]:
    ensure_model_loaded()

    sequence_length = int(MODEL_INFO.get("sequence_length", DEFAULT_SEQUENCE_LENGTH))
    encoded = encode_prompt(prompt)

    prompt_count = len(encoded)
    generated_ids: List[int] = []
    steps: List[Dict[str, Any]] = []

    start_time = time.perf_counter()

    for step_index in range(int(max_tokens)):
        input_seq = make_input_sequence(encoded, sequence_length)
        raw_logits = MODEL.predict(input_seq, verbose=0)
        raw_logits = np.asarray(raw_logits).reshape(-1)

        probs = softmax_with_temperature(raw_logits, temperature)
        probs = clean_probabilities(probs, encoded)
        next_id, alternatives = top_k_sample(probs, top_k)
        next_token = I2W.get(next_id, f"<id:{next_id}>")

        encoded.append(next_id)
        generated_ids.append(next_id)

        elapsed = time.perf_counter() - start_time
        generated_count = len(generated_ids)
        remaining = int(max_tokens) - generated_count
        progress = generated_count / max(1, int(max_tokens)) * 100
        tps = generated_count / elapsed if elapsed > 0 else 0.0

        steps.append(
            {
                "index": step_index + 1,
                "token_id": next_id,
                "token": next_token,
                "alternatives": alternatives,
                "generated_text": " ".join(I2W.get(i, f"<id:{i}>") for i in generated_ids),
                "metrics": {
                    "prompt_tokens": prompt_count,
                    "shown_tokens": generated_count,
                    "remaining_tokens": remaining,
                    "current_context_size": min(len(encoded), sequence_length),
                    "progress": progress,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": tps,
                    "temperature": float(temperature),
                    "top_k": int(top_k),
                },
            }
        )


        if next_token in {"<EOS>", "<END>", "</s>"}:
            break

    total_time = time.perf_counter() - start_time
    final_text = " ".join(I2W.get(i, f"<id:{i}>") for i in generated_ids)
    return {
        "prompt_tokens": prompt_count,
        "tokens": [s["token"] for s in steps],
        "steps": steps,
        "final_text": final_text,
        "stats": {
            "total_time": total_time,
            "tokens_generated": len(steps),
            "tokens_per_second": (len(steps) / total_time) if total_time > 0 else 0.0,
            "sequence_length": sequence_length,
            "model_info": MODEL_INFO,
        },
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    try:
        ensure_model_loaded()
        return jsonify(MODEL_INFO)
    except Exception as exc:
        return jsonify({"loaded": False, "error": str(exc)})


@app.route("/api/prepare", methods=["POST"])
def api_prepare():
    try:
        prompt = request.form.get("prompt", "")
        temperature = float(request.form.get("temperature", DEFAULT_TEMPERATURE))
        top_k = int(request.form.get("top_k", DEFAULT_TOP_K))
        max_tokens = int(request.form.get("max_tokens", DEFAULT_MAX_TOKENS))
        use_rag = request.form.get("use_rag", "0") == "1"
        use_ner = request.form.get("use_ner", "0") == "1"
        use_ollama = request.form.get("use_ollama", "0") == "1"
        conversation_history_raw = request.form.get("conversation_history", "[]")
        try:
            conversation_items = json.loads(conversation_history_raw)
            conversation_lines = []
            for item in conversation_items[-8:]:
                role = "Utilisateur" if item.get("role") == "user" else "Assistant"
                content = str(item.get("content", "")).strip()
                if content:
                    conversation_lines.append(f"{role} : {content}")
            conversation_history = "\n".join(conversation_lines)
        except Exception:
            conversation_history = ""

        documents = read_uploaded_files(request.files.getlist("files"))
        rag_chunks: List[Dict[str, Any]] = []
        ner_entities: List[Dict[str, Any]] = []
        ner_error = ""

        if use_rag and documents:
            rag_chunks = simple_rag_retrieve(prompt, documents, top_n=3)
            prompt_for_model = build_rag_prompt(prompt, rag_chunks)
        else:
            uploaded_text = "\n".join(
                f"\n[FICHIER : {doc['filename']}]\n{doc['text'][:1000]}"
                for doc in documents
            )
            prompt_for_model = prompt + ("\n\n[CONTENU DES FICHIERS]\n" + uploaded_text if uploaded_text else "")


        if use_ner:
            try:
                if rag_chunks:
                    ner_text = " ".join(chunk["text"] for chunk in rag_chunks)
                elif documents:
                    ner_text = " ".join(doc["text"][:1500] for doc in documents)
                else:
                    ner_text = prompt
                ner_entities = summarize_entities(run_biomedical_ner(ner_text), limit=30)
            except Exception as exc:
                ner_error = str(exc)

        if use_ollama:


            ollama_prompt = build_ollama_prompt(prompt, rag_chunks, conversation_history=conversation_history)
            ollama_result = call_ollama_local(ollama_prompt, model_name=OLLAMA_MODEL)
            prompt_token_count = len(tokenize(ollama_prompt))
            result = ollama_response_to_steps(
                response_text=ollama_result["response"],
                prompt_token_count=prompt_token_count,
                elapsed_seconds=ollama_result["elapsed_seconds"],
                max_tokens=max_tokens,
            )
            result["generation_mode"] = "ollama"
            result["ollama_model"] = OLLAMA_MODEL
            result["ollama_api_url"] = OLLAMA_API_URL
            result["prompt_used"] = ollama_prompt[:2500]
        else:
            result = generate_steps(
                prompt=prompt_for_model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            result["generation_mode"] = "nexus_lm"
            result["prompt_used"] = prompt_for_model[:2500]

        result["rag_chunks"] = rag_chunks
        result["rag_enabled"] = use_rag
        result["ner_enabled"] = use_ner
        result["ollama_enabled"] = use_ollama
        result["ner_entities"] = ner_entities
        result["ner_error"] = ner_error
        return jsonify(result)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("Chargement de l'interface Nexus...")
    print(f"Dossier courant : {BASE_DIR}")
    print(f"Vocabulaire attendu : {VOCAB_PATH.name}")
    print(f"Poids attendus : {WEIGHTS_PATH.name}")
    print(f"Config attendue : {CONFIG_PATH.name}")
    print("Ouvrir : http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
