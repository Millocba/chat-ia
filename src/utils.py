import json
import os
import requests
import streamlit as st
import chromadb
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

FILE_LIST = "src/archivos.txt"
INDEX_NAME = 'taller'

# Ajusta la conexión al cliente de ChromaDB si se está ejecutando en un contenedor
chroma_client = chromadb.HttpClient(host='localhost', port=8000)  # Cambia el puerto según tu configuración

def save_name_files(path, new_files):
    old_files = load_name_files(path)
    with open(path, "a") as file:
        for item in new_files:
            if item not in old_files:
                file.write(item + "\n")
                old_files.append(item)
    return old_files


def load_name_files(path):
    archivos = []
    with open(path, "r") as file:
        for line in file:
            archivos.append(line.strip())
    return archivos


def clean_files(path):
    with open(path, "w") as file:
        pass
    chroma_client.delete_collection(name=INDEX_NAME)
    chroma_client.create_collection(name=INDEX_NAME)
    return True


def text_to_chromadb(pdf):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, pdf.name)
    with open(temp_filepath, "wb") as f:
        f.write(pdf.getvalue())

    loader = PyPDFLoader(temp_filepath)
    text = loader.load()

    with st.spinner(f'Creando embeddings para el fichero: {pdf.name}'):
        create_embeddings(pdf.name, text)

    return True


def get_chroma_collection(index_name):
    try:
        return chroma_client.get_collection(name=index_name)
    except Exception as e:
        print(f"Error al obtener la colección Chroma: {e}")
        return None


def search_chroma(collection, query, k=5):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=k
        )
        return results.get("documents", [])
    except Exception as e:
        print(f"Error al realizar la búsqueda: {e}")
        return []


def create_embeddings(file_name, text):
    print(f"Creando embeddings para el archivo: {file_name}")

    # Dividir el texto en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(text)

    # Añadir 'page_number' si no está en los metadatos
    for i, chunk in enumerate(chunks):
        if 'page_number' not in chunk.metadata:
            chunk.metadata['page_number'] = i + 1  # Asignar un número de página genérico

    # Configurar embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Obtener colección Chroma
    collection = chroma_client.get_or_create_collection(name=INDEX_NAME)

    # Agregar embeddings a la colección
    for chunk in chunks:
        collection.add(
            documents=[chunk.page_content],
            metadatas=[chunk.metadata],
            ids=[f"{file_name}-{chunk.metadata['page_number']}"]
        )
    print(f"Embeddings creados y agregados a la colección {INDEX_NAME}.")
