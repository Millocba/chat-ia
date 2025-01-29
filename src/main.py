import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import requests
from utils import *

# Configuración de la página
st.set_page_config(page_title="Preguntas y Respuestas con PDF", layout="centered")
st.header("📄 Pregunta a tu PDF usando DistilBERT (Local)")

# URL base del contenedor (ajustado según el puerto y la configuración del contenedor)
HUGGINGFACE_API_URL = "http://localhost:11434/qa"  # Cambiar al puerto del contenedor Hugging Face si es diferente

# Función para interactuar con el modelo en el contenedor
def query_huggingface_qa(context, question):
    """
    Envía una solicitud al modelo QA ejecutándose en el contenedor.
    """
    payload = {"context": context, "question": question}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(HUGGINGFACE_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()  # Devuelve la respuesta del modelo
    except requests.exceptions.RequestException as e:
        st.error(f"Error al interactuar con el modelo: {e}")
        return None

# Sidebar para subir archivos
with st.sidebar:
    archivos = load_name_files(FILE_LIST)
    files_uploaded = st.file_uploader(
        "Carga tu archivo",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button('Procesar'):
        for pdf in files_uploaded:
            if pdf is not None and pdf.name not in archivos:
                archivos.append(pdf.name)
                text_to_chromadb(pdf)

        archivos = save_name_files(FILE_LIST, archivos)

    if len(archivos) > 0:
        st.write("Archivos cargados:")
        st.write(archivos)
        if st.button('Borrar documentos'):
            archivos = []
            clean_files(FILE_LIST)

# Lógica para realizar la búsqueda y enviar los resultados al modelo
if archivos:
    user_question = st.text_input("Pregunta:")
    if user_question:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        collection = get_chroma_collection(INDEX_NAME)
        if collection is None:
            st.error("El índice ChromaDB no existe. Procesa documentos primero.")
        else:
            docs = search_chroma(collection, user_question, k=3)
            st.write(docs)
            if docs:
                # Preparar el contexto uniendo los documentos
                context = "Dame una respuesta elaborada en base al siguiente contexto ".join([str(doc) for doc in docs])
                st.write(f"Contexto utilizado: {context}")

                # Enviar el contexto y la pregunta al modelo QA
                model_response = query_huggingface_qa(context, user_question)

                if model_response:
                    st.write("Respuesta del modelo:")
                    st.write(model_response.get("answer", "No se generó una respuesta."))
                else:
                    st.warning("No se pudo obtener una respuesta del modelo.")
            else:
                st.warning("No se encontraron resultados relevantes.")

