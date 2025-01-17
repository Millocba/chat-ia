import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from utils import *

# Configuración de la página
st.set_page_config(page_title="Preguntas y Respuestas con PDF", layout="centered")
st.header("📄 Pregunta a tu PDF usando Ollama (Local)")

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

# Lógica para realizar la búsqueda y enviar los resultados a LLaMA
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
            if docs:
                # Asegurarnos de que cada elemento en docs sea una cadena de texto (string)
                context = "\n".join([str(doc) for doc in docs])  # Convertimos cada doc a string
                prompt = f"Pregunta: {user_question}\nContexto: {context}\nRespuesta:"

                # Enviar el prompt a LLaMA
                llama_response = query_llama_32(prompt)

                if llama_response:
                    st.write("Respuesta de LLaMA 3.2:")
                    st.write(llama_response)  # Mostrar solo la respuesta generada por el modelo
                else:
                    st.warning("No se pudo obtener una respuesta de LLaMA.")
            else:
                st.warning("No se encontraron resultados relevantes.")
