import streamlit as st
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

st.set_page_config(page_title="Preguntas y Respuestas con PDF", layout="centered")
st.header("📄 Pregunta a tu PDF usando Ollama (Local)")

# Subida del archivo PDF
pdf_file = st.file_uploader("📥 Carga un archivo PDF", type="pdf")

@st.cache_resource
def procesar_pdf(pdf_file):
    """Procesa el PDF y genera embeddings para preguntas y respuestas."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Dividir el texto en fragmentos manejables
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    # Crear embeddings usando FastEmbedEmbeddings
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    return vectorstore

if pdf_file:
    # Procesar el PDF y generar el vectorstore
    vectorstore = procesar_pdf(pdf_file)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Configuración del modelo Ollama
    llm = OllamaLLM(model="llama3.2")  # Asegúrate de haber descargado este modelo localmente con `ollama pull `

    # Plantilla de preguntas y respuestas
    custom_prompt_template = """Usa la siguiente información para responder la pregunta del usuario:
    Si la respuesta no está dentro del protocolo, simplemente indica que esta fuera del contexto.

    Contexto: {context}
    Pregunta: {question}

    Solo responde en español.
    Respuesta:
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )

    # Input para la pregunta del usuario
    user_question = st.text_input("🧠 Haz una pregunta sobre el contenido del PDF:")
    if user_question:
        # Ejecutar la cadena de preguntas y respuestas
        response = qa_chain.invoke({"query": user_question})
        if 'result' in response:
            st.write(f"🤖 Respuesta: {response['result']}")
        else:
            st.write("🤖 Respuesta no disponible.")


