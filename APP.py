import streamlit as st
import pandas as pd
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

# Configuración de la página
st.set_page_config(page_title="RAG vs Base Model LLM", layout="wide")
st.title("🧠 Comparativa: Modelo Base vs RAG (Groq Llama 3.3 70B)")

# 1. Ingreso de API y Configuración en el Sidebar
st.sidebar.header("Configuración")
api_key = st.sidebar.text_input("Ingresa tu Groq API Key", type="password")
chunk_size = st.sidebar.slider("Tamaño del Chunk (caracteres)", min_value=100, max_value=1000, value=300)

st.sidebar.markdown("""
**Modelo utilizado:** `llama-3.3-70b-versatile`
""")

# 2. Entradas de Usuario
col1, col2 = st.columns([1, 1])
with col1:
    contexto = st.text_area("📚 Ingresa el Contexto (Documento base para el RAG)", height=200, 
                            placeholder="Pega aquí un texto largo, un artículo, o información específica...")
with col2:
    prompt_usuario = st.text_area("💬 Ingresa tu Prompt (Pregunta)", height=200, 
                                  placeholder="Haz una pregunta relacionada al contexto...")

def aproximar_tokens(texto):
    # Aproximación básica: 1 token ≈ 4 caracteres en inglés/español promedio
    return len(texto) // 4

if st.button("Ejecutar Análisis y Comparar", type="primary"):
    if not api_key:
        st.error("Por favor, ingresa tu API Key de Groq en la barra lateral.")
        st.stop()
    if not prompt_usuario:
        st.warning("Por favor, ingresa un prompt.")
        st.stop()

    client = Groq(api_key=api_key)
    modelo = "llama-3.3-70b-versatile"

    contexto_relevante = ""

    # --- MOSTRAR FUNCIONAMIENTO DEL RAG ---
    if contexto:
        st.divider()
        st.subheader("🔍 Funcionamiento Interno del RAG")
        
        # A. Chunking y Tokenización
        chunks = textwrap.wrap(contexto, width=chunk_size, break_long_words=False)
        
        st.write("### 1. Tokenizer y Chunking")
        chunk_data = []
        for i, chunk in enumerate(chunks):
            tokens = aproximar_tokens(chunk)
            chunk_data.append({"Chunk ID": i, "Texto": chunk[:50] + "...", "Caracteres": len(chunk), "Tokens (Aprox)": tokens})
        
        st.dataframe(pd.DataFrame(chunk_data), use_container_width=True)

        # B. Evaluación Matemática / Geométrica (Vectorización y Similitud del Coseno)
        st.write("### 2. Evaluación Matemática y Geométrica (Similitud del Coseno)")
        st.markdown(r"Se evalúa la distancia vectorial entre la pregunta y los chunks usando TF-IDF y la fórmula: $\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$")
        
        # Vectorizamos el prompt y los chunks
        vectorizer = TfidfVectorizer()
        textos_totales = [prompt_usuario] + chunks
        matriz_tfidf = vectorizer.fit_transform(textos_totales)
        
        # Separar el vector del prompt de los vectores de los chunks
        vector_prompt = matriz_tfidf[0:1]
        vectores_chunks = matriz_tfidf[1:]
        
        # Calcular similitud del coseno
        similitudes = cosine_similarity(vector_prompt, vectores_chunks)[0]
        
        # Guardar resultados
        similitud_data = []
        for i, similitud in enumerate(similitudes):
            similitud_data.append({"Chunk ID": i, "Similitud Geométrica (0 a 1)": round(similitud, 4), "Texto Completo": chunks[i]})
        
        df_similitud = pd.DataFrame(similitud_data).sort_values(by="Similitud Geométrica (0 a 1)", ascending=False)
        st.dataframe(df_similitud[["Chunk ID", "Similitud Geométrica (0 a 1)", "Texto Completo"]], use_container_width=True)

        # Seleccionar los top chunks como contexto
        top_k = min(3, len(chunks)) # Tomamos los 3 mejores
        mejores_chunks = df_similitud.head(top_k)["Texto Completo"].tolist()
        contexto_relevante = "\n\n".join(mejores_chunks)
        
        st.success(f"Se seleccionaron los {top_k} fragmentos con mayor similitud geométrica para inyectar al modelo.")

    st.divider()
    st.subheader("🤖 Comparación de Respuestas del Modelo")
    
    resp_col1, resp_col2 = st.columns(2)

    # Llama 3.3 SIN Contexto
    with resp_col1:
        st.markdown("#### Llama 3.3 (Sin Contexto)")
        with st.spinner("Generando respuesta base..."):
            try:
                chat_completion_base = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "Eres un asistente útil y directo."},
                        {"role": "user", "content": prompt_usuario}
                    ],
                    model=modelo,
                    temperature=0.3,
                )
                st.info(chat_completion_base.choices[0].message.content)
            except Exception as e:
                st.error(f"Error en la API: {e}")

    # Llama 3.3 CON Contexto (RAG)
    with resp_col2:
        st.markdown("#### Llama 3.3 (Con RAG / Contexto)")
        if not contexto:
            st.warning("No ingresaste contexto, por lo que el resultado será igual.")
        else:
            with st.spinner("Generando respuesta con contexto..."):
                prompt_enriquecido = f"Utiliza el siguiente contexto para responder a la pregunta de forma precisa. Si la respuesta no está en el contexto, indícalo.\n\nContexto:\n{contexto_relevante}\n\nPregunta: {prompt_usuario}"
                try:
                    chat_completion_rag = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Eres un asistente útil. Basa tu respuesta en el contexto proporcionado."},
                            {"role": "user", "content": prompt_enriquecido}
                        ],
                        model=modelo,
                        temperature=0.3,
                    )
                    st.success(chat_completion_rag.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error en la API: {e}")
