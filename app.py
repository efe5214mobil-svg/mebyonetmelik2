import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="MEB Yönetmelik Asistanı", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS (Görseldeki Stil İçin) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTitle { color: white; text-align: center; font-size: 3rem !important; }
    
    /* Kart Stilleri */
    .card-container { display: flex; gap: 20px; justify-content: center; margin-bottom: 30px; }
    .card {
        background-color: #1a1c24;
        border-radius: 15px;
        padding: 20px;
        width: 30%;
        border-top: 5px solid;
    }
    .card-red { border-color: #ff4b4b; }
    .card-blue { border-color: #0083ff; }
    .card-green { border-color: #00d488; }
    .card h3 { color: white; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
    .card ul { color: #a3a8b4; list-style-type: none; padding: 0; font-size: 0.9rem; }
    .card li { margin-bottom: 8px; }
    </style>
    """, unsafe_allow_header_allowed=True)

# --- BAŞLIK ---
st.markdown("<h1 class='stTitle'>🏛️ MEB Yönetmelik Asistanı</h1>", unsafe_allow_html=True)

# --- HIZLI SORULAR (KARTLAR) ---
st.markdown("### 💡 Hızlı Sorular")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card card-red">
        <h3>📜 Kayıt & Disiplin</h3>
        <ul>
            <li>• Disiplin cezaları nelerdir?</li>
            <li>• Kopya cezası nedir?</li>
            <li>• "Kınama" cezası alan öğrencinin dosyasına işlenir mi?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card card-blue">
        <h3>⌛ Devamsızlık</h3>
        <ul>
            <li>• 10/30 gün kuralı nedir?</li>
            <li>• Yarım gün izin devamsızlık sayılır mı?</li>
            <li>• Toplam devamsızlık sınırı ne zaman 60 güne çıkar?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card card-green">
        <h3>🎓 Başarı & Nakil</h3>
        <ul>
            <li>• Kaç zayıfla kalınır?</li>
            <li>• Nakil dönemi ne zamandır?</li>
            <li>• Onur Belgesi alma şartları nelerdir?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- MODÜLLER VE MANTIK (Buradan sonrası eski kodunuzla aynı) ---

@st.cache_resource
def load_existing_vector_db():
    persist_dir = "okul_asistani_v2_db" 
    if not os.path.exists(persist_dir):
        st.error("Vektör dosyası bulunamadı!")
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def ask_asistant(v_db, query):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    docs = v_db.similarity_search(query, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])
    
    system_msg = """Sen MEB Mevzuat Asistanısın. Yanıtların ÇOK KISA ve NET olmalı.
    ASLA DEĞİŞMEZ KURALLAR: 8 gün devamsızlık belgeye engel değildir. 50 ve üzeri not sorumluluktan geçer."""

    chat = client.chat.completions.create(
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": f"Bağlam: {baglam}\n\nSoru: {query}"}],
        model="llama-3.1-8b-instant", temperature=0
    )
    return chat.choices[0].message.content

v_db = load_existing_vector_db()

if v_db:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat Mesajlarını Görüntüle
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Girdisi
    if prompt := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = ask_asistant(v_db, prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
