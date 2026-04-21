import streamlit as st
import os
import sys

# --- CHROMADB SQLITE HATASI ÇÖZÜMÜ ---
# Streamlit Cloud'da SQLite versiyon hatası almamak için bu blok gereklidir
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="MEB Yönetmelik Asistanı", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTitle { color: white; text-align: center; font-size: 3rem !important; margin-bottom: 2rem; }
    
    /* Kart Stilleri */
    .card {
        background-color: #1a1c24;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 10px;
        border-top: 5px solid;
        min-height: 250px;
    }
    .card-red { border-color: #ff4b4b; }
    .card-blue { border-color: #0083ff; }
    .card-green { border-color: #00d488; }
    .card h3 { color: white; margin-bottom: 15px; font-size: 1.2rem; }
    .card ul { color: #a3a8b4; list-style-type: none; padding: 0; font-size: 0.9rem; }
    .card li { margin-bottom: 8px; }
    </style>
    """, unsafe_allow_html=True)

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

# --- VEKTÖR VERİTABANI VE ASİSTAN MANTIĞI ---

@st.cache_resource
def load_existing_vector_db():
    persist_dir = "okul_asistani_v2_db" 
    if not os.path.exists(persist_dir):
        st.error(f"Vektör dosyası '{persist_dir}' dizininde bulunamadı! Lütfen klasörü GitHub'a yüklediğinizden emin olun.")
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def ask_asistant(v_db, query):
    # API anahtarını st.secrets üzerinden alıyoruz
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # Benzer dokümanları getir
    docs = v_db.similarity_search(query, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])
    
    system_msg = """Sen MEB Mevzuat Asistanısın. Yanıtların ÇOK KISA ve NET olmalı. 
    Verilen bağlama dayanarak cevap ver. Bilgin yoksa 'Bu konuda net bir bilgi bulamadım' de.
    8 gün devamsızlık belgeye engel değildir (5 günü geçerse engeldir). 50 ve üzeri not sorumluluktan geçer."""

    chat = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Bağlam: {baglam}\n\nSoru: {query}"}
        ],
        model="llama-3.1-8b-instant", 
        temperature=0
    )
    return chat.choices[0].message.content

# --- UYGULAMA AKIŞI ---

v_db = load_existing_vector_db()

if v_db:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat Mesaj Geçmişini Görüntüle
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Kullanıcı Girdisi
    if prompt := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                response = ask_asistant(v_db, prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
