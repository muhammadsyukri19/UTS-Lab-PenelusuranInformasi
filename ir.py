import os
import sys
import pandas as pd
import re
import time
import csv
import numpy as np

# --- 1. LIBRARY YANG DIIZINKAN OLEH UTS ---
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, STORED
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 2. LIBRARY TAMBAHAN UNTUK PREPROCESSING (SASTRAWI) ---
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- KONFIGURASI GLOBAL ---
INDEX_DIR = "whoosh_index"
CLEAN_DATA_PATH = "datasets_clean"
DATASET_FILES = [
    'etd-usk_clean.csv', 
    'etd-ugm_clean.csv', 
    'kompas_clean.csv', 
    'tempo_clean.csv', 
    'mojok_clean.csv'
]

# Menambah batas ukuran field (wajib untuk CSV dengan teks panjang)
csv.field_size_limit(sys.maxsize) 

# Inisialisasi Sastrawi (Wajib diulang untuk preprocessing query)
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# Variabel Global untuk VSM dan Data
df_documents = pd.DataFrame()
vectorizer = None
doc_term_matrix = None
is_ready = False
# List untuk menampung seluruh konten bersih (untuk BoW)
doc_contents = []

# --- 1. PREPROCESSING UNTUK QUERY (HARUS SAMA DENGAN 01_preprocessing.py) ---
def preprocess_query(text):
    """
    Melakukan Preprocessing untuk query input agar konsisten dengan dokumen.
    """
    if pd.isna(text) or text is None:
        return ""

    text = str(text) 
    
    # 1. Case Folding
    text = text.lower()
    
    # 2. PEMBESIHAN NEWLINE DAN TAB
    text = re.sub(r'[\n\r\t]', ' ', text)
    
    # 3. PENGHILANGAN KARAKTER NON-STANDAR & ANGKA
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = re.sub(r'[^a-z\s]', ' ', text) 
    
    # 4. Hapus spasi ganda
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = text.split()
    tokens = [word for word in tokens if len(word) > 1] 
    
    # 5. Stopword Removal
    text = stopword_remover.remove(" ".join(tokens))
    
    # 6. Stemming
    stemmed_text = stemmer.stem(text)
    
    # 7. Koreksi Over-Stemming
    final_tokens = []
    for word in stemmed_text.split():
        if word == 'cuku':
            final_tokens.append('cukup')
        else:
            final_tokens.append(word)

    return " ".join(final_tokens)


# --- 2. FUNGSI LOAD & INDEX DATASET ---
def load_and_index_dataset():
    """
    Menggabungkan semua dokumen bersih, membuat Whoosh Index, dan menyiapkan VSM.
    """
    global df_documents, doc_term_matrix, vectorizer, doc_contents, is_ready

    # 1. Mengumpulkan Dokumen Bersih dari folder 'datasets_clean'
    print("\n--- 1. Mengumpulkan Dokumen Bersih dari 'datasets_clean' ---")
    
    list_df = []
    for file_name in DATASET_FILES:
        file_path = os.path.join(CLEAN_DATA_PATH, file_name)
        source = file_name.replace('_clean.csv', '').upper()
        
        if not os.path.exists(file_path):
            print(f"Peringatan: File bersih '{file_name}' tidak ditemukan. Melewati.")
            continue
        
        try:
            # Membaca file bersih yang sudah dipreprocessing (tidak perlu engine='python' lagi)
            df_temp = pd.read_csv(file_path, encoding='utf-8')
            df_temp['source'] = source
            list_df.append(df_temp)
            print(f"  -> Berhasil memuat {len(df_temp)} dokumen dari {file_name}.")
        except Exception as e:
            print(f"ERROR membaca {file_name}: {e}")
            continue

    if not list_df:
        print("Error: Tidak ada dokumen bersih yang berhasil dimuat. Pastikan Anda menjalankan 01_preprocessing.py terlebih dahulu!")
        return
        
    # Menggabungkan semua DataFrame
    df_documents = pd.concat(list_df, ignore_index=True)
    df_documents.reset_index(names=['doc_id'], inplace=True)
    
    total_documents = len(df_documents)
    print(f"\n[SUKSES] Total {total_documents} dokumen bersih berhasil digabungkan.")
    doc_contents = df_documents['konten_bersih'].tolist()

    # 2. Implementasi Indexing dengan Whoosh
    print("\n--- 2. Implementasi Indexing dengan Whoosh ---")
    
    schema = Schema(
        doc_id=ID(stored=True),
        title=STORED,
        source=STORED,
        konten_bersih=TEXT(stored=True)
    )

    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)

    ix = create_in(INDEX_DIR, schema)
    writer = ix.writer()
    
    start_time_index = time.time()
    for index, row in df_documents.iterrows():
        writer.add_document(
            doc_id=str(row['doc_id']),
            title=str(row['judul']),
            source=str(row['source']),
            konten_bersih=str(row['konten_bersih'])
        )
        
        # Progress bar Whoosh (setiap 10000 dokumen)
        if index > 0 and index % 10000 == 0:
            print(f"\r  -> Indexing Whoosh: {index}/{total_documents} ({index/total_documents*100:.1f}%)", end="", flush=True)

    writer.commit()
    end_time_index = time.time()
    print(f"\nIndexing Whoosh selesai dalam {end_time_index - start_time_index:.2f} detik. Total {total_documents} dokumen di-index.")

    # 3. Representasi Dokumen (Bag of Words / VSM)
    print("\n--- 3. Representasi Dokumen (Bag of Words / VSM) ---")
    start_time_vsm = time.time()
    
    # Inisialisasi dan latih CountVectorizer
    vectorizer = CountVectorizer()
    # Membuat Term-Document Matrix (Matriks BoW)
    doc_term_matrix = vectorizer.fit_transform(doc_contents)
    
    end_time_vsm = time.time()
    print(f"Matriks BoW dibuat ({doc_term_matrix.shape[0]} doks, {doc_term_matrix.shape[1]} terms) dalam {end_time_vsm - start_time_vsm:.2f} detik.")
    
    is_ready = True
    print("\n[SUKSES] Sistem siap untuk melakukan pencarian VSM/Ranking.")


# --- 3. FUNGSI SEARCH & RANKING ---
def search_and_rank(query, top_n=5):
    """
    Memproses query, menghitung Cosine Similarity, dan menampilkan ranking.
    """
    global doc_term_matrix, vectorizer, df_documents

    start_time = time.time()
    
    # 1. Preprocessing Query
    clean_query = preprocess_query(query)
    if not clean_query:
        print("Peringatan: Query tidak valid atau menjadi kosong setelah preprocessing/stopword removal.")
        return

    # 2. Vektorisasi Query
    # Menggunakan vectorizer yang sama yang sudah dilatih pada dokumen
    query_vector = vectorizer.transform([clean_query])
    
    # 3. Perhitungan Cosine Similarity
    # Menghitung kemiripan antara query (1 vektor) dan semua dokumen (banyak vektor)
    cosine_similarities = cosine_similarity(query_vector, doc_term_matrix).flatten()

    # 4. Ranking Hasil
    # Mendapatkan indeks dokumen yang diurutkan dari skor tertinggi
    ranked_indices = np.argsort(cosine_similarities)[::-1]
    
    # Filter dokumen dengan skor di atas nol
    relevant_indices = [idx for idx in ranked_indices if cosine_similarities[idx] > 0]
    
    if not relevant_indices:
        print("\nTidak ditemukan dokumen yang relevan.")
        return

    # Mengambil Top N
    top_n_indices = relevant_indices[:top_n]
    
    end_time = time.time()
    print(f"\nDitemukan {len(relevant_indices)} dokumen relevan (dari {len(doc_term_matrix.toarray())} total).")
    print(f"Ranking selesai dalam {end_time - start_time:.4f} detik.")

    # 5. Tampilkan Hasil
    print(f"\n=== TOP {top_n} HASIL PENCARIAN (Cosine Similarity) ===")
    
    results = []
    for rank, doc_index in enumerate(top_n_indices):
        score = cosine_similarities[doc_index]
        doc_data = df_documents.iloc[doc_index]
        
        results.append({
            'rank': rank + 1,
            'score': score,
            'title': doc_data['judul'],
            'source': doc_data['source']
        })

        print(f"[{rank + 1}] Skor: {score:.4f} | Judul: {doc_data['judul']} ({doc_data['source']})")
    
    print("="*60)
    # Tampilkan query bersih
    print(f"Query Bersih: {clean_query}")
    print("="*60)
    

# --- 4. FUNGSI UTAMA CLI ---
def main_cli():
    """
    Antarmuka Command Line Interface utama.
    """
    global is_ready
    
    # Periksa apakah index Whoosh sudah ada (untuk memulai dengan status yang benar)
    if os.path.exists(INDEX_DIR) and os.path.exists(os.path.join(INDEX_DIR, 'MAIN_CONTENT')):
        # Whoosh index ditemukan, tapi VSM perlu dimuat ulang
        is_ready = False
        status = "Index Whoosh ditemukan. Pilih [1] untuk memuat VSM/BoW."
    else:
        status = "Perlu Indexing"

    while True:
        status = "Siap Mencari" if is_ready else status
        
        print("\n" + "="*40)
        print("=== INFORMATION RETRIEVAL SYSTEM ===")
        print(f"Status: {status}")
        print("="*40)
        print("[1] Load & Index Dataset")
        print("[2] Search Query")
        print("[3] Exit")
        print("="*40)

        choice = input("Pilih menu [1/2/3]: ")

        if choice == '1':
            load_and_index_dataset()
        
        elif choice == '2':
            if not is_ready:
                print("\n[Peringatan] Sistem belum siap. Silakan pilih [1] Load & Index Dataset terlebih dahulu.")
                continue
            
            query = input("Masukkan Query Pencarian Anda: ")
            search_and_rank(query)
            
        elif choice == '3':
            print("Terima kasih. Sistem dimatikan.")
            sys.exit()
            
        else:
            print("Pilihan tidak valid. Silakan pilih 1, 2, atau 3.")


if __name__ == "__main__":
    main_cli()
