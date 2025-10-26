import os
import sys
import pandas as pd
import re
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, STORED
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import csv # Import library csv

# Menambah batas ukuran field untuk mengatasi error field terlalu besar pada CSV korup
# Ini sering terjadi pada file tesis/disertasi
csv.field_size_limit(sys.maxsize) 


# --- KONFIGURASI GLOBAL ---
INDEX_DIR = "whoosh_index"
dataset_PATH = "dataset"
# Daftar file CSV WAJIB. Jika ingin debugging 1 dataset, ubah list ini (misal: ["etd-ugm.csv"])
DATASET_FILES = ["etd_usk.csv", "etd_ugm.csv", "kompas.csv", "tempo.csv", "mojok.csv"]

# Variabel Global untuk VSM dan Data
df_documents = pd.DataFrame()
vectorizer = None
doc_term_matrix = None
doc_contents = [] 

# Inisialisasi Sastrawi (Stemmer & Stopword Removal)
stemmer = StemmerFactory().create_stemmer()
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
stopword_factory = StopWordRemoverFactory()
stop_words = stopword_factory.get_stop_words()


# --- FASE I: PREPROCESSING ---
def preprocess_text(text):
    """Melakukan Preprocessing Teks (Case Folding, Cleaning, Stemming, Stopword Removal)."""
    if pd.isna(text) or text is None:
        return ""

    text = str(text) 
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text) # Ganti karakter non-huruf dengan spasi
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi ganda

    # Stemming
    stemmed_text = stemmer.stem(text)

    tokens = stemmed_text.split()
    # Stopword Removal
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    return " ".join(tokens)

def collect_documents():
    """Mengumpulkan dan memproses dokumen dari semua file dataset CSV."""
    global df_documents
    data = []
    doc_id_counter = 0

    print("Mulai mengumpulkan dan memproses dokumen dari file CSV...")
    
    if not os.path.exists(dataset_PATH):
        print(f"Error: Direktori '{dataset_PATH}/' tidak ditemukan.")
        return False
        
    # --- PENTING: KOLOM TEXT DAN JUDUL ANDA ---
    # Ganti 'konten' dan 'judul' jika nama kolom di CSV Anda berbeda.
    TEXT_COLUMN_CANDIDATES = ['konten', 'judul', 'content', 'text', 'abstract', 'body']
    # -------------------------------------------------------------------
        
    total_files = len(DATASET_FILES)
    files_processed = 0

    for file_name in DATASET_FILES:
        file_path = os.path.join(dataset_PATH, file_name)
        source = file_name.replace('.csv', '')
        
        if not os.path.exists(file_path):
            print(f"Peringatan: File dataset '{file_path}' tidak ditemukan. Melewati.")
            files_processed += 1
            continue
            
        print(f"  -> Memproses dataset: {file_name}...")
        
        try:
            # --- PERBAIKAN KRITIS: Menghapus low_memory=False karena tidak didukung engine='python' ---
            df_temp = pd.read_csv(
                file_path, 
                sep=',', 
                encoding='latin1', 
                engine='python',
                on_bad_lines='warn', # Memberikan peringatan jika ada baris korup, tapi tidak menghentikan
            ) 
            
            # 1. Mencari kolom teks utama (Konten)
            text_column = None
            for col in TEXT_COLUMN_CANDIDATES:
                if col in df_temp.columns and len(df_temp[col].dropna()) > 0:
                    text_column = col
                    break

            if text_column is None:
                print(f"  [SKIPPED] Tidak ditemukan kolom teks relevan di {file_name}")
                files_processed += 1
                continue
            
            # 2. Mencari kolom Judul
            title_column = 'judul' if 'judul' in df_temp.columns else (
                'title' if 'title' in df_temp.columns else text_column
            )
            
            # Inisialisasi progress bar per file
            total_rows = len(df_temp)
            rows_counter = 0
            
            # Variabel untuk melacak progres terakhir yang dicetak (kelipatan 5%)
            last_percentage_printed = -5 

            # Iterasi per baris (dokumen)
            for index, row in df_temp.iterrows():
                # Pastikan konten teks ada (tidak NaN)
                raw_content = str(row[text_column]) if pd.notna(row[text_column]) else ""
                
                # Ambil Judul
                title = str(row[title_column]) if pd.notna(row[title_column]) else f"{source} Doc {index+1}"

                # Preprocessing
                clean_content = preprocess_text(raw_content)
                
                # Simpan data hanya jika konten bersih tidak kosong
                if clean_content:
                    data.append({
                        'doc_id': doc_id_counter,
                        'title': title.strip().title(), 
                        'source': source,
                        'raw_content': raw_content,
                        'clean_content': clean_content
                    })
                    doc_id_counter += 1
                
                rows_counter += 1
                
                # --- LOGIKA PROGRESS BAR (MENCETAK KELIPATAN 5%) ---
                current_percentage = int((rows_counter / total_rows) * 100)
                
                # Cek apakah persentase saat ini adalah kelipatan 5% yang belum dicetak, atau sudah baris terakhir
                if current_percentage >= last_percentage_printed + 5 or rows_counter == total_rows:
                    if rows_counter != total_rows:
                        last_percentage_printed = current_percentage
                        
                    # Mencetak progres menggunakan print biasa (tidak menimpa baris)
                    print(f"  -> Progress {file_name}: {rows_counter}/{total_rows} ({current_percentage}%)", flush=True)


            print(f"  -> Progress {file_name}: Selesai ({total_rows} dokumen).") # Baris baru setelah selesai

        except Exception as e:
            print(f"\nGagal membaca/memproses file {file_path}: {e}")
            
        files_processed += 1
        # Menampilkan progress total dataset
        total_percentage = (files_processed / total_files) * 100
        print(f"  -> [TOTAL PROGRESS DATASET: {files_processed}/{total_files} ({total_percentage:.0f}%)]")

    if not data:
        print("Error: Tidak ada dokumen yang berhasil dimuat.")
        return False

    df_documents = pd.DataFrame(data)
    print(f"\nTotal {len(df_documents)} dokumen berhasil dimuat dan diproses.")
    return True

# --- FASE II: INDEXING (WHOOSH) ---
def create_whoosh_schema():
    """Mendefinisikan skema untuk Whoosh Index."""
    return Schema(
        doc_id=ID(stored=True, unique=True),
        title=STORED, 
        source=STORED, 
        clean_content=TEXT(stored=True) 
    )

def index_documents():
    """Membuat Whoosh Index dari dokumen yang sudah diproses."""
    global doc_contents

    if df_documents.empty:
        print("Dataframe dokumen kosong. Silakan jalankan Load Dataset terlebih dahulu.")
        return

    print(f"\nMembuat Whoosh Index di direktori: {INDEX_DIR}")

    # 1. Persiapan Index
    schema = create_whoosh_schema()
    if not os.path.exists(INDEX_DIR):
        os.mkdir(INDEX_DIR)
        
    ix = create_in(INDEX_DIR, schema)
    writer = ix.writer()
    
    # 2. Menulis Dokumen
    start_time = time.time()
    
    total_docs = len(df_documents)
    
    for index, row in df_documents.iterrows():
        try:
            writer.add_document(
                doc_id=str(row['doc_id']),
                title=row['title'],
                source=row['source'],
                clean_content=row['clean_content']
            )
        except Exception as e:
            print(f"Gagal meng-index dokumen {row['doc_id']} ({row['title']}): {e}")
            
        # Tampilkan progress bar Whoosh Indexing (diperbarui setiap 5000 dokumen atau pada akhir)
        # Menggunakan \r dan end="" di sini karena total dokumen sudah fix dan lebih stabil
        if (index + 1) % 5000 == 0 or index == total_docs - 1:
             percentage = ((index + 1) / total_docs) * 100
             print(f"\r  -> Indexing Whoosh: {index + 1}/{total_docs} ({percentage:.1f}%)", end="", flush=True)


    writer.commit()
    end_time = time.time()
    print(f"\rIndexing Whoosh selesai dalam {end_time - start_time:.2f} detik. Total {total_docs} dokumen di-index.")
    
    # 3. Persiapan untuk VSM (BoW)
    doc_contents = df_documents['clean_content'].tolist()
    print("Data konten bersih siap untuk BoW.")

# --- FASE III & IV: VSM, SEARCH & RANKING ---
def prepare_vsm():
    """Membuat Matriks Bag-of-Words (BoW) untuk perhitungan Cosine Similarity."""
    global vectorizer, doc_term_matrix

    if not doc_contents:
        print("Konten dokumen kosong. Pastikan indexing sudah dilakukan.")
        return False

    print("\nMembuat Matriks Bag-of-Words (BoW) dengan CountVectorizer...")
    start_time = time.time()
    
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(doc_contents)
    
    end_time = time.time()
    print(f"BoW Matrix (TD-Matrix) dibuat ({doc_term_matrix.shape[0]} doks, {doc_term_matrix.shape[1]} terms) dalam {end_time - start_time:.2f} detik.")
    return True

def search_and_rank(query_text, top_k=5):
    """Melakukan pencarian Whoosh dan ranking Cosine Similarity."""
    if df_documents.empty or vectorizer is None or doc_term_matrix is None:
        print("\n[PERINGATAN] Sistem belum siap. Silakan jalankan menu [1] terlebih dahulu.")
        return

    print("\n--- PROSES PENCARIAN & RANKING ---")
    
    clean_query = preprocess_text(query_text)
    if not clean_query:
        print("Query setelah diproses kosong. Coba gunakan kata kunci yang lebih spesifik.")
        return

    query_vector = vectorizer.transform([clean_query])
    similarity_scores = cosine_similarity(query_vector, doc_term_matrix).flatten()
    ranked_indices = np.argsort(similarity_scores)[::-1]
    
    top_results_data = []
    
    for rank, doc_index in enumerate(ranked_indices):
        score = similarity_scores[doc_index]
        if score > 0 and len(top_results_data) < top_k:
            doc_data = df_documents.iloc[doc_index]
            top_results_data.append({
                'rank': len(top_results_data) + 1,
                'score': score,
                'title': doc_data['title'],
                'source': doc_data['source'],
                'doc_id': doc_data['doc_id']
            })
        elif len(top_results_data) >= top_k:
            break

    print(f"Ditemukan {len(top_results_data)} dokumen relevan (dari {len(df_documents)} total).")
    
    if top_results_data:
        print("\n=== TOP 5 HASIL PENCARIAN (Cosine Similarity) ===")
        for res in top_results_data:
            print(f"[{res['rank']}] Skor: {res['score']:.4f} | Judul: {res['title']} ({res['source']}) | ID: {res['doc_id']}")
        print("===================================================")
    else:
        print("\nTidak ada dokumen yang relevan ditemukan dengan query Anda (Skor = 0).")


# --- CLI INTERFACE ---
def load_and_index_process():
    """Handler untuk menu [1] Load & Index Dataset."""
    if collect_documents():
        index_documents()
        prepare_vsm()
        print("\n[SUKSES] Sistem siap untuk melakukan pencarian. Silakan pilih menu [2].")

def search_query_process():
    """Handler untuk menu [2] Search Query."""
    if df_documents.empty or vectorizer is None or doc_term_matrix is None:
        print("\n[PERINGATAN] Sistem belum siap. Silakan jalankan menu [1] terlebih dahulu.")
        return

    query = input("\nMasukkan Query Pencarian Anda: ")
    if query:
        search_and_rank(query, top_k=5)
    else:
        print("Query tidak boleh kosong.")


def main_cli():
    """Fungsi Utama CLI."""
    
    if os.path.exists(INDEX_DIR) and os.path.isdir(INDEX_DIR):
        print(f"Index Whoosh ditemukan di '{INDEX_DIR}'. Memuat data...")
        try:
            if collect_documents():
                 prepare_vsm()
                 print("[READY] Sistem dimuat dari data yang sudah ada. Siap mencari.")
            else:
                print("Gagal memuat dokumen meskipun index ada. Silakan jalankan [1].")

        except Exception as e:
            print(f"Error saat memuat index/data: {e}. Silakan jalankan menu [1] untuk buat ulang.")
            
    else:
        print("[INFO] Index Whoosh belum ditemukan. Silakan jalankan menu [1] untuk membuat index.")


    while True:
        print("\n" + "=" * 35)
        print("=== INFORMATION RETRIEVAL SYSTEM ===")
        print(f"Status: {'Siap Mencari' if vectorizer is not None else 'Perlu Indexing'}")
        print("=" * 35)
        print("[1] Load & Index Dataset")
        print("[2] Search Query")
        print("[3] Exit")
        print("=" * 35)

        choice = input("Pilih menu [1/2/3]: ")

        if choice == '1':
            load_and_index_process()
        elif choice == '2':
            search_query_process()
        elif choice == '3':
            print("Terima kasih. Program dihentikan.")
            sys.exit(0)
        else:
            print("Pilihan tidak valid. Silakan coba lagi.")


if __name__ == "__main__":
    main_cli()
