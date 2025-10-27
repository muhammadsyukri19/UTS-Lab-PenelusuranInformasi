import pandas as pd
import re
import os
import sys
import csv
import time

# --- KONFIGURASI UMUM ---
# 1. Tentukan folder data mentah dan folder hasil pemrosesan
RAW_DATA_PATH = "datasets"
CLEAN_DATA_PATH = "datasets_clean"

# 2. Daftar Stopword Bahasa Indonesia (Pure Python Standard Library)
# List ini diambil dari sumber umum (Stopwords Indonesia) dan dihardcode untuk menghindari Sastrawi/NLTK.
# Jika ingin lebih lengkap, tambahkan kata-kata di sini.
STOPWORDS = set([
    "yang", "untuk", "pada", "ke", "para", "namun", "menurut", "tentang", "demikian", 
    "adalah", "dengan", "adanya", "adapun", "akan", "adalah", "adanya", "adapun", 
    "agak", "agar", "akibat", "akhir", "akhirnya", "aku", "akan", "adalah", "adanya", 
    "adapun", "agak", "agar", "akibat", "akhir", "akhirnya", "aku", "anda", "apakah", 
    "apabila", "atau", "bagai", "bahkan", "bahwa", "begini", "begitu", "belum", "bisa", 
    "cuma", "dapat", "dari", "dan", "di", "dia", "diri", "dlm", "dua", "empat", "enggak",
    "hal", "hanya", "ia", "ini", "itu", "jangan", "jika", "kami", "kamu", "kali", "karena", 
    "ketika", "kita", "kok", "lah", "lima", "maka", "mana", "mereka", "nih", "nya", "oleh", 
    "pada", "paling", "para", "pun", "saja", "sambil", "sana", "satu", "sebagai", "sejak", 
    "selalu", "semua", "sendiri", "seperti", "serta", "siapa", "sudah", "tiga", "tujuh", 
    "untuk", "wah", "yakni", "yaitu", "yg", "uu", "aa"
])

# Menambah batas ukuran field (wajib untuk CSV dengan teks panjang)
csv.field_size_limit(sys.maxsize) 

# --- FUNGSI PREPROCESSING ---
def preprocess_text(text):
    """Melakukan Case Folding, Cleaning Teks, dan Stopword Removal."""
    if pd.isna(text) or text is None:
        return ""

    text = str(text) 
    
    # 1. Case Folding (Wajib)
    text = text.lower()
    
    # 2. Pembersihan Karakter (Hapus non-huruf dan non-spasi, termasuk angka)
    text = re.sub(r'[^a-z\s]', ' ', text) 
    
    # 3. Hapus spasi ganda dan spasi di awal/akhir
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Tokenization dan Stopword Removal (Wajib)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 1]
    
    return " ".join(tokens)

# --- FUNGSI UTAMA UNTUK ANGGOTA TIM ---
def process_and_save_datasets(file_name, text_column_name='konten', title_column_name='judul', max_rows=None):
    """
    Membaca satu file CSV, memprosesnya, dan menyimpan hasilnya ke folder CLEAN_DATA_PATH.
    
    Parameters:
        file_name (str): Nama file CSV yang akan diproses (e.g., 'etd-usk.csv').
        text_column_name (str): Nama kolom yang berisi teks utama dokumen (e.g., 'konten').
        title_column_name (str): Nama kolom yang berisi judul dokumen (e.g., 'judul').
        max_rows (int, optional): Jumlah maksimum baris yang akan dibaca. Default None (baca semua).
    """
    input_path = os.path.join(RAW_DATA_PATH, file_name)
    output_path = os.path.join(CLEAN_DATA_PATH, file_name.replace('.csv', '_clean.csv'))

    if not os.path.exists(input_path):
        print(f"ERROR: File input tidak ditemukan di '{input_path}'. Cek folder RAW_DATA_PATH.")
        return

    if not os.path.exists(CLEAN_DATA_PATH):
        os.makedirs(CLEAN_DATA_PATH)
        print(f"INFO: Folder hasil '{CLEAN_DATA_PATH}' dibuat.")

    print(f"\n--- Memulai Pemrosesan {file_name} ---")
    start_time = time.time()
    
    try:
        # Menggunakan engine='python' untuk membaca seluruh baris data dengan robust
        df = pd.read_csv(
            input_path, 
            sep=',', 
            encoding='latin1', 
            engine='python',
            on_bad_lines='warn', # Memberikan peringatan tapi tidak menghentikan
            nrows=max_rows if max_rows else None # Tambahan untuk membatasi baris
        )
    except Exception as e:
        print(f"ERROR membaca {file_name}: {e}")
        return

    # Validasi Kolom
    if text_column_name not in df.columns:
        print(f"ERROR: Kolom teks '{text_column_name}' tidak ditemukan di {file_name}. Cek penamaan kolom!")
        print(f"Kolom yang tersedia: {list(df.columns)}")
        return

    total_rows = len(df)
    processed_counter = 0
    last_percentage_printed = -5
    
    print(f"INFO: Total {total_rows} baris ditemukan. Mulai Normalisasi...")
    
    # Tambahkan Kolom Baru untuk Teks yang Sudah Bersih
    df['clean_content'] = ''
    
    for index, row in df.iterrows():
        # Lakukan Preprocessing
        clean_text = preprocess_text(row[text_column_name])
        df.loc[index, 'clean_content'] = clean_text
        
        processed_counter += 1
        
        # LOGIKA PROGRESS BAR (MENCETAK KELIPATAN 5%)
        current_percentage = int((processed_counter / total_rows) * 100)
        
        if current_percentage >= last_percentage_printed + 5 or processed_counter == total_rows:
            if processed_counter != total_rows:
                last_percentage_printed = current_percentage
                
            print(f"\r  -> Progress {file_name}: {processed_counter}/{total_rows} ({current_percentage}%)", end="", flush=True)

    
    # --- DEMO PREPROCESSING (HANYA JIKA MEMBATASI BARIS) ---
    if max_rows is not None:
        print(f"\n\n--- DEMO HASIL PREPROCESSING (5 Dokumen Pertama) ---")
        demo_df = df[[title_column_name, text_column_name, 'clean_content']].head(5)
        
        for idx, row in demo_df.iterrows():
            print("="*50)
            print(f"JUDUL: {row[title_column_name]}")
            print("-" * 50)
            print(f"ASLI: {row[text_column_name][:100]}...")
            print(f"BERSIH: {row['clean_content'][:100]}...")
        print("="*50)
        
    # Simpan hanya kolom penting ke file CSV bersih
    # Pastikan kolom title_column_name ada untuk disimpan
    cols_to_save = [title_column_name, text_column_name, 'clean_content']
    
    if title_column_name not in df.columns:
        cols_to_save = [text_column_name, 'clean_content'] # Hanya simpan konten jika judul tidak ditemukan
        
    df[cols_to_save].to_csv(output_path, index=False, encoding='utf-8')

    end_time = time.time()
    print(f"\nSUCCESS: {file_name} selesai diproses.")
    print(f"Hasil disimpan di: {output_path}")
    print(f"Waktu total: {end_time - start_time:.2f} detik.")
    
    return True

# --- RUNNING SCRIPT ---
if __name__ == "__main__":
    
    print("\n=============================================")
    print("  SKRIP PREPROCESSING datasets (NON-SASTRAWI)")
    print("=============================================")
    
    # --- PANDUAN PENGGUNAAN TIM ---
    # CARA PENGGUNAAN: 
    #   1. NON-AKTIFKAN (Komentar) semua datasets yang TIDAK menjadi tugas Anda.
    #   2. AKTIFKAN (Hapus komentar) datasets yang menjadi tugas Anda.
    
    MAX_ROWS_TEST = 30 # SET KE 30 BARIS UNTUK UJI COBA CEPAT. SET KE None UNTUK PROSES SEMUA DATA!
    
    datasets_to_run = [
        # AKTIFKAN salah satu (atau lebih) di bawah ini sesuai pembagian tugas:
        'etd_usk.csv',
        # 'etd_ugm.csv',
        # 'kompas.csv',
        # 'tempo.csv',
        # 'mojok.csv',
    ]
    
    # Keterangan: Jika Kolom Teks Anda bukan 'konten' atau Judul bukan 'judul', ganti di bawah ini.
    
    for ds in datasets_to_run:
         # Asumsi Kolom Teks = 'konten' dan Judul = 'judul'
         process_and_save_datasets(ds, text_column_name='konten', title_column_name='judul', max_rows=MAX_ROWS_TEST)

    print("\n--- SEMUA datasets BERSIH TELAH DIBUAT DI FOLDER 'datasets_clean' ---")
    print("Anggota tim yang mengerjakan IR System utama (02_ir_system.py) dapat menggunakan file-file ini.")
