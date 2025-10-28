import pandas as pd
import re
import os
import sys
import csv
import time

# --- MENGAKTIFKAN SASTRAWI (Telah diizinkan) ---
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi Stemmer dan Stopword Remover
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# --- KONFIGURASI UMUM ---
RAW_DATA_PATH = "datasets"
CLEAN_DATA_PATH = "datasets_clean"

# Menambah batas ukuran field (wajib untuk CSV dengan teks panjang)
csv.field_size_limit(sys.maxsize) 

# --- FUNGSI PREPROCESSING ---
def preprocess_text(text):
    """
    Melakukan Case Folding, Cleaning Teks, Stopword Removal, dan Stemming.
    Proses ini diterapkan pada SELURUH karakter input 'text'.
    """
    if pd.isna(text) or text is None:
        return ""

    text = str(text) 
    
    # 1. Case Folding (Wajib)
    text = text.lower()
    
    # 2. PEMBESIHAN NEWLINE DAN TAB (Meratakan paragraf panjang)
    text = re.sub(r'[\n\r\t]', ' ', text)
    
    # 3. PENGHILANGAN KARAKTER NON-STANDAR & ANGKA AGRESIVE
    # Menghilangkan karakter anomali non-ASCII (seperti â, ™, ©)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    
    # 4. Pembersihan Karakter (Hapus angka dan non-huruf, hanya menyisakan huruf (a-z) dan spasi)
    text = re.sub(r'[^a-z\s]', ' ', text) 
    
    # 5. Hapus spasi ganda dan spasi di awal/akhir
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Konversi ke tokens untuk pembersihan awal
    tokens = text.split()
    
    # Pembersihan final sebelum Stemming (menghapus token 1 huruf)
    tokens = [word for word in tokens if len(word) > 1] 
    
    # 6. Stopword Removal (Menggunakan Sastrawi)
    text = stopword_remover.remove(" ".join(tokens))
    
    # 7. Stemming (Menggunakan Sastrawi)
    stemmed_text = stemmer.stem(text)
    
    # 8. Koreksi Over-Stemming (Contoh: 'cuku' -> 'cukup')
    final_tokens = []
    for word in stemmed_text.split():
        if word == 'cuku':
            final_tokens.append('cukup')
        else:
            final_tokens.append(word)

    return " ".join(final_tokens)

# --- FUNGSI UTAMA UNTUK ANGGOTA TIM ---
def process_and_save_dataset(file_name, text_column_name='konten', title_column_name='judul', max_rows=None, enable_demo=True):
    """
    Membaca satu file CSV, memprosesnya, dan menyimpan hasilnya ke folder CLEAN_DATA_PATH.
    
    Parameters:
        file_name (str): Nama file CSV yang akan diproses (e.g., 'etd-usk.csv').
        text_column_name (str): Nama kolom yang berisi teks utama dokumen (e.g., 'konten').
        title_column_name (str): Nama kolom yang berisi judul dokumen (e.g., 'judul').
        max_rows (int, optional): Jumlah maksimum baris yang akan dibaca. Default None (baca semua).
        enable_demo (bool): Kontrol untuk menampilkan output demo perbandingan di terminal.
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
    
    # Logika demo: Hanya tampil jika max_rows dibatasi DAN enable_demo = True
    display_demo = (max_rows is not None) and enable_demo and (total_rows > 0)

    print(f"INFO: Total {total_rows} baris ditemukan. Mulai Normalisasi (Stemming Aktif)...")
    
    # Tambahkan Kolom Baru untuk Teks yang Sudah Bersih
    df['konten_bersih'] = ''
    
    for index, row in df.iterrows():
        # Lakukan Preprocessing
        # clean_text dihitung dari seluruh konten row[text_column_name]
        clean_text = preprocess_text(row[text_column_name])
        df.loc[index, 'konten_bersih'] = clean_text
        
        processed_counter += 1
        
        # LOGIKA PROGRESS BAR (MENCETAK KELIPATAN 5%)
        current_percentage = int((processed_counter / total_rows) * 100)
        
        if current_percentage >= last_percentage_printed + 5 or processed_counter == total_rows:
            if processed_counter != total_rows:
                last_percentage_printed = current_percentage
                
            print(f"\r  -> Progress {file_name}: {processed_counter}/{total_rows} ({current_percentage}%)", end="", flush=True)

    
    # --- DEMO PREPROCESSING (HANYA JIKA MEMBATASI BARIS) ---
    if display_demo:
        print(f"\n\n--- DEMO HASIL PREPROCESSING (5 Dokumen Pertama) ---")
        demo_df = df[[title_column_name, text_column_name, 'konten_bersih']].head(5)
        
        # Mengatur batas tampilan demo ke 100 karakter
        DISPLAY_CHAR_LIMIT = 100 
        
        for idx, row in demo_df.iterrows():
            print("="*50)
            print(f"JUDUL: {row[title_column_name]}")
            print("-" * 50)
            # Tampilkan 100 karakter pertama dari Teks Asli dan Teks Bersih
            print(f"ASLI ({DISPLAY_CHAR_LIMIT} Karakter): {row[text_column_name][:DISPLAY_CHAR_LIMIT]}...")
            print(f"BERSIH (Stemmed, {DISPLAY_CHAR_LIMIT} Karakter): {row['konten_bersih'][:DISPLAY_CHAR_LIMIT]}...")
        print("="*50)
        
    # Simpan hanya kolom penting ke file CSV bersih
    cols_to_save = [title_column_name, text_column_name, 'konten_bersih']
    
    if title_column_name not in df.columns:
        cols_to_save = [text_column_name, 'konten_bersih'] 
        
    df[cols_to_save].to_csv(output_path, index=False, encoding='utf-8')

    end_time = time.time()
    print(f"\nSUCCESS: {file_name} selesai diproses.")
    print(f"Hasil disimpan di: {output_path}")
    print(f"Waktu total: {end_time - start_time:.2f} detik.")
    
    return True

# --- RUNNING SKRIP ---
if __name__ == "__main__":
    
    print("\n=============================================")
    print("  SKRIP PREPROCESSING DATASET (SASTRAWI AKTIF)")
    print("=============================================")
    
    # *** PENTING ***
    # DITETAPKAN KE 30 BARIS UNTUK UJI COBA CEPAT SESUAI PERMINTAAN!
    MAX_ROWS_TEST = None   
    
    # KONTROL DEMO: Atur ke True untuk menampilkan demo 100 karakter di terminal (saat uji coba)
    # Atur ke False untuk hanya melihat progress bar (lebih clean).
    ENABLE_DEMO = True
    
    datasets_to_run = [
        # AKTIFKAN dataset yang ingin diuji coba
        # 'etd_usk.csv',
        'etd_ugm.csv',
        # 'kompas.csv',
        # 'tempo.csv',
        # 'mojok.csv',
    ]
    
    for ds in datasets_to_run:
         # Asumsi Kolom Teks = 'konten' dan Judul = 'judul'
         process_and_save_dataset(ds, text_column_name='konten', title_column_name='judul', max_rows=MAX_ROWS_TEST, enable_demo=ENABLE_DEMO)

    print("\n--- SEMUA DATASET BERSIH TELAH DIUJI (30 Baris) DI FOLDER 'datasets_clean' ---")
    print("Jika hasil demo bagus, ubah MAX_ROWS_TEST = None dan ENABLE_DEMO = False untuk proses penuh.")
