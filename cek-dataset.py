import pandas as pd
import os

def check_csv_columns(file_path):
    """Membaca CSV dan menampilkan daftar semua nama kolom."""
    if not os.path.exists(file_path):
        print(f"Error: File tidak ditemukan di {file_path}")
        return
    try:
        # Membaca hanya beberapa baris pertama untuk efisiensi
        df = pd.read_csv(file_path, nrows=5) 
        print(f"\nNama-nama kolom yang ditemukan di {os.path.basename(file_path)}:")
        # Tampilkan kolom dan contoh isinya
        for col in df.columns:
            # Ambil contoh data non-NaN pertama
            sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else "No Data"
            print(f"  - Kolom: '{col}' (Contoh data: {str(sample_value)[:50]}...)")

    except Exception as e:
        print(f"Gagal membaca file {file_path}: {e}")

# Ubah path di bawah sesuai dengan nama file Anda
check_csv_columns('dataset/etd_usk.csv')
# Anda bisa tambahkan file lain untuk pengecekan
# check_csv_columns('datasets/kompas.csv') 
