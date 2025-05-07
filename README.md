
# Laporan Proyek Machine Learning - Mifta Rizaldirahmat

## Project Overview

Buku adalah jendela pengetahuan. Tanpa buku, kita akan kesulitan dalam memahami berbagai hal. Buku hadir dalam beragam jenis seperti Novel, Ensiklopedia, Biografi, Sains, dan lainnya. Buku berperan sebagai sarana penyampaian informasi dari penulis ke pembaca. Manfaat dari membaca buku antara lain memperluas wawasan, memperdalam ilmu pengetahuan, dan menjadi panduan dalam kehidupan. Proyek ini bertujuan membangun sistem rekomendasi buku berdasarkan bacaan sebelumnya.

## Business Understanding

Setiap orang memiliki preferensi bacaan yang unik. Misalnya, ada yang menyukai buku karena penulisnya, genrenya, atau bahkan desain sampulnya. Proyek ini bertujuan menciptakan sistem rekomendasi berdasarkan minat pengguna terhadap buku.

### Problem Statements

Masalah yang ingin diselesaikan dalam proyek ini antara lain:
- Bagaimana cara memberikan rekomendasi buku berdasarkan buku yang telah dibaca pengguna?
- Bagaimana cara membangun sistem rekomendasi yang mempertimbangkan penilaian (rating) buku?

### Goals

Tujuan dari proyek ini adalah:
- Mengembangkan metode rekomendasi berdasarkan buku yang telah dibaca pengguna.
- Membuat sistem rekomendasi buku yang memanfaatkan data rating.

### Solution statements

Proyek ini memanfaatkan dua metode algoritma *Machine Learning*:
1. **Content Based Filtering**, memberikan rekomendasi berdasarkan kemiripan item dengan riwayat bacaan pengguna, misalnya merekomendasikan buku bergenre horor jika sebelumnya membaca buku horor.
2. **Collaborative Filtering**, mengandalkan opini dari pengguna lain tanpa perlu atribut konten buku.

## Data Understanding

Dataset yang digunakan diambil dari situs Kaggle dan dapat diakses melalui tautan [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Data yang digunakan mencakup **Books.csv** dan **Ratings.csv**.

Deskripsi data **Books.csv**:
- ISBN: Nomor unik buku
- Book-Title: Judul buku
- Book-Author: Nama penulis
- Year-of-Publication: Tahun terbit
- Publisher: Nama penerbit
- Image-URL-S/M/L: Tautan gambar sampul ukuran kecil/sedang/besar

Deskripsi data **Ratings.csv**:
- User-ID: ID pengguna
- ISBN: Nomor buku
- Book-Rating: Nilai penilaian buku

### Univariate Exploratory Data Analysis

**Books**

Tabel informasi data book:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 271360 entries, 0 to 271359
```

Jumlah:
- Buku unik: 242135
- Penulis unik: 102024

**Ratings**

Tabel informasi data rating:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1149780 entries, 0 to 1149779
```

- Jumlah user unik: 105283
- Skor rating bervariasi dari 0 sampai 10

## Data Preparation

Proyek ini menggunakan subset data dengan 5000 buku dan 1000 rating:

```python
book = book[:5000]
rating = rating[:1000]
```

**Content Based Filtering** menggunakan langkah berikut:
- Menghapus nilai kosong (NA)
- Menghapus baris duplikat
- Mengonversi data book menjadi list
- Membuat struktur dictionary dari list

**Collaborative Filtering** menggunakan langkah:
- Melakukan encoding terhadap User-ID dan ISBN
- Memetakan kembali ke data frame
- Mengecek ukuran dan mengonversi rating menjadi float
- Melatih data dengan membagi train-test 82%:18%

## Modeling and Results 

### Content Based Filtering

- Menggunakan TF-IDF Vectorizer pada kolom penulis buku
- Hasil transformasi ke dalam bentuk matriks: `(5000, 3538)`
- Cosine Similarity menghasilkan matriks `(5000, 5000)`
- Uji coba dilakukan dengan buku *Classical Mythology* oleh Mark P.O. Morford

Rekomendasi yang dihasilkan berdasarkan kata kunci "Mark":

| Judul Buku | Penulis |
|------------|---------|
| Fishboy: A Ghost's Story | Mark Richard |
| The Diaries of Adam and Eve | Mark Twain |
| Adventures of Huckleberry Finn | Mark Twain |
| A Connecticut Yankee in King Arthur's Court | Mark Twain |
| Adventures of Huckleberry Finn | Mark Twain |

### Collaborative Filtering

- Melakukan embedding menggunakan *RecommenderNet*
- Kompilasi dengan *BinaryCrossentropy*, *Adam Optimizer*, dan evaluasi dengan RMSE

Hasil Top 2 rekomendasi buku:

| Judul Buku | Penulis |
|------------|---------|
| Tales of the City (Tales of the City Series, V. 1) | Armistead Maupin |
| Nocturne indien | Antonio Tabucchi |

## Evaluation

Pada metode **Content Based Filtering**, digunakan metrik presisi. Dari 5 rekomendasi, tidak ada yang sesuai dengan penulis buku yang sudah dibaca, menghasilkan presisi 0%:

```python
Accuracy = real_author/5*100
print("Accuracy of the model is {}%".format(Accuracy))
```

Output:
```
Accuracy of the model is 0.0%
```
 
Sedangkan untuk **Collaborative Filtering**, evaluasi menggunakan RMSE (Root Mean Square Error). Semakin kecil nilai RMSE, maka prediksi semakin akurat.

Plot berikut menunjukkan penurunan RMSE untuk data training dan testing:

![MSE](https://github.com/user-attachments/assets/adb1209e-f290-4d34-91b7-123c50fa9ed2)


**Kesimpulan:** Penurunan nilai RMSE menunjukkan bahwa model Collaborative Filtering bekerja dengan cukup baik.
