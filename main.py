import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_dft(image):
    # Mengubah gambar menjadi skala abu-abu
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Melakukan Discrete Fourier Transform (DFT)
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Menghitung magnitudo spektrum
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    
    return dft_shift, magnitude_spectrum

# Jika d0 (frekuensi pusat) tidak sesuai dengan frekuensi noise, filter mungkin menghapus frekuensi yang tidak seharusnya, yang juga dapat menyebabkan blurring atau tidak efektif dalam menghilangkan noise.
# Jika W (lebar band) terlalu besar, filter akan menghapus lebih banyak frekuensi di sekitar frekuensi pusat d0. Ini dapat menyebabkan hilangnya detail penting pada gambar, yang mengakibatkan gambar menjadi blur.

def apply_band_reject_filter(dft_shift, d0=100, W=50):
    rows, cols = dft_shift.shape[:2]
    crow, ccol = rows // 2, cols // 2

    # Membuat Band Reject Filter
    mask = np.ones((rows, cols, 2), np.uint8)
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if d0 - W/2 < d < d0 + W/2:
                mask[i, j] = 0
    
    # Menerapkan filter pada DFT yang di-shift
    fshift = dft_shift * mask
    
    return fshift

def apply_idft(fshift):
    # Mengembalikan ke domain spasial dengan Inverse DFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    return img_back

def main():
    # Membaca gambar
    image = cv2.imread('image.jpg')

    if image is None:
        print("Gagal memuat gambar. Periksa path dan pastikan file gambar ada.")
    else:
        print("Gambar berhasil dimuat.")
    
    # Menerapkan DFT
    dft_shift, magnitude_spectrum = apply_dft(image)
    
    # Menerapkan Band Reject Filter pada hasil DFT
    fshift = apply_band_reject_filter(dft_shift)
    
    # Mengembalikan gambar yang ditingkatkan menggunakan Inverse DFT
    enhanced_image = apply_idft(fshift)
    
    # Normalisasi hasil untuk menampilkan gambar
    cv2.normalize(enhanced_image, enhanced_image, 0, 255, cv2.NORM_MINMAX)
    enhanced_image = np.uint8(enhanced_image)
    
    # Menampilkan gambar asli, spektrum magnitudo, dan gambar yang ditingkatkan
    plt.figure(figsize=(12, 6))
    plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Magnitude Spectrum')
    plt.subplot(133), plt.imshow(enhanced_image, cmap='gray'), plt.title('Enhanced Image')
    plt.show()

if __name__ == "__main__":
    main()