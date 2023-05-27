import cv2
import numpy as np
import glob
import time
import sys

# Template Matching fonksiyonu
def template_matching(template, image):
    method = cv2.TM_CCOEFF_NORMED
    result = cv2.matchTemplate(image, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val


# Template Matching uygulanacak görsel
image_path = "resim/21.png"
image = cv2.imread(image_path)

# Template dosyalarının bulunduğu dizin
templates_dir = "resim/*.png"

# Template dosyalarını okuma ve benzerlik oranlarını hesaplama
templates = glob.glob(templates_dir)
similarities = []
for template_path in templates:
    template = cv2.imread(template_path)

    # Template'yi 20x20 piksel boyutunda ölçeklendirme
    width = 20
    height = 20
    resized_template = cv2.resize(template, (width, height))



    # Çalışma zamanını hesaplama
    start_time = time.time()  # Başlangıç zaman damgası
    similarity = template_matching(resized_template, image)
    end_time = time.time()  # Bitiş zaman damgası
    elapsed_time = end_time - start_time  # Geçen süreyi hesaplama


    # Bellek kullanımını hesaplama
    template_size = sys.getsizeof(template)
    image_size = sys.getsizeof(image)
    resized_template_size = sys.getsizeof(resized_template)
    total_memory_usage = template_size + image_size + resized_template_size
    similarities.append((template_path, similarity, elapsed_time, total_memory_usage))


# Benzerlik oranlarına göre sıralama
similarities.sort(key=lambda x: x[1], reverse=True)


# Benzerlik oranında Virgülden sonraki kısmı 5 basamağını sayı olarak yazar
#5 de zaten birbirerinin aynısı
def get_decimal_places(number):
    decimal_part = str(number).split('.')[1]  # Ondalık kısmı ayırır
    decimal_places = decimal_part[:5]  # İlk 4 haneli kısmı alır
    return decimal_places


print("Benzerlik yüzdelerini 0.sayı  Şeklinde değil noktadan sonra ilk 5 basamağı gösterecek şekilde sıralattım\n")
# Sonuçları yazdırma
i = 0
for template_path, similarity, elapsed_time, memory_usage in similarities:
    i = i+1
    print(i, template_path, "\nBenzerlik Yüzdeleri: ", get_decimal_places(similarity))
    print("Çalışma Süresi:", elapsed_time, "saniye")
    print()


print()
print("Bellek Kullanımı:", memory_usage, "byte")