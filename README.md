# Türkçe Metinler için Çoklu N-Gram CNN-LSTM Hibrit Duygu Analizi Modeli

Bu proje, Türkçenin eklemeli dil yapısı ve duygu durumlarının genellikle kelime öbekleri (n-gram) içinde saklı olması gerçeğini göz önünde bulundurarak tasarlanmış, paralel işleme yeteneğine sahip hibrit bir derin öğrenme mimarisini içermektedir.

Klasik tek katmanlı modeller yerine, yerel kalıpları (CNN) ve uzun vadeli bağlamsal ilişkileri (LSTM) aynı anda işleyebilen özgün bir yapı kurgulanmıştır.

## Model Mimarisi ve Tasarımı

Model, metindeki anlamsal, yerel ve ardışık bilgileri kayıpsız toplayabilmek için üç ana aşamadan oluşur:

### 1. Özellik Çıkarımı (Embedding ve Paralel CNN)
* **Kelime Gömme (Embedding) Katmanı:** Sayısal dizilere dönüştürülmüş metin verileri, modelin matematiksel uzayda anlamsal yakınlıkları kavrayabilmesi için **128 boyutlu yoğun (dense) vektörlerle** ifade edilir.
* **Çoklu N-Gram CNN Katmanları:** Üç farklı pencere boyutuna (**kernel_size=2, 3 ve 4**) sahip paralel CNN katmanları, ikili (bigram), üçlü (trigram) ve dörtlü (4-gram) kelime öbeklerini aynı anda analiz eder. Bu, "hiç adil değil" gibi duygu yüklü kalıpların yüksek hassasiyetle yakalanmasını sağlar.

### 2. Bağlamsal İşleme ve Boyut Yönetimi (MaxPooling, Concatenate, LSTM)
* **Boyut İndirgeme ve Birleştirme:** Paralel CNN'lerden çıkan öznitelik haritaları, LSTM girişleriyle uyum sağlamak adına `MaxPooling1D (pool_size=2)` ile sıkıştırılır. Ardından `Concatenate` işlemiyle veri kaybı yaşanmadan tek bir matriste birleştirilerek LSTM'in beklediği 3 boyutlu tensör yapısı elde edilir.
* **LSTM (Uzun Kısa Süreli Bellek) Katmanı:** CNN'den elde edilen öznitelikler, Türkçenin gramer yapısından kaynaklanan cümle başı ve sonu arasındaki uzun vadeli bağlamsal ilişkileri kaybetmemek adına ardışık iki LSTM katmanıyla işlenir.

### 3. Sınıflandırma ve Düzenlileştirme (Dropout, Dense)
* **Aşırı Öğrenmeyi (Overfitting) Önleme:** Modelin ezberlemesini engellemek amacıyla, LSTM katmanlarına **%30 oranında dropout ve recurrent_dropout** eklenmiştir. Çıkış katmanından hemen önce fazladan bir genel Dropout işlemi uygulanmıştır.
* **Sınıflandırma:** Model, 7 temel duygu sınıfına (**mutlu, nötr, üzgün, öfkeli, iğrenme, korku ve şaşkınlık**) uygun çıktı verebilmesi için `Softmax` aktivasyon fonksiyonlu Dense katmanı kullanır. Model, **Adam** optimizasyonu ve **categorical_crossentropy** kayıp fonksiyonu ile derlenmiştir.


![Model Katman Mimarisi ve Parametre Özeti](https://github.com/Uygulama-Tasarimi-Projesi/Metin-Analizi-Modeli-CNN-LSTM--Tasarimi-ve-Kodlanmasi/blob/main/Katman-mimarisi-ve-parametre-ozeti.png) 

**Tablo 4: Geliştirilen Hibrit CNN-LSTM Metin Analizi Modelinin Katman Mimarisi ve Parametre Özeti**

*(Not: Yukarıdaki görselde modelin (None, 50) girişinden başlayıp (None, 7) çıkış vektörüne kadar olan tüm boyut değişimleri ve 1.3 milyonu aşkın parametre sayısı doğrulanmıştır.)*

