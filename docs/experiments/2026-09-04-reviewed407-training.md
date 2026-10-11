# 407 onaylı görüntüyle oyuncu tespiti eğitimi — 4 Eylül 2026

Kullanıcının kaydettiği 407 onaylı görüntüden denetlenebilir bir veri anlık kopyası
oluşturuldu. Bölünmede 359 eğitim, 32 doğrulama görüntüsü ve 5.145 eğitim oyuncu
kutusu bulunur. Aynı maç anının diğer kamera görüntüsünün doğrudan doğrulamadan
eğitime sızmaması için zamansal tampon kullanıldı. Bununla birlikte doğrulama aynı
sahadan geldiği için bağımsız maç testi değildir.

Mevcut YOLO26n ağırlığından 12 epoch, 960 giriş boyutu, batch 4, AdamW ve Apple
MPS ile yeni aday eğitildi. Eski 20 görüntülük testteki sonuçlar:

| Model | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Mevcut preserve pilot | %89,07 | %78,52 | %87,88 | %57,13 |
| Önceki 200 görüntü pilotu | %86,88 | %79,98 | %88,96 | %58,45 |
| Yeni 407 görüntü adayı | %89,95 | %78,52 | %87,91 | %56,80 |

Yeni aday mevcut modeli açık biçimde geçmediği için varsayılan model
değiştirilmedi. Ham karşılaştırma
`outputs/experiments/reviewed407_20260904/eval_old_test20/comparison.json`
dosyasındadır. Yeni model
`outputs/training/yolo26n_halisaha_reviewed407_20260904/weights/best.pt`, korunan
varsayılan ise
`outputs/training/yolo26n_halisaha_preserve_pilot/weights/best.pt` yolundadır.

Üçüncü maçın 30:00–31:00 aralığında tam görüntü 1920, uzak bölge 1280, güven
eşiği 0,15 ve ByteTrack tamponu 75 kareyle takip edildi. Benzersiz ID sayıları
oyuncu sayısı veya doğruluk değildir: bir oyuncu birkaç ID'ye bölünebilir veya
bir ID birden fazla oyuncuya geçebilir. Tespit, takip, Re-ID ve ID birleştirme
çıktıları `outputs/experiments/reviewed407_20260904` altında korunur. Isı haritası
ve radar kodu/çıktıları 5 Eylül'de kullanıcı isteğiyle projeden kaldırılmıştır.
