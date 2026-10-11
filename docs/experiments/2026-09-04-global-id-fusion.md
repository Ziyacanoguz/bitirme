# İki kamera global ID prototipi — 4 Eylül 2026

Bu deneyin amacı iki kameradaki geçici ByteTrack kimliklerini aynı oyuncu adayına
bağlamaktır. Radar veya ısı haritası üretmez. Üçüncü maçın 30:00–31:00 aralığında
önceden hesaplanan tespit/takip satırları kullanıldı; bu aşamada YOLO yeniden
eğitilmedi.

İki kameranın saha yönleri ortaklaştırıldı. Oyuncu kutusunun yaklaşık ayak noktası
normalize saha düzlemine taşınıyor; sonra kısa takip parçaları konum devamlılığı,
birden fazla karede destek ve OSNet görünüş benzerliğiyle eşleştiriliyor. Aynı
kamerada aynı anda görülen iki ayrı iz aynı global gruba alınamıyor. Yakın ve
belirsiz alternatifler zorla birleştirilmiyor. Büyük kamera-konum çelişkileri de
tek noktaya ortalanmıyor.

Sahanın gerçek boyutları bilinmediği için koordinatlar 0–1 aralığındadır; metre,
hız veya mesafe iddiası yoktur. Çizgi tabanlı kalibrasyon global ID eşleştirmesinin
iç bağımlılığı olarak korunmuştur:

- `configs/calibration_camera1_lines_v1.json`
- `configs/calibration_camera2_lines_v1.json`
- `src/halisaha/field_projection.py`
- `src/halisaha/line_calibration.py`

Uzun bir yerel ID'nin yanlış kişiye geçmesini bütün dakikaya yaymamak için izler
en fazla 50 karelik parçalara ayrılır. Parça görünüşleri ayrı çıkarılır. Son deney
verisi:

- Kamera 1: 73 yerel ID → 449 kısa parça; 397 görünüş vektörü.
- Kamera 2: 101 yerel ID → 485 kısa parça; 385 görünüş vektörü.
- İki kameradan aynı grup içinde birleştirilen gözlem-kare satırı: 14.280.
- Eşleşmemiş çıktı satırı: 1.977; ayrı tutulan çelişkili satır: 560.

Bu rakamlar doğruluk, IDF1 veya gerçek oyuncu sayısı değildir. İnsan onaylı global
kimlik etiketi bulunmadığı için global ID'ler hâlâ adaydır. Asıl veriler
`outputs/experiments/calibration_fusion_20260904/fusion_v2/fused.csv`, eşleme
tablosu `mapping.csv`, aday kararları `candidates.csv` ve ayarlar `run.json`
dosyalarındadır.

Yeniden üretim sırası:

1. `project_tracks_to_field.py` ile iki kamera CSV'sini saha düzlemine taşı.
2. `segment_camera_tracks.py --block-frames 50 --max-gap 8` ile kısa parçalara ayır.
3. `extract_track_embeddings.py` ile her parça için OSNet görünüşü çıkar.
4. `fuse_camera_tracks.py --min-support 8 --min-fraction .45 --max-distance .10`
   ile ortak aday kimlik verisini üret.

Bir sonraki güvenilir adım, kısa bir klipte gerçek oyunculara elle ortak kimlik
etiketi verip IDF1, ID switch ve yanlış birleştirmeleri ölçmektir.
