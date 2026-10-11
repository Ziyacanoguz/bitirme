# Halı saha projesi — devam notu

Son doğrulama: 8 Eylül 2026, akşam. Bu not eski Codex görevinin bağlam hatasından
sonra ilgili geçmişin parça parça okunması ve mevcut dosyaların denetlenmesiyle
hazırlandı. Kaynak görev: `01a01f43-a635-78f0-960b-b44623fa06cc` — Halı saha projesi.

## Çalışma alanı ve korunan kararlar

- Asıl kod: `/Users/ziyacan/Desktop/project`; çalıştırıcı `.venv-boxmot/bin/python`.
- Mevcut kapsam: oyuncu tespiti → kamera içi takip → Re-ID → Global ID.
- Kullanıcı isteğiyle ısı haritası ve kuş bakışı radar kaldırılmıştır.
- Takımlar sabit beyaz/mavi gibi renk sınıflarıyla ayrılmaz; yelek/rengi yardımcı
  sinyal olarak değerlendirmek mümkündür. Kimlikler 16 kişiye zorlanmaz.
- Gerçek saha ölçüleri bilinmiyor. Saha koordinatları normalize; metre/hız iddiası yok.
- Ana model: `outputs/training/yolo26n_halisaha_preserve_pilot/weights/best.pt`.
  200/407 onaylı görüntü adayları geçmiş deneylerde net üstünlük göstermediği için
  ana model değiştirilmemişti. Bu devam çalışmasında yeniden eğitim yapılmadı.
- Ana/crop giriş 1920/1280; confidence 0.15; ByteTrack track_thresh 0.30,
  match_thresh 0.80, track_buffer 75. Re-ID: `models/osnet_x0_25_msmt17.pt`.
- Çizgi kalibrasyonları: `configs/calibration_camera1_lines_v1.json` ve
  `configs/calibration_camera2_lines_v1.json`. Eski tahmin edilen köşelerin başarılı
  yeniden izdüşümü gerçek konum doğruluğu sayılmamalıdır.

## Kaynaklar

Üç maçın iki kamera kaydı `assets/videos/matches/match_01..03` içindedir. Bu turda
Match 03 kaynakları yeniden açıldı: 1920×1080, yaklaşık 25 FPS, 88749/88750 kare.
Match 03'ün 30:00–31:00 aralığı mevcut doğrulama ve inceleme örneğidir.

Geçmiş kararlara göre `datasets/halisaha_player_v2` toplam 680 görüntü içerir;
train/val Match 01–02, bağımsız test Match 03 olarak ayrılmıştır. Bu turda veri seti
ve eğitim metrikleri yeniden ölçülmedi. Ön etiketler insan onayı yerine geçmez.
Eğitim ayrıntıları `docs/experiments/2026-09-04-reviewed407-training.md` içindedir.

## Bu devam çalışmasında tamamlananlar

1. Görüntü etiketleri için önceki görevde hazırlanmış koyu zeminli, çakışmayı azaltan
   çizim koduyla 5 ve 60 saniyelik videolar yeniden üretildi. Eşleştirme kararı
   değiştirilmedi; belirsiz/çelişkili gözlemler kamera içi `C1:L…` / `C2:L…` taşır.
2. İnceleme paketindeki tarayıcı kayıtları artık kaynak, segment, kesin kare ve
   gösterilen iki JPEG'in SHA-256 değerlerine bağlı. Aynı segmentin farklı karesine
   eski karar taşınmaz. CSV'de `review_source_id` ve `review_evidence_id` bulunur.
3. Hiç kameralar arası aday bulunmayan geçerli fusion çalışması artık başlıkları
   olan boş candidates.csv ile tamamlanır. Kalibrasyona uygun konum yoksa boş
   fused.csv ve tanısal rapor üretilir; önizleme yerel ID'lerle çalışabilir.
4. `--sync-method assumed_offset` eklendi. `manual_event` ve `assumed_offset`
   boş olmayan `--sync-note` gerektirir.
5. Gerçek 5 saniyelik iki kamera hattı yeniden çalıştırıldı:
   `outputs/runs/global_id_resume_20260908_5s/pipeline.json`.
   Sekiz aşama tamamlandı, iki kamerada 125 ortak kare; kaynak CSV'ler 1272/1591 satır.
   Yeni fused.csv, mapping.csv ve candidates.csv önceki 5 saniyelik çalışmayla
   SHA-256 düzeyinde aynı. Bu, düzeltmelerin mevcut eşleştirmeyi değiştirmediğini
   gösterir; kimlik doğruluğu ölçümü değildir.

## Doğrulama

`.venv-boxmot/bin/python -m unittest discover -s tests`: 89 test geçti.
Yeni regresyonlar boş fusion çıktıları, senkron yöntem/not zorunluluğu, tam kanıta
bağlı kayıt yükleme, eski kanıt reddi ve CSV kaçış/bozuk depolama davranışını kapsar.

60 saniyelik H.264 çıktı 1500 kare ve 1920×592; FFprobe ile tüm kareleri çözüldü,
örnek kareler görsel olarak ve video Codex tarayıcısında oynatılarak kontrol edildi.
40 kartın 80 görselinin dosya hash'leri doğrulandı. Ayrı TEST kopyasında karar/not
girildi, yenileme sonrası korundu ve indirilen CSV okunarak 40 satır/tek test kararı
doğrulandı. Gerçek teslim paketinde bütün kararlar boş bırakıldı.

## Teslim dosyaları

Dosyalar `/Users/ziyacan/Documents/Codex/2026-09-08/ha/outputs` içindedir:

- `global_id_60s.mp4`: Match 03, 30:00–31:00; mevcut 60 saniyelik v3 fusion.
- `global_id_5s.mp4`: 30:00–30:05 kısa işlem hattı kontrolü.
- `global_id_preview.jpg`: 60 saniyelik videodan örnek kare.
- `global_id_review/index.html`: 60 saniyelik çalışmadan 40 riskli kabul edilmiş aday.
- `global_id_review.zip`: index.html ve images klasörü birlikte taşınabilir paket.
- `proje_devam_notu.md`: bu notun teslim kopyası.
- `verification.json`: kaynaklar, kod hash'leri ve doğrulama özeti.

ZIP açıldıktan sonra index.html tarayıcıda açılır. Her kart için aynı/farklı/kararsız
seçilir; not/kişi etiketi isteğe bağlıdır. Otomatik kayıt aynı tarayıcıya aittir;
paylaşmak için “Sonuç CSV'sini indir” kullanılmalıdır. Riskli kabul edilmiş adayların
seçilmesi örnekleme yanlılığı taşır; bu 40 karttan tüm sistem doğruluğu çıkarılmaz.

İki video farklı uzunluktaki bağımsız fusion çalışmalarını gösterir; aynı G numarası
iki ayrı çalışmada aynı gerçek oyuncuyu ifade etmek zorunda değildir. İnceleme
paketi 60 saniyelik videoya aittir. 60 saniyelik kaynak fusion:
`outputs/experiments/global_id_quality_20260908_60s`; segment girdileri
`outputs/experiments/calibration_fusion_20260904/camera{1,2}_segments.csv`.

## Kalan gerçek doğrulama işi

Sıfır kare ofseti bu çalışmada varsayım olarak kaydedildi; ortak bir görsel/ses olayı
ile henüz doğrulanmadı. Mevcut hareket ofset denetimi en az 100 kare aday desteği
isterken varsayılan segmentler 50 karedir. Boş ofset raporu senkron kanıtı değildir.

Sonraki somut adım: aynı klipte ortak olayın iki kamera karesini kaydetmek ve
inceleme paketini ekipçe etiketlemek; ardından küçük bir klipte gerçek oyuncu
kimlikleriyle zamansal referans veri hazırlamak. HOTA/IDF1/ID switch ancak uygun
referansla ölçülmelidir. Tek kare kararı tüm iz kimliğini doğrulamaz. Örtüşme,
kaçırılan uzak oyuncu ve ID kopması sürüyor; model/eşik değişikliği bu sabit
referans üzerinde tek değişkenli deneyle değerlendirilmelidir.

Beş kişilik ekip için önceki modül önerisi: (1) detector/tracking, (2) senkron/Global ID,
(3) top/olay/highlight, (4) backend/veri tabanı, (5) uygulama/entegrasyon.
Son üç bölüm bu turda uygulanmadı.
