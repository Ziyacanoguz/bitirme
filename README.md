# Halı Saha Video Analizi

## 27 Eylül 2026: arkadaş veri setinin entegrasyonu

Farklı maçlardan gelen Roboflow verisi, mevcut onaylı anlık kopyayla
`datasets/halisaha_merged_v1` altında birleştirildi: 841 train, 133 validation,
65 test; toplam 1.039 görüntü ve 14.847 oyuncu kutusu. Orijinal kaynaklar ve
mevcut model korundu. İki karışık etiket dosyasındaki 15 çokgen kutuya çevrildi;
etiketi boş iki oyunculu görüntü dışarıda bırakıldı. Eğitim başlatılmadı.
Kaynak bilgisi, inceleme durumu ve kullanım komutları
[birleşik veri seti notunda](datasets/halisaha_merged_v1/README.md).
Yeni eğitim denetimi, seçilen YAML'daki gerçek görüntüleri manifestle eşleştirir;
`--dry-run` veri bağlantısını eğitim başlatmadan kontrol eder.

İki sabit kamera kaydından oyuncu tespiti, object tracking, ortak saha düzleminde
yörünge üretimi ve daha sonra top/olay/highlight analizi geliştirmek için kullanılan
proje alanıdır.

## 8 Eylül akşamı: yarım kalan çalışmaya devam

Eski Codex görevinin bağlam hatasından sonra son kararlar ve dosyalar doğrulandı.
Güncel başlangıç noktası, test sonuçları ve teslim bağlantıları
[devam notunda](docs/CONTINUE_HERE.md). İnceleme kayıtlarının yanlış görüntüye
taşınması ve sıfır kamera eşleşmesinde çalışma hatası düzeltildi; 89 test geçti.
Okunaklı 5/60 saniyelik Global ID videoları ve yeni 40 örneklik inceleme paketi hazır.

## 4 Eylül 2026: ara eğitim ve global ID deneyi

407 kaydedilmiş onaylı görüntüyle yeni bir YOLO26n adayı eğitildi. Mevcut ve önceki
200 görüntülük pilotla ölçüldü; açık bir üstünlük görülmediğinden eski model
değiştirilmedi. İki kameranın üçüncü maçtaki 30:00–31:00 aralığı için oyuncu
tespiti, kamera içi takip ve deneysel kameralar arası global ID verileri hazırlandı.

Ayrıntılar, ölçümler ve sınırlamalar:
[407 görüntü eğitim deneyi](docs/experiments/2026-09-04-reviewed407-training.md) ve
[global ID birleştirme deneyi](docs/experiments/2026-09-04-global-id-fusion.md).

- Çıktılar: `outputs/experiments/reviewed407_20260904`.
- Tek komutluk analiz: `.venv-boxmot/bin/python scripts/run_player_analysis.py --help`.
- Global ID'ler henüz insan tarafından doğrulanmış oyuncu kimlikleri değildir.
- Deneysel `match-thresh=0.85` daha az ID parçası üretti; yanlış eşleşme kontrolü
  yapılmadan varsayılan 0.80 değiştirilmedi.
- Eğitimde açıkça verilen öğrenme hızının yok sayılmaması için optimizer seçimi
  düzeltildi. Eski komutların öğrenme hızı artık gerçekten uygulanır.

## 8 Eylül 2026: takip ve Global ID sertleştirmesi

Ana kare ile uzak saha crop'undan gelen tespitleri birleştiren kod ayrıştırılıp
doğrulandı; koordinat dönüşümü ve deterministik NMS artık bağımsız test
edilebiliyor. Takip CSV'leri için kısa ID, iç boşluk, ani konum sıçraması, yüksek
IoU ve olası parçalanma bağlantılarını raporlayan ground-truth'suz kalite denetimi
eklendi. Bu göstergeler doğruluk ölçümü değildir.

İki kamera hattında saha izdüşümü, kısa parçalara ayırma, OSNet Re-ID ve temkinli
Global ID v3 tek komutla yeniden üretilebilir hale getirildi. V3; ardışık destek,
konum farkının p95 değeri, görünüş eksikliği ve bileşen içi çelişkiler için ek
korumalar taşır. Eşleşme puanları olasılık veya ölçülmüş doğruluk değildir.

Ayrıntılı değişiklikler ve gerçek A/B kayıtları:
[takip ve Global ID sertleştirme raporu](docs/experiments/2026-09-08-tracking-hardening.md).
Mevcut model, confidence ve tracker varsayılanları başarı kanıtı olmadan
değiştirilmedi. Isı haritası ve kuş bakışı radar geri alınmış durumda kalır.

## Mevcut durum

- İki 1080p/25 FPS kamera videosu bulunuyor.
- Videoları yan yana ve kare bazında inceleyen bir senkronizasyon prototipi var.
- YOLO26n ile `person` sınıfı için oyuncu tespiti çalışıyor.
- ByteTrack tabanlı object tracking ve saha ROI desteği çalışıyor.
- BoxMOT ayrı bir Python 3.12 ortamına kuruldu; tracker karşılaştırma altyapısı hazır.
- Önceki 7 noktalı TPS kalibrasyonu korunuyor; çizgi tabanlı ortak yönlü kalibrasyon adayı eklendi.
- Konum, hareket geçmişi ve OSNet görünüşüyle global ID prototipi çalışıyor.
- Kamera içindeki kopuk kısa tracklet'leri temkinli biçimde birleştiren bir ön aşama eklendi.
- Yerel tracking CSV'leri için doğruluk iddiası taşımayan tanısal kalite raporu üretiliyor.
- Tek kamera analiz hattı ham ve birleştirilmiş izleri otomatik olarak denetliyor.
- İki kamera Global ID hattı kaynakları ve aşamaları bir manifestte kaydediyor.
- Global ID adayları, provenansı fusion çalışmasına bağlı bir HTML arayüzünde insan tarafından incelenebiliyor.
- Isı haritası ve kuş bakışı radar üretimi geri alınmıştır.

## Klasör yapısı

```text
assets/videos/       Ham kamera kayıtları
configs/             Ortak deney ve model ayarları
docs/                Teknik notlar ve deney dokümantasyonu
models/              Model ağırlıkları
outputs/detections/  Tespit çıktıları
outputs/tracks/      Takip çıktıları
outputs/tracks_tiled/ Uzak saha ikinci tarama çıktıları
outputs/reports/     Ölçüm ve karşılaştırma raporları
outputs/audits/      Ground-truth'suz takip risk göstergeleri
outputs/reviews/     İnsan Global ID inceleme paketleri
outputs/runs/        Yeni uçtan uca çalışma klasörleri
outputs/boxmot/      BoxMOT video ve MOT-formatlı takip çıktıları
scripts/             Çalıştırılabilir prototipler
src/halisaha/        Yeniden kullanılabilir proje kodu
third_party/         Açık kaynak algoritma/lisans notları
```

## Kurulum

Python 3.10 veya daha yeni bir sürüm önerilir.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
## Test çalıştırma

Projedeki tüm birim ve regresyon testlerini otomatik ortam algılamasıyla çalıştırmak için:

```bash
python3 scripts/run_tests.py
```

Detaylı test çıktısı almak için:

```bash
python3 scripts/run_tests.py -v
```

## Mevcut araçlar

Oyuncu tespiti:

```bash
python scripts/detect_players.py
```

Farklı video/model veya eşik vermek mümkündür:

```bash
python scripts/detect_players.py \
  --video assets/videos/camera2.mp4 \
  --model models/yolo26n.pt \
  --confidence 0.20 \
  --image-size 1280
```

İki kamera senkron incelemesi:

```bash
python scripts/sync_test.py
```

30 saniyelik ByteTrack denemesi:

```bash
python scripts/track_players.py \
  --video assets/videos/camera1.mp4 \
  --tracker bytetrack \
  --start 300 \
  --duration 30
```

Sanal ortam etkin değilse aynı komut `.venv/bin/python` ile çalıştırılabilir.

Sonuç videosu ve kare bazlı CSV verisi `outputs/tracks/` altında oluşturulur.
Önizleme penceresi istenirse komuta `--show` eklenebilir. Aynı klibi BoT-SORT ile
çalıştırmak için yalnız `--tracker botsort` kullanılmalıdır.

## Tracking karşılaştırma planı

İlk aşamada aynı YOLO çıktıları üzerinde iki tracker karşılaştırılacaktır:

1. **ByteTrack:** düşük güvenli tespitleri ikinci association aşamasında kullanır;
   hızlı ve açıklanabilir baseline'dır.
2. **BoT-SORT:** hareket, association ve gerektiğinde görünüş/Re-ID bilgisini
   birleştirir; örtüşme ve kimlik devamlılığı için karşılaştırma yöntemidir.

DeepSORT ve StrongSORT, ilk iki baseline ölçüldükten sonra ikinci aşama deneyleri
olarak değerlendirilecektir. Kaynak ve lisans bilgileri `third_party/README.md`
dosyasındadır.

## Saha ROI oluşturma

Her kamera için saha sınırı bir kez seçilir:

```bash
.venv/bin/python scripts/select_roi.py \
  --video assets/videos/camera1.mp4 \
  --output configs/roi_camera1.json \
  --time 300
```

Saha köşeleri ve gerekirse kenar çizgileri sırayla sol tıkla seçilir. `U` son
noktayı geri alır, `R` bütün seçimi temizler, `Enter` veya `S` kaydeder.

ROI ve halı saha ByteTrack profiliyle takip:

```bash
.venv/bin/python scripts/track_players.py \
  --video assets/videos/camera1.mp4 \
  --tracker bytetrack \
  --tracker-config configs/bytetrack_halisaha.yaml \
  --roi configs/roi_camera1.json \
  --start 300 \
  --duration 30 \
  --show
```

Tracker karşılaştırmasında detector, video bölümü, confidence eşiği ve giriş
çözünürlüğü sabit tutulmalıdır. Değerlendirme için HOTA, IDF1, ID switch,
fragmentation ve işlem hızı raporlanacaktır.

## BoxMOT deneyi

BoxMOT, sistem Python'undan bağımsız olarak `.venv-boxmot` ortamına kurulmuştur.
Kurulumu doğrulayan kısa ByteTrack testi:

```bash
.venv-boxmot/bin/python -m boxmot.engine.cli track \
  --source assets/videos/clips/camera1_300s_5s.mp4 \
  --detector models/yolo26n.pt \
  --tracker bytetrack \
  --classes 0 --imgsz 1280 --conf 0.15 \
  --device cpu --batch-size 1 --no-auto-batch \
  --project outputs/boxmot --save --save-txt --show-timing
```

BoT-SORT + OSNet Re-ID deneyi için `models/osnet_x0_25_msmt17.pt` ağırlığı
yerleştirildikten sonra `--tracker botsort --reid models/osnet_x0_25_msmt17.pt`
kullanılacaktır. Deneylerin adil olması için aynı video, detector, çözünürlük ve
confidence değeri korunmalıdır.

## Uzak saha için ikinci tarama

Uzak oyuncuları büyütmek için tüm videoya yapay zekâ ile super-resolution
uygulamak yerine ana kare ve uzak saha crop'u ayrı ayrı tespit edilir. Kutular
orijinal koordinatlara çevrilir, NMS ile yinelenen kutular kaldırılır ve birleşik
tespitler ByteTrack'e verilir:

```bash
.venv-boxmot/bin/python scripts/track_players_tiled.py \
  --start 1800 --duration 30 \
  --confidence 0.10 --image-size 1280 --crop-image-size 1280
```

Kamera 1'in uzak tarama dikdörtgeni `configs/far_crop_camera1.json`, saha maskesi
ise `configs/roi_camera1.json` içindedir. Camgöbeği dikdörtgen videoda ikinci kez
taranan alanı, sarı çizgi saha ROI'sini gösterir.

## Yakın geliştirme sırası

1. Aynı olayı iki kamerada elle işaretleyip gerçek senkronizasyon kanıtını kaydetme
2. İki kameranın yerel tracking çıktılarını kalite denetiminden geçirme
3. Temkinli Global ID v3 çalışmasını yeni bir output klasöründe üretme
4. Riskli kamera eşleşmelerini HTML inceleme paketinde doğru/yanlış/kararsız etiketleme
5. Küçük ama gerçek kimlik etiketli bir doğrulama klibi hazırlama
6. Yalnız bu veri üzerinde HOTA, IDF1, ID switch ve fragmentation ölçme
7. Sonuçlara göre tracker/Re-ID/eşikleri tek değişkenli A/B deneyleriyle ayarlama
8. Oyuncu kimliği yeterince kararlı olduktan sonra top ve olay modülünü bağlama

## İki kamera saha kalibrasyonu

Global ID üretmeden önce iki kamerada aynı dört fiziksel saha köşesi aynı sırayla
işaretlenir. Buradaki sol ve sağ, ekrandaki yön değil gerçek sahanın sabit yönüdür:

1. Sol kale tarafı - üst kenar
2. Sağ kale tarafı - üst kenar
3. Sağ kale tarafı - alt kenar
4. Sol kale tarafı - alt kenar

```bash
.venv-boxmot/bin/python scripts/calibrate_field.py \
  --video assets/videos/camera1.mp4 \
  --output configs/calibration_camera1.json \
  --time 1800

.venv-boxmot/bin/python scripts/calibrate_field.py \
  --video assets/videos/camera2.mp4 \
  --output configs/calibration_camera2.json \
  --time 1800
```

Sol tık nokta ekler, `U` son noktayı geri alır, `R` temizler ve dört noktadan
sonra `Enter` kaydeder. Kamera 2'de saha ekranda ters görünse bile fiziksel nokta
adlarının sırası değiştirilmemelidir.

Bir track CSV'sini ortak normalize saha koordinatlarına dönüştürme örneği:

```bash
.venv-boxmot/bin/python scripts/project_tracks_to_field.py \
  --tracks outputs/tracks_tiled/camera1_1800s_30s/camera1_tiled_1280_0.1.csv \
  --calibration configs/calibration_camera1.json \
  --video assets/videos/camera1.mp4 \
  --output outputs/tracks_tiled/camera1_1800s_30s/camera1_field.csv
```

## Global ID prototipi

İki kameranın saha koordinatları önce konum, hareket yönü ve geçmiş eşleşme
hafızasıyla birleştirilir. İsteğe bağlı OSNet dosyaları görünüş benzerliğini
yumuşak bir maliyet olarak ekler:

```bash
.venv-boxmot/bin/python scripts/associate_cameras_temporal.py \
  --camera1 outputs/global_id_test/camera1/camera1_field.csv \
  --camera2 outputs/global_id_test/camera2/camera2_field.csv \
  --reid1 outputs/global_id_test/camera1/reid_embeddings.npz \
  --reid2 outputs/global_id_test/camera2/reid_embeddings.npz \
  --output-dir outputs/global_id_test/association_temporal_reid
```

Global eşleştirme puanı; ortak saha konumu, saha üzerindeki hız/yön, önceki
kamera eşleşmesine verilen devamlılık bonusu ve OSNet kosinüs uzaklığından oluşur.
Hazır OSNet küçük ve benzer formalı oyuncularda tek karar verici değildir.

Kamera içinde kısa süre kaybolup yeni ID alan tracklet'leri birleştirme örneği:

```bash
.venv-boxmot/bin/python scripts/stitch_tracklets.py \
  --tracks outputs/global_id_30s/camera1/camera1_field.csv \
  --reid outputs/global_id_30s/camera1/reid_embeddings.npz \
  --output-tracks outputs/global_id_30s/camera1/camera1_field_stitched.csv \
  --output-reid outputs/global_id_30s/camera1/reid_embeddings_stitched.npz \
  --mapping outputs/global_id_30s/camera1/tracklet_mapping.csv \
  --max-gap 50 --max-position 0.07 --max-appearance 0.32
```

Bu işlem gerçek oyuncu sayısını zorla 16'ya indirmez. Yalnızca zamanda çakışmayan,
saha üzerinde birbirine yakın devam eden ve görünüşü yeterince benzeyen izleri
birleştirir. Böylece farklı oyuncuları yanlışlıkla tek ID yapmak yerine şüpheli
izler ayrı bırakılır.

30 saniyelik ilk iki kamera deneyi ve yorumları
`docs/experiments/2026-08-28-global-id-30s.md` dosyasındadır.

## Halisaha oyuncu detector egitimi

BoxMOT ortamını bozmadan ayrı eğitim ortamı kullanılır:

```bash
source .venv-train/bin/activate
python -c "import torch; print(torch.backends.mps.is_available())"
```

Pilot veri seti iki kameranın birbirinden ayrılmış zaman aralıklarından 80 train,
20 validation ve 20 test görüntüsü içerir. Hazır YOLO kutuları yalnızca ön etikettir;
eğitimden önce manuel kontrol edilmelidir:

```bash
.venv-train/bin/python scripts/review_yolo_labels.py --split train
.venv-train/bin/python scripts/review_yolo_labels.py --split val
.venv-train/bin/python scripts/review_yolo_labels.py --split test
```

Etiket kontrol ekranında sol fare sürüklemesi eksik oyuncu kutusu ekler, bir kutu
içindeki sağ tık o kutuyu siler. `N`, sağ ok veya Enter kaydedip sonraki görüntüye;
`P` veya sol ok önceki görüntüye gider. `Q` programdan çıkar.

Train ve validation görüntülerinin tamamı kontrol edilmeden eğitim komutu güvenlik
gereği çalışmaz. Bu küçük veri setinde `data.yaml` ile tek sınıflı eğitim yapmak
hazır COCO çıkış katmanını yeniden kurduğu için önerilmez. Oyuncu etiketleri yine
`class 0` olarak kalır; fakat hazır person bilgisini korumak için
`data_coco80.yaml`, düşük öğrenme hızı ve dondurulmuş ilk 10 katman kullanılır.

Önce tek epoch smoke test yapılır:

```bash
.venv-train/bin/python scripts/train_player_detector.py \
  --data-yaml datasets/halisaha_player/data_coco80.yaml \
  --epochs 1 --image-size 960 --batch 4 --device mps \
  --learning-rate 0.0005 --freeze 10 \
  --name yolo26n_halisaha_preserve_smoke
```

Smoke test başarılıysa pilot eğitim:

```bash
.venv-train/bin/python scripts/train_player_detector.py \
  --data-yaml datasets/halisaha_player/data_coco80.yaml \
  --epochs 30 --image-size 960 --batch 4 --device mps \
  --learning-rate 0.0005 --freeze 10 \
  --name yolo26n_halisaha_preserve_pilot
```

İlk korumalı pilotta en iyi ağırlık 24. epoch'ta seçildi. 20 validation
görüntüsünde precision 0.898, recall 0.806, mAP50 0.885 ve mAP50-95 0.574
ölçüldü. Kullanılacak model:

```text
outputs/training/yolo26n_halisaha_preserve_pilot/weights/best.pt
```

5 saniyelik takip kontrollerinde bu model 5. dakikada başlangıç modeliyle aynı
13 ID'yi üretirken uzun iz sayısını 6'dan 7'ye çıkardı. 30. dakikada kare başına
ortalama oyuncu sayısını 14.22'den 14.55'e ve maksimumu 15'ten 16'ya yükseltti;
ancak toplam ID 19'dan 22'ye çıktığı için association/Re-ID iyileştirmesi hâlâ
gereklidir. Validation metriği takip kimliği başarısı değildir; test bölümü manuel
kontrol edilmeden bağımsız test sonucu raporlanmamalıdır.

Test bölümündeki 20 görüntü ve 298 oyuncu kutusu manuel kontrol edildikten sonra
bağımsız test ölçümü yapılmıştır. Hazır YOLO26n için precision 0.916, recall 0.733,
mAP50 0.844 ve mAP50-95 0.560; korumalı ince ayarlı model için precision 0.891,
recall 0.785, mAP50 0.879 ve mAP50-95 0.571 ölçülmüştür. Yeni model recall'u
5.2 yüzde puan, mAP50'yi 3.5 yüzde puan artırmıştır. Precision'daki 2.5 yüzde
puanlık düşüş, daha fazla gerçek oyuncu yakalanırken az miktarda daha fazla yanlış
tespit üretilebileceğini gösterir.

### Çok maçlı v2 veri seti

`assets/videos/matches` altındaki üç maçtan daha geniş bir veri seti hazırlanır:

```bash
.venv-train/bin/python scripts/prepare_multimatch_player_dataset.py
```

Bu yapı `match_01` ve `match_02` kayıtlarından 480 train ve 80 validation;
tamamen görülmeyen `match_03` kaydından 120 test görüntüsü üretir. Toplam 680
görüntü `datasets/halisaha_player_v2` altında tutulur. Ön etiketler korumalı ince
ayarlı model, tam görüntü ve uzak saha crop'u birlikte kullanılarak oluşturulur.

Etiket kontrol sırası:

```bash
.venv-train/bin/python scripts/review_yolo_labels.py \
  --dataset datasets/halisaha_player_v2 --split train
.venv-train/bin/python scripts/review_yolo_labels.py \
  --dataset datasets/halisaha_player_v2 --split val
.venv-train/bin/python scripts/review_yolo_labels.py \
  --dataset datasets/halisaha_player_v2 --split test
```

Yeni model eğitilirken 80 sınıflı hazır çıkış katmanı korunmalıdır:

```bash
.venv-train/bin/python scripts/train_player_detector.py \
  --dataset datasets/halisaha_player_v2 \
  --data-yaml datasets/halisaha_player_v2/data_coco80.yaml \
  --model outputs/training/yolo26n_halisaha_preserve_pilot/weights/best.pt \
  --epochs 30 --image-size 960 --batch 4 --device mps \
  --learning-rate 0.0005 --freeze 10 \
  --name yolo26n_halisaha_multimatch_v2
```

## GitHub kopyasının kapsamı

Bu depo kaynak kodu, yapılandırmaları, testleri ve proje belgelerini içerir. Büyük veri setleri, kamera videoları, model ağırlıkları, sanal ortamlar ve üretilmiş analiz/eğitim çıktıları yerel proje klasöründe tutulur; bu depoya dahil değildir. Çalıştırmadan önce gerekli model ve verileri README içindeki konumlara yerleştirin. Veri seti notlarındaki yerel dosya bağlantıları bu kopyada bulunmayabilir.
