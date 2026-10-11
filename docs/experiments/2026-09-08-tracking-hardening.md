# Takip ve Global ID Sertleştirmesi — 8 Eylül 2026

## Amaç ve kapsam

Bu çalışma mevcut oyuncu tespit modelini yeniden eğitmekten çok, model çıktısının
takip ve iki kamera eşleştirme hattına güvenli, tekrar üretilebilir ve incelenebilir
biçimde taşınmasına odaklanır. Şu parçalar geliştirildi:

1. tam görüntü ve uzak saha crop tespitlerinin güvenli birleştirilmesi;
2. ground truth gerektirmeyen tracking risk denetimi;
3. tek kamera analizinin tek komut ve manifest ile yürütülmesi;
4. iki kamera saha izdüşümü, segmentasyon, Re-ID ve Global ID v3 hattı;
5. Global ID adayları için provenans kontrollü insan inceleme paketi.

Bu turda ısı haritası veya kuş bakışı radar eklenmedi. Önceki isteğe uygun olarak
bu çıktılar geri alınmış durumda kalır. Mevcut tercih edilen YOLO ağırlığı,
`confidence=0.15`, ana görüntü boyutu `1920`, crop boyutu `1280`,
`track_thresh=0.30`, `match_thresh=0.80` ve `track_buffer=75` varsayılanları da
bağımsız başarı kanıtı olmadan değiştirilmedi.

## Yeni bileşenler

### Tespit birleştirme

`src/halisaha/detection_fusion.py`, tam kare ve büyütülmüş crop tespitlerinin
birleştirilmesini takip döngüsünden ayırır:

- crop içindeki kutuları tam kare koordinatlarına taşır;
- girdilerin boyutunu ve sonlu sayılardan oluştuğunu doğrular;
- yalnız `person` sınıfıyla çalışıldığı için sınıftan bağımsız NMS uygular;
- güven puanına göre kararlı sıralama kullanır; eşit puanda kaynak sırası
  korunur ve sonuç tekrar üretilebilir olur;
- çağıranın girdi dizilerini yerinde değiştirmez.

Bu işlem, iki gerçek oyuncuyu tek kutu halinde algılayan detector hatasını çözmez.
Yalnız tam kare ve crop'un aynı oyuncu için ürettiği iki kutunun takipçiye iki
ayrı kişi olarak verilme riskini azaltır.

### Tracking kalite denetimi

`scripts/audit_tracking_quality.py`, bir veya daha fazla CSV için şunları üretir:

- kare başına aktif iz sayısı ve CSV kare kapsamı;
- kısa ID'ler, ID içindeki eksik kare aralıkları ve ani ayak noktası sıçramaları;
- farklı ID'lerin yüksek IoU ile üst üste geldiği olası çift kutu olayları;
- yakın konumda bitip başlayan olası track parçalanma bağlantıları;
- confidence dağılımı;
- özet JSON, tam JSON, kare/ID/olay bazlı CSV tabloları ve okunabilir Markdown.

“Kare kapsamı”, bir karede CSV'de en az bir iz bulunmasıdır; bütün gerçek
oyuncuların tespit edildiği anlamına gelmez. Kısa ID, yüksek IoU veya yakın
başlangıç/bitiş de tek başına hata değildir. Bu nedenle raporun tamamı
`tracking_quality_diagnostics_without_ground_truth` olarak işaretlenir.

### Tek kamera uçtan uca hattı

`scripts/run_player_analysis.py` artık detector ve tracker'ın temel ayarlarını
komut satırından alır. Akış sırasıyla:

```text
YOLO tam kare + crop → ByteTrack → saha izdüşümü → OSNet görünüş çıkarımı
→ temkinli tracklet birleştirme → ham/birleştirilmiş kalite denetimi
→ yerel ID video çıktısı
```

Çalışmanın ayarları ve çağrılan komutlar `run.json` içine yazılır. Hem ham takip
CSV'si hem `stitched.csv` aynı kalite denetiminde yer alır. Oluşturulan kimlikler
tek kameraya ait ve doğrulanmamış kimliklerdir.

### Tek komutluk iki kamera Global ID hattı

`scripts/run_global_id_pipeline.py`, hazır yerel takip CSV'lerinden şu aşamaları
aynı çalışma dizininde yürütür:

```text
girdi denetimi → yerel tracking kalite denetimi → iki saha izdüşümü
→ kısa segmentler → iki OSNet arşivi → temkinli Global ID fusion
```

Girdi CSV'lerinde yinelenen `(frame, track_id)`, geçersiz kutu, video sınırını
aşan kare ve yetersiz iki kamera kare örtüşmesi çalışmayı erken durdurur. Her
aşamanın komutu, durumu ve süresi `pipeline.json` içinde tutulur.

`--sync-method` zorunludur; manuel eşlemede ayrıca açıklayıcı `--sync-note`
kullanılmalıdır. Pipeline'ın iki CSV'de aynı kare numaralarını görmesi yalnız
yapısal örtüşmedir. Görüntü veya ses içeriğinin gerçekten senkron olduğunu
kendiliğinden kanıtlamaz ve ofseti sessizce değiştirmez.

Akşam devam çalışmasında doğrulanmamış ön denemeler için `assumed_offset` yöntemi
eklendi. Hem `manual_event` hem `assumed_offset` boş olmayan not ister. Gerçek olay
işaretlenmeden önceki bir ofseti kullanmak `manual_event` olarak kaydedilmemelidir.

### Temkinli Global ID v3

V3, önce yerel kamera süreklilik bağlarını, sonra kameralar arası bağları işler.
Temel korumalar şunlardır:

- aynı kamerada aynı anda bulunan farklı track'ler aynı bileşende birleşemez;
- bir eşleşmenin toplam desteğinin yanında en uzun ardışık destek dizisi ölçülür;
- yalnız medyan konuma bakılmaz; kötü uç örnekleri yakalamak için p95 saha
  mesafesi sınırlandırılır;
- Re-ID görünüşü yoksa daha fazla destek, daha yüksek destek oranı ve daha düşük
  p95 mesafe gerekir;
- yeni bir bağın bileşen içindeki başka görünüşlerle açıkça çelişmesi birleşmeyi
  engeller;
- doğrudan eşleşme, bileşen ve yerel süreklilik kararlarının kabul/ret nedeni
  ayrı tablolara yazılır;
- büyük iki kamera konum anlaşmazlığı ortalanmaz; iki gözlem çelişkili olarak
  ayrı tutulur.

`identity_evidence_score` alanı 0–1 aralığında sezgisel kanıt özetidir. Olasılık,
confidence veya ölçülmüş kimlik doğruluğu değildir. Sistem oyuncu sayısını 16'ya
zorlamaz, takım rengine göre kimlik birleştirmez ve doğrulanmamış Global ID'yi
gerçek kişi kimliği olarak sunmaz.

### Global ID insan incelemesi

`scripts/create_global_id_review_pack.py`, kabul edilmiş kameralar arası adaylar
içinden riskli örnekleri seçip iki oyuncu crop'unu yan yana gösteren HTML üretir.
Yeni `--fusion-run` girdisi zorunludur:

- candidates ve mapping aynı fusion klasöründen gelmelidir;
- takip dosyaları fusion `run.json` kayıtları ve varsa SHA-256 değerleriyle
  eşleşmelidir;
- kamera ofseti run kaydından alınır veya açıkça verilen değer onunla aynı
  olmak zorundadır;
- v3 adaylarında örnek kare, yalnız gerçek `support_frames` içinden seçilir.

İşaretlemeler tarayıcının yerel depolamasına otomatik kaydedilir. Takımla paylaşmak
için “Sonuç CSV'sini indir” düğmesiyle `review_completed.csv` alınmalıdır. Bir
karttaki karar tek bir görüntü çiftini doğrular; oyuncunun bütün maçtaki Global ID
güzergâhını tek başına doğrulamaz.

İnceleme paketi v3, kayıt anahtarını kaynak dosyalarına ve gösterilen kesin karelere
bağlar; iki JPEG'in SHA-256 değerlerini de saklar. Yalnız segment numaralarına
dayanan eski tarayıcı kayıtları otomatik taşınmaz. Güncel teslim ve doğrulama
sonuçları [devam notundadır](../CONTINUE_HERE.md).

## Gerçek tanısal sonuçlar

### 60 saniyelik yerel takip durumu

Her iki kamera için 30:00–31:00 arasındaki 1.500 kare incelendi. Kaynaklar,
önceden oluşturulmuş temkinli `stitched.csv` dosyalarıdır.

| Kamera | CSV kapsamı | Ort. aktif iz/kare | Benzersiz ID | 1 sn'den kısa ID | İç boşluklu ID | Olası parçalanma bağı |
|---|---:|---:|---:|---:|---:|---:|
| Kamera 1 | %100 | 11,98 | 73 | 27 | 37 | 27 |
| Kamera 2 | %100 | 11,77 | 101 | 49 | 53 | 28 |

Bu tablo, özellikle kamera 2'de kimlik devrinin yüksek olduğuna dair inceleme
önceliği verir. “73/101 ID” gerçek oyuncu sayısı değildir ve tablodan ID switch
veya tracking doğruluğu çıkarılamaz.

Kaynak tanısal rapor:
`outputs/audits/tracking_quality_20260908_two_camera_60s/summary.json`.

### Uzak saha crop boyutu A/B deneyi — 5 saniye

Aynı Match 03 bölümü, model ve eşiklerle yalnız `crop_image_size` 1280/1920
değiştirildi. Her çalışma 125 kare içerir.

| Kamera / crop | NMS sonrası tespit | Tracking satırı | Ort. aktif iz | Benzersiz ID | Kısa ID | İç boşluklu ID | Olası parçalanma bağı | Ort. confidence | Süre |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K1 / 1280 | 1.350 | 1.272 | 10,176 | 18 | 5 | 7 | 1 | 0,812 | 7,50 sn |
| K1 / 1920 | 1.335 | 1.289 | 10,312 | 18 | 5 | 8 | 0 | 0,845 | 7,48 sn |
| K2 / 1280 | 1.602 | 1.591 | 12,728 | 17 | 3 | 5 | 1 | 0,849 | 12,32 sn |
| K2 / 1920 | 1.620 | 1.603 | 12,824 | 16 | 3 | 5 | 0 | 0,856 | 10,73 sn |

1920 crop bazı sayaçlarda küçük iyileşmeler gösterse de kamera 1'de boşluklu ID
sayısı 7'den 8'e çıktı; kamera 1 NMS sonrası tespit sayısı da azaldı. Üstelik bu
yalnız beş saniyelik ve ground-truth'suz bir örnektir. Süreler tek tekrar olduğu
için güvenilir performans benchmark'ı değildir. Bu yüzden `crop_image_size=1280`
varsayılanı korundu.

Kaynaklar: ilgili çalışma JSON dosyaları ve
`outputs/audits/tracking_quality_20260908_crop_ab_both_cameras_5s/summary.json`.

### Global ID fusion v2–v3 karşılaştırması — 60 saniye

Aynı kamera segmentleri, OSNet arşivleri ve 477 kameralar arası aday üzerinde
karşılaştırma yapıldı.

| Gösterge | V2 | Temkinli V3 |
|---|---:|---:|
| Kabul edilen aday/kenar | 321 | 288 |
| Ortak iki kamera bileşeni | 36 | 35 |
| Toplam bileşen | 213 | 224 |
| İki gözlemle birleştirilen satır | 14.280 | 13.278 |
| Ortak bileşenin tek kamera satırı | 4.187 | 5.844 |
| Eşleşmemiş satır | 1.977 | 2.304 |
| Çelişki olayı | 280 | 290 |
| Çelişki satırı | 560 | 580 |
| Geçersiz saha konumu satırı | 334 | 334 |

V3'ün 288 kabulü, bileşenleri gerçekten birleştiren 103 bağ ile mevcut
bileşenleri destekleyen 185 ek kanıttan oluşur. V3 ayrıca 607 yerel süreklilik
bileşen bağı kaydetti. Reddedilen 189 adayın önemli nedenleri yetersiz destek
(129), yetersiz destek oranı (25), aynı kamera çakışması (19), görünüş eksikken
zayıf geometri (10), yüksek p95 mesafe (5) ve bileşen görünüş çelişkisidir (1).

V3'ün daha az satırı iki kamera arasında birleştirip daha fazlasını tek kamera
veya eşleşmemiş bırakması tasarlanan temkinli davranıştır. Ancak bunun daha doğru
olduğunu söylemek için insan kimlikli ground truth gerekir. V2'deki
`disagreement_frames` alanı da ayrı video karesi sayısı değil, bileşen-kare
çelişki olayı olarak yorumlanmalıdır.

Kaynaklar:

- `outputs/experiments/calibration_fusion_20260904/fusion_v2/run.json`
- `outputs/experiments/calibration_fusion_20260904/fusion_quality_v3_final/run.json`

## Hızlı kullanım

Her komutta daha önce bulunmayan yeni bir `output-dir` seçilmelidir; araçlar eski
sonuçların üzerine yazmayı reddeder.

### 1. Tek kamerayı çalıştır

```bash
.venv-boxmot/bin/python scripts/run_player_analysis.py \
  --video assets/videos/matches/match_03/camera1.mp4 \
  --model outputs/training/yolo26n_halisaha_preserve_pilot/weights/best.pt \
  --roi configs/roi_camera1.json \
  --far-crop configs/far_crop_camera1.json \
  --calibration configs/calibration_camera1_lines_v1.json \
  --output-dir outputs/runs/camera1_tracking_next_01 \
  --camera "Kamera 1" --start 1800 --duration 10 --device mps
```

İkinci kamera için video, ROI, crop, calibration, kamera adı ve output klasörü
değiştirilir.

### 2. Var olan CSV'leri tek başına denetle

```bash
.venv-boxmot/bin/python scripts/audit_tracking_quality.py \
  --tracks camera1=outputs/experiments/reviewed407_20260904/camera1/stitched.csv \
  --tracks camera2=outputs/experiments/reviewed407_20260904/camera2/stitched.csv \
  --fps 25 \
  --output-dir outputs/audits/tracking_quality_next_01
```

### 3. Global ID planını yazmadan doğrula

Önce aynı fiziksel olay iki videoda elle bulunmalı ve gerçek kare farkı
`--offset-frames` olarak yazılmalıdır. Aşağıdaki açıklama metni gerçek gözlemle
değiştirilmelidir.

```bash
.venv-boxmot/bin/python scripts/run_global_id_pipeline.py \
  --video1 assets/videos/matches/match_03/camera1.mp4 \
  --video2 assets/videos/matches/match_03/camera2.mp4 \
  --tracks1 outputs/experiments/reviewed407_20260904/camera1/stitched.csv \
  --tracks2 outputs/experiments/reviewed407_20260904/camera2/stitched.csv \
  --calibration1 configs/calibration_camera1_lines_v1.json \
  --calibration2 configs/calibration_camera2_lines_v1.json \
  --offset-frames 0 \
  --sync-method manual_event \
  --sync-note "DEĞİŞTİR: seçilen ortak olay ve iki kameradaki kare numaraları" \
  --output-dir outputs/runs/global_id_next_01 \
  --dry-run
```

Plan doğruysa aynı komuttan `--dry-run` kaldırılır. Buradaki `0`, örnek değerdir;
manuel olay doğrulaması yapılmadan gerçek senkron ofseti sayılmaz.

### 4. Riskli Global ID adaylarını incelemeye aç

Bu komut ancak bir önceki gerçek çalışma tamamlandıktan sonra kullanılmalıdır:

```bash
.venv-boxmot/bin/python scripts/create_global_id_review_pack.py \
  --video1 assets/videos/matches/match_03/camera1.mp4 \
  --video2 assets/videos/matches/match_03/camera2.mp4 \
  --tracks1 outputs/runs/global_id_next_01/camera1_segments.csv \
  --tracks2 outputs/runs/global_id_next_01/camera2_segments.csv \
  --candidates outputs/runs/global_id_next_01/global_ids/candidates.csv \
  --mapping outputs/runs/global_id_next_01/global_ids/mapping.csv \
  --fusion-run outputs/runs/global_id_next_01/global_ids/run.json \
  --output-dir outputs/reviews/global_id_next_01 \
  --max-pairs 40 --per-global 3
```

`index.html` tarayıcıda açılır; cevaplar otomatik saklansa da paylaşılabilir kayıt
için tamamlanan CSV ayrıca indirilmelidir.

## Beş kişilik ekipte modül sınırları

| Sorumlu | Sahip olduğu bölüm | Başka modüle verdiği sözleşme |
|---|---|---|
| 1 — Detector/tracking | YOLO, ROI/crop, `detection_fusion`, ByteTrack, yerel kalite denetimi | `frame, track_id, confidence, x1, y1, x2, y2` CSV + çalışma JSON'u |
| 2 — Senkron/Global ID | Senkron kanıtı, saha calibration, segmentasyon, OSNet, fusion v3, insan incelemesi | `fused.csv`, `mapping.csv`, `components.csv`, `candidates.csv`, `run.json` |
| 3 — Top/olay | Top detector'ı, şut/gol/oyun dışı adayları ve highlight zamanları | Global ID'yi değiştirmeden olay JSON'u |
| 4 — Backend | Video yükleme, iş kuyruğu, komut çalıştırma, manifest ve sonuç API'si | Sürümü belli API/iş durumu sözleşmesi |
| 5 — Uygulama | Maç yükleme, işlem durumu, video/oyuncu/olay ekranları ve inceleme deneyimi | Backend API tüketimi; CV dosyalarına doğrudan yazmaz |

Birinci kişi Global ID kurallarını, ikinci kişi detector eşiklerini sessizce
değiştirmemelidir. Değişiklikler yeni output klasöründe tek değişkenli deney ve
ortak raporla yapılmalıdır. Üçüncü, dördüncü ve beşinci kişiler görüntü işleme
hattını beklemeden sürümlenmiş örnek JSON/CSV sözleşmeleriyle paralel ilerleyebilir.

## Sonraki doğrulama adımı

Şimdiki en değerli çalışma, v3 inceleme paketindeki aynı/farklı/kararsız
kararlarını tamamlayıp küçük bir klip için gerçek oyuncu kimliği ground truth'u
oluşturmaktır. Ancak bundan sonra IDF1, HOTA, ID switch ve fragmentation sayıları
hesaplanmalı; crop 1920, farklı tracker veya yeni eşikler bu sabit doğrulama
kümesinde karşılaştırılmalıdır.
