# Halı Saha Video Analizi — model paylaşımı ve kullanım

Oyuncu tespiti, ByteTrack takibi ve iki kameradan saha analizi için proje kodu.
Arkadaş ekip kendi videosunu/veri setini kullanabilir. Model ağırlıkları Git geçmişinde
bulunmaz; [Releases](https://github.com/Ziyacanoguz/bitirme/releases/tag/ziya-2026-10-12)
bölümünden indirilir. Veri setleri, maç videoları ve sanal ortamlar paylaşılmaz.

## 1. Kurulum

Python 3.12 önerilir. Komutları depo kökünde çalıştırın.

```bash
git clone https://github.com/Ziyacanoguz/bitirme.git
cd bitirme
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`

macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-inference.txt
```

Temel tespit ve ByteTrack için BoxMOT gerekmez. İleri iki kamera/Re-ID araçları
ek bağımlılık ve kendi kameranıza ait kalibrasyon ister; aşağıdaki hızlı başlangıç
tek kamera içindir. NVIDIA kullanacaksanız donanımınıza uygun PyTorch/CUDA kurulumunu
[PyTorch kurulum sayfasından](https://pytorch.org/get-started/locally/) yapın.

## 2. Modelleri indir

Release sayfasındaki üç `.pt` dosyasını depo kökünde oluşturacağınız `models/`
klasörüne koyun. Alternatif olarak GitHub CLI kuruluysa:

```bash
gh release download ziya-2026-10-12 --repo Ziyacanoguz/bitirme --pattern "*.pt" --dir models
python scripts/verify_models.py
```

Dosya | Açıklama | Eğitim görüntü boyutu
--- | --- | ---
`yolo26n.pt` | Yerel projenin mevcut varsayılan dosyasının aynı kopyası; `extra_labels_augmented_20261003` eğitimi, 80 sınıflı başlık; yalnız sınıf 0 (`person`) kullanılır | 960
`yolo26n_merged_v3_1280.pt` | Birleşik v3 veri setinde eğitilmiş YOLO26n; tek sınıf `person` | 1280
`yolo11s_merged_v3_1280.pt` | Birleşik v3 veri setinde eğitilmiş YOLO11s; tek sınıf `person` | 1280

Dosyalar Ultralytics 8.4.129 ile açılarak doğrulanmıştır. `SHA256SUMS.txt`
indirilen dosyaların yayımlanan kopyayla aynı olduğunu denetler. Yeni sürüm için yeni
release etiketi kullanılmalıdır; yayımlanan ağırlıkları değiştirmeyin.

## 3. Kendi videosunda çalıştır

```bash
python scripts/track_players.py --video "C:/videos/mac.mp4" --model models/yolo11s_merged_v3_1280.pt --tracker-config configs/bytetrack_halisaha.yaml --start 0 --duration 30 --image-size 1280 --confidence 0.15
```

Video yolunu kendi dosyanızla değiştirin; macOS/Linux'ta örneğin `/home/user/mac.mp4`.
Sonuç videosu ve kare/kutu/takip kimliği CSV'si `outputs/tracks/` altına yazılır.
`--show` önizleme açar. `--start 0` videonun başından işlem yapar.

Takip ayarları başlangıç profilidir: farklı kamera, FPS ve görüntü kalitesi için
ayarlama gerekebilir. Tek kamera takip kimliği, doğrulanmış kişiler arası veya
kameralar arası kalıcı kimlik değildir. Depodaki `roi_camera*` ve `calibration_camera*`
dosyaları bizim kameralara aittir; kendi videonuza doğrudan uygulamayın.

## 4. Kendi YOLO veri setinde değerlendir

Veri seti YOLO detection biçiminde olmalı:

```text
my_halisaha/
  images/train/  images/val/
  labels/train/  labels/val/
  data.yaml
```

Her resme aynı adlı `.txt` etiketi karşılık gelir. Satır biçimi:
`class_id x_center y_center width height`; koordinatlar 0–1 aralığında normalize edilir.
Bu proje oyuncuyu sınıf `0` olarak bekler. Roboflow dışa aktarımındaki `train/images`
ve `valid/images` düzeni de kendi YAML'ında doğru tanımlandığında kullanılabilir.
`configs/dataset.example.yaml` örneğini veri setinize kopyalayıp `path` değerini
mutlak konumunuzla, split yollarını kendi düzeninizle değiştirin. Örnek resim/etiket
üretilmez; gerçek veri setinizi kullanın.

```bash
python scripts/evaluate_detector_models.py --data "C:/datasets/my_halisaha/data.yaml" --model yolo26n=models/yolo26n_merged_v3_1280.pt --model yolo11s=models/yolo11s_merged_v3_1280.pt --split val --image-size 1280 --device cpu --output-dir outputs/friend_eval
```

`--device cpu` tüm platformlar için başlangıç seçeneğidir; NVIDIA için `--device 0`,
Apple Silicon için `--device mps` kullanılabilir. Rapor `comparison.json` dosyasına
precision, recall, mAP50 ve mAP50-95 yazar. Eğitimde kullanılmamış aynı split ve aynı
ayarlarla karşılaştırın. Test split'i varsa son ölçümü `--split test` ile yapın.

## 5. Kendi veri setinde eğit

Projedeki `train_player_detector.py` bizim inceleme manifestimizi zorunlu tutar.
Arkadaş ekibin standart YOLO veri seti için doğrudan Ultralytics komutu kullanılabilir:

```bash
yolo detect train model=models/yolo11s_merged_v3_1280.pt data="C:/datasets/my_halisaha/data.yaml" epochs=50 imgsz=1280 batch=4 device=cpu workers=0 project=outputs/training name=friend_finetune
```

GPU için `device=0`, Apple Silicon için `device=mps` seçin. Bellek yetersizse batch
veya görüntü boyutunu azaltın. Bu komut paylaşılan ağırlıktan devam eder; bağımsız
başlangıç deneyi için `model=yolo11s.pt` kullanılır (ilk kullanımda indirilebilir).
Yeni en iyi model normalde `outputs/training/friend_finetune/weights/best.pt` olur;
aynı isimle tekrar çalıştırılırsa Ultralytics farklı bir çıktı klasörü oluşturabilir.
Yeni dosyayı `--model` ile takip/değerlendirme komutlarına verin.

## Kayıtlı değerlendirme

Aşağıdaki değerler **yereldeki mevcut değerlendirme raporundan** aktarılmıştır;
bu paylaşım sırasında tam veri seti değerlendirmesi tekrar çalıştırılmamıştır.
Split: `halisaha_merged_v3/val`, `imgsz=1280`. Arkadaş ekibin veri setindeki
sonuçlarla doğrudan karşılaştırılamaz; bağımsız maçlara genelleme kanıtı değildir.

Model | Precision | Recall | mAP50 | mAP50-95
--- | --- | --- | --- | ---
YOLO26n v3 1280 | 0.9641 | 0.9505 | 0.9814 | 0.7384
YOLO11s v3 1280 | 0.9805 | 0.9626 | 0.9853 | 0.7465

[Ham kayıt](docs/model_evaluation_1280.json). Mevcut `yolo26n.pt` dosyasına bu
puanlar atfedilmez; o farklı bir checkpoint'tir. Tüm modellerde oyuncu sınıfı 0'dır.

## Geliştirme

```bash
python scripts/run_tests.py --python .venv/bin/python
```

Windows'ta `--python .venv/Scripts/python.exe` kullanın.
[Önceki proje notları ve ileri araçlar](PROJECT_HISTORY.md),
[üçüncü taraf kaynakları](third_party/README.md).
