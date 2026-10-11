# Birleşik oyuncu veri seti v1

27 Eylül 2026'da iki kaynağın bağımsız kopyaları birleştirildi. Orijinal görüntüler, etiketler ve mevcut model değiştirilmedi.

| Kaynak | Train | Val | Test | Toplam |
|---|---:|---:|---:|---:|
| Senin onaylı anlık kopyan | 359 | 32 | 0 | 391 |
| Arkadaşının temizlenmiş verisi | 482 | 101 | 65 | 648 |
| Toplam | 841 | 133 | 65 | 1.039 |

Toplam 14.847 oyuncu kutusu bulunur. İki dosyadaki 15 çokgen, çevreleyen kutuya dönüştürüldü. Oyuncu içeren ama etiketi boş olan iki görüntü dışarıda bırakıldı; tam yolları `merge.json` içindedir. Görüntü pikselleri değiştirilmedi; arkadaşının 1280×1280 esnetilmiş görüntüleri aynı kaldı.

Kendi kaynağın olarak `halisaha_reviewed407_20260904` kullanıldı: zamansal tamponu ve onaylı etiketleri olan 391 görüntü. V2'nin tamamı henüz onaylı olmadığı için eklenmedi. Aynı görüntülerin tekrar eklenmesini önlemek için eski pilot ve V2 klasörleri topluca birleştirilmedi. Eski test kümelerin olduğu yerde korundu.

Arkadaşının maçlarının seninkilerden farklı olduğu kullanıcı tarafından doğrulandı. Arkadaşının kendi train/valid/test bölümlerinin maç veya zaman açısından bağımsız olduğu henüz bilinmiyor. Bu sürüm mevcut bölünmeyi korur; test görüntüleri eğitime taşınmadı. 65 görüntülük test için 'bağımsız maç testi' iddiası yapılmamalıdır. `manifest.csv`, her görüntünün kaynağını, eski bölümünü, dosya hash'lerini ve inceleme durumunu içerir.

`data.yaml` tek sınıflı; `data_coco80.yaml` projede kullanılan 80 sınıflı çıkış yapısını koruyan tanımdır. Her iki tanımda etiketler yalnız person=0'dır. Arkadaşının kaynak açıklamaları ve CC BY 4.0 atıf bilgisi `provenance/` altında korunur.

## Kontrol ve eğitim

Aşağıdaki komutlar proje kökünde çalıştırılır. Veri bağlantısını eğitim başlatmadan doğrulama:

```bash
.venv-train/bin/python scripts/train_player_detector.py \
  --dataset datasets/halisaha_merged_v1 \
  --data-yaml datasets/halisaha_merged_v1/data_coco80.yaml \
  --model outputs/training/yolo26n_halisaha_preserve_pilot/weights/best.pt \
  --dry-run
```

Arkadaşının etiketleri yapısal olarak doğrulandı ancak bütün kutular insan tarafından yeniden incelenmiş sayılmadı. Yerel inceleme aracı ilk onaysız görüntüden başlar; onaylı eski görüntülere yeniden başlamaz:

```bash
.venv-train/bin/python scripts/review_yolo_labels.py --dataset datasets/halisaha_merged_v1 --split train
.venv-train/bin/python scripts/review_yolo_labels.py --dataset datasets/halisaha_merged_v1 --split val
.venv-train/bin/python scripts/review_yolo_labels.py --dataset datasets/halisaha_merged_v1 --split test
```

Kaydetme, manifestte onay durumunu ve güncel etiket hash'ini birlikte yeniler. Kaynak etiket hash'i değişmeden kalır. `verification.json` ilk içe aktarma anının kaydıdır; etiketleri sonradan düzenlersen bu kayıt geçmiş doğrulamayı temsil eder.

Train/val incelemesi tamamlandıktan sonra kısa eğitim kontrolü:

```bash
.venv-train/bin/python scripts/train_player_detector.py \
  --dataset datasets/halisaha_merged_v1 \
  --data-yaml datasets/halisaha_merged_v1/data_coco80.yaml \
  --model outputs/training/yolo26n_halisaha_preserve_pilot/weights/best.pt \
  --epochs 1 --image-size 960 --batch 4 --device mps \
  --learning-rate 0.0005 --freeze 10 \
  --name yolo26n_merged_v1_smoke
```

Bu bir başlangıç profilidir; başarı iddiası değildir. Bu birleştirme sırasında eğitim başlatılmadı. Mevcut model karşılaştırma referansı olarak korundu. Test etiketlerini de değerlendirmeden önce incelemek gerekir.

## Yeniden üretim

Yeni ve henüz var olmayan bir çıktı klasörüne:

```bash
.venv-boxmot/bin/python scripts/merge_player_datasets.py \
  --friend /Users/ziyacan/Desktop/Halisaha_Oyuncular.v3i.yolov11 \
  --output datasets/halisaha_merged_v2
```

Araç mevcut çıktı klasörünün üzerine yazmaz. Tekrarlanan birebir görüntü içeriğini reddeder. İçe aktarma, kaynak veriye insan onayı eklemez ve maç bilgisi uydurmaz.
