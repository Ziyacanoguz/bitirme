# Tracking kaynakları

İlk karşılaştırmada detector sabit tutulacak ve Ultralytics'in yerleşik tracker
entegrasyonu üzerinden ByteTrack ile BoT-SORT çalıştırılacaktır. Böylece sonuç
farkı detector değişikliğinden değil, association/tracking yönteminden gelir.

## Orijinal projeler

| Algoritma | Orijinal kaynak | Lisans | Bu projedeki rol |
| --- | --- | --- | --- |
| ByteTrack | https://github.com/ifzhang/ByteTrack | MIT | İlk ve hızlı baseline |
| BoT-SORT | https://github.com/NirAharon/BoT-SORT | MIT | Re-ID ve association karşılaştırması |
| DeepSORT | https://github.com/nwojke/deep_sort | GPL-3.0 | Eğitimsel/ikinci aşama karşılaştırma |
| StrongSORT | https://github.com/dyhBUPT/StrongSORT | GPL-3.0 | Gelişmiş ikinci aşama karşılaştırma |

Kod kopyalamadan veya bir depoyu projeye vendor etmeden önce lisans ve bağımlılık
uyumu tekrar kontrol edilmelidir. Özellikle GPL kodu ürün koduna doğrudan dahil
etmek, MIT bileşenlere göre farklı dağıtım yükümlülükleri doğurabilir.
