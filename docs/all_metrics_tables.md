# All Metrics: Lengkap Hasil Eksperimen (auto-generated)

> 📋 **Auto-generated** dari `scripts/build_results_tables.py`. Regenerate kapan saja saat sweep baru selesai.
>
> ⏳ = combination belum dijalankan / sedang berjalan. Cell ?  = file ada tapi metric tidak ada.
>
> Setiap baris tabel = satu run unik. Metrik: macro_f1, weighted_f1, accuracy.

---

## 1. Primer Unimodal (Landmark + Image)

Source data: `models/frontonly_conf60/{3,7}class/Unified/`.

### 1.1 Landmark (raw_136 / facs_28 / blendshape_52 / facs_plus_bs_80)

| Feature | Source | Arch | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| raw_136 | MP | FCNN | 3c | B1 | 0.5087 | 0.7867 | 0.8202 |
| raw_136 | MP | FCNN | 7c | B1 | 0.2255 | 0.7574 | 0.7578 |
| raw_136 | MP | FCNN | 3c | B2 | 0.5104 | 0.6307 | 0.5511 |
| raw_136 | MP | FCNN | 7c | B2 | 0.1827 | 0.6768 | 0.6416 |
| raw_136 | MP | FCNN | 3c | B3 | 0.5652 | 0.6979 | 0.6738 |
| raw_136 | MP | FCNN | 7c | B3 | 0.2392 | 0.6920 | 0.6017 |
| raw_136 | MP | CNN1D | 3c | B1 | 0.5336 | 0.7878 | 0.8235 |
| raw_136 | MP | CNN1D | 7c | B1 | 0.2471 | 0.7704 | 0.7675 |
| raw_136 | MP | CNN1D | 3c | B2 | 0.5163 | 0.6856 | 0.6297 |
| raw_136 | MP | CNN1D | 7c | B2 | 0.2673 | 0.7385 | 0.6868 |
| raw_136 | MP | CNN1D | 3c | B3 | 0.7005 | 0.8172 | 0.8041 |
| raw_136 | MP | CNN1D | 7c | B3 | 0.2226 | 0.5846 | 0.4930 |
| raw_136 | FA | FCNN | 3c | B1 | 0.7119 | 0.8451 | 0.8471 |
| raw_136 | FA | FCNN | 7c | B1 | 0.3122 | 0.8445 | 0.8450 |
| raw_136 | FA | FCNN | 3c | B2 | 0.6827 | 0.8156 | 0.7836 |
| raw_136 | FA | FCNN | 7c | B2 | 0.2386 | 0.5489 | 0.4467 |
| raw_136 | FA | FCNN | 3c | B3 | 0.6928 | 0.8341 | 0.7976 |
| raw_136 | FA | FCNN | 7c | B3 | 0.2885 | 0.7289 | 0.6319 |
| raw_136 | FA | CNN1D | 3c | B1 | 0.6627 | 0.8213 | 0.8332 |
| raw_136 | FA | CNN1D | 7c | B1 | 0.3256 | 0.8837 | 0.8827 |
| raw_136 | FA | CNN1D | 3c | B2 | 0.6885 | 0.8187 | 0.7922 |
| raw_136 | FA | CNN1D | 7c | B2 | 0.2309 | 0.6518 | 0.5544 |
| raw_136 | FA | CNN1D | 3c | B3 | 0.7358 | 0.8619 | 0.8525 |
| raw_136 | FA | CNN1D | 7c | B3 | 0.2218 | 0.6142 | 0.5135 |
| facs_28 | MP | FCNN | 3c | B1 | 0.5594 | 0.7314 | 0.7061 |
| facs_28 | MP | FCNN | 7c | B1 | 0.1972 | 0.7126 | 0.7449 |
| facs_28 | MP | FCNN | 3c | B2 | 0.6122 | 0.7815 | 0.7653 |
| facs_28 | MP | FCNN | 7c | B2 | 0.2351 | 0.6361 | 0.5307 |
| facs_28 | MP | FCNN | 3c | B3 | 0.6246 | 0.8112 | 0.7879 |
| facs_28 | MP | FCNN | 7c | B3 | 0.2038 | 0.5244 | 0.4220 |
| facs_28 | MP | CNN1D | 3c | B1 | 0.5274 | 0.7593 | 0.7664 |
| facs_28 | MP | CNN1D | 7c | B1 | 0.2239 | 0.7499 | 0.7524 |
| facs_28 | MP | CNN1D | 3c | B2 | 0.5495 | 0.7072 | 0.6329 |
| facs_28 | MP | CNN1D | 7c | B2 | 0.2077 | 0.5529 | 0.4682 |
| facs_28 | MP | CNN1D | 3c | B3 | 0.5613 | 0.6995 | 0.6286 |
| facs_28 | MP | CNN1D | 7c | B3 | 0.1769 | 0.5464 | 0.4273 |
| facs_28 | FA | FCNN | 3c | B1 | 0.7585 | 0.8889 | 0.8870 |
| facs_28 | FA | FCNN | 7c | B1 | 0.3090 | 0.8581 | 0.8579 |
| facs_28 | FA | FCNN | 3c | B2 | 0.7489 | 0.8622 | 0.8536 |
| facs_28 | FA | FCNN | 7c | B2 | 0.3036 | 0.8521 | 0.8138 |
| facs_28 | FA | FCNN | 3c | B3 | 0.7372 | 0.8714 | 0.8547 |
| facs_28 | FA | FCNN | 7c | B3 | 0.2907 | 0.7834 | 0.7126 |
| facs_28 | FA | CNN1D | 3c | B1 | 0.7143 | 0.8601 | 0.8676 |
| facs_28 | FA | CNN1D | 7c | B1 | 0.3114 | 0.8672 | 0.8708 |
| facs_28 | FA | CNN1D | 3c | B2 | 0.6997 | 0.8530 | 0.8342 |
| facs_28 | FA | CNN1D | 7c | B2 | 0.3211 | 0.8433 | 0.8149 |
| facs_28 | FA | CNN1D | 3c | B3 | 0.7120 | 0.8652 | 0.8525 |
| facs_28 | FA | CNN1D | 7c | B3 | 0.3117 | 0.8192 | 0.7772 |
| blendshape_52 | MP | FCNN | 3c | B1 | 0.5688 | 0.8028 | 0.8202 |
| blendshape_52 | MP | FCNN | 7c | B1 | 0.2782 | 0.8181 | 0.8235 |
| blendshape_52 | MP | FCNN | 3c | B2 | 0.6474 | 0.7864 | 0.7664 |
| blendshape_52 | MP | FCNN | 7c | B2 | 0.2177 | 0.7009 | 0.6459 |
| blendshape_52 | MP | FCNN | 3c | B3 | 0.6211 | 0.7926 | 0.7772 |
| blendshape_52 | MP | FCNN | 7c | B3 | 0.2654 | 0.7342 | 0.6846 |
| blendshape_52 | MP | CNN1D | 3c | B1 | 0.5311 | 0.8045 | 0.8353 |
| blendshape_52 | MP | CNN1D | 7c | B1 | 0.2411 | 0.7981 | 0.8202 |
| blendshape_52 | MP | CNN1D | 3c | B2 | 0.6016 | 0.8006 | 0.7901 |
| blendshape_52 | MP | CNN1D | 7c | B2 | 0.2390 | 0.7687 | 0.7287 |
| blendshape_52 | MP | CNN1D | 3c | B3 | 0.6340 | 0.8093 | 0.7750 |
| blendshape_52 | MP | CNN1D | 7c | B3 | 0.2178 | 0.7000 | 0.6372 |
| facs_plus_bs_80 | MP | FCNN | 3c | B1 | 0.5600 | 0.8283 | 0.8525 |
| facs_plus_bs_80 | MP | FCNN | 7c | B1 | 0.2631 | 0.8131 | 0.8213 |
| facs_plus_bs_80 | MP | FCNN | 3c | B2 | 0.6363 | 0.8087 | 0.7815 |
| facs_plus_bs_80 | MP | FCNN | 7c | B2 | 0.2969 | 0.7930 | 0.7578 |
| facs_plus_bs_80 | MP | FCNN | 3c | B3 | 0.6376 | 0.7997 | 0.7686 |
| facs_plus_bs_80 | MP | FCNN | 7c | B3 | 0.2654 | 0.7435 | 0.7051 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B1 | 0.6175 | 0.8116 | 0.8267 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B1 | 0.2849 | 0.8236 | 0.8321 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B2 | 0.6382 | 0.8092 | 0.8009 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B2 | 0.2496 | 0.7765 | 0.7309 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B3 | 0.5910 | 0.8074 | 0.7901 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B3 | 0.2244 | 0.7330 | 0.6803 |
| facs_plus_bs_80 | FA | FCNN | 3c | B1 | 0.7350 | 0.8693 | 0.8687 |
| facs_plus_bs_80 | FA | FCNN | 7c | B1 | 0.2687 | 0.8124 | 0.8192 |
| facs_plus_bs_80 | FA | FCNN | 3c | B2 | 0.7116 | 0.8621 | 0.8471 |
| facs_plus_bs_80 | FA | FCNN | 7c | B2 | 0.3331 | 0.8360 | 0.8181 |
| facs_plus_bs_80 | FA | FCNN | 3c | B3 | 0.6603 | 0.8299 | 0.8095 |
| facs_plus_bs_80 | FA | FCNN | 7c | B3 | 0.2886 | 0.7946 | 0.7524 |
| facs_plus_bs_80 | FA | CNN1D | 3c | B1 | 0.6936 | 0.8663 | 0.8751 |
| facs_plus_bs_80 | FA | CNN1D | 7c | B1 | 0.3279 | 0.8851 | 0.8870 |
| facs_plus_bs_80 | FA | CNN1D | 3c | B2 | 0.7156 | 0.8553 | 0.8288 |
| facs_plus_bs_80 | FA | CNN1D | 7c | B2 | 0.2820 | 0.8325 | 0.8062 |
| facs_plus_bs_80 | FA | CNN1D | 3c | B3 | 0.7261 | 0.8657 | 0.8536 |
| facs_plus_bs_80 | FA | CNN1D | 7c | B3 | 0.3079 | 0.8163 | 0.7761 |

### 1.2 Image (CNN scratch / CNN_TL)

| Arch | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| CNN_SCRATCH | 3c | B1 | 0.5095 | 0.7759 | 0.8019 |
| CNN_SCRATCH | 7c | B1 | 0.2432 | 0.7577 | 0.7922 |
| CNN_SCRATCH | 3c | B2 | 0.4124 | 0.5059 | 0.4855 |
| CNN_SCRATCH | 7c | B2 | 0.1903 | 0.6543 | 0.6286 |
| CNN_SCRATCH | 3c | B3 | 0.5700 | 0.7523 | 0.7374 |
| CNN_SCRATCH | 7c | B3 | 0.2582 | 0.7900 | 0.7783 |
| CNN_TL | 3c | B1 | 0.6348 | 0.8031 | 0.8062 |
| CNN_TL | 7c | B1 | 0.2763 | 0.8087 | 0.8105 |
| CNN_TL | 3c | B2 | 0.7107 | 0.8279 | 0.8149 |
| CNN_TL | 7c | B2 | 0.2964 | 0.8164 | 0.8030 |
| CNN_TL | 3c | B3 | 0.6911 | 0.8164 | 0.7955 |
| CNN_TL | 7c | B3 | 0.2833 | 0.7943 | 0.7750 |

---

## 2. Primer Multimodal (Fusion)

Source data: `models/frontonly_conf60/{3,7}class/Unified/fusion_*/`. Tabel dipecah per jenis fusion (Early / Intermediate / Late) supaya feature × variant × source × scenario × scheme bisa di-compare dalam satu tabel.

### 2.1 Early Fusion (raw_136 only) — concat vs gated mode

| Fusion | Mode | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| early | concat | scratch | raw_136 | MP | 3c | B1 | 0.5937 | 0.8011 | 0.8105 |
| early | concat | scratch | raw_136 | MP | 7c | B1 | 0.2365 | 0.7914 | 0.8138 |
| early | concat | scratch | raw_136 | MP | 3c | B2 | 0.4464 | 0.5909 | 0.5533 |
| early | concat | scratch | raw_136 | MP | 7c | B2 | 0.2175 | 0.6747 | 0.6566 |
| early | concat | scratch | raw_136 | MP | 3c | B3 | 0.6217 | 0.8171 | 0.8019 |
| early | concat | scratch | raw_136 | MP | 7c | B3 | 0.2599 | 0.7615 | 0.7330 |
| early | concat | scratch | raw_136 | FA | 3c | B1 | 0.5368 | 0.7846 | 0.8019 |
| early | concat | scratch | raw_136 | FA | 7c | B1 | 0.2250 | 0.7624 | 0.7987 |
| early | concat | scratch | raw_136 | FA | 3c | B2 | 0.4483 | 0.6563 | 0.6308 |
| early | concat | scratch | raw_136 | FA | 7c | B2 | 0.1799 | 0.5488 | 0.5145 |
| early | concat | scratch | raw_136 | FA | 3c | B3 | 0.6078 | 0.8132 | 0.8009 |
| early | concat | scratch | raw_136 | FA | 7c | B3 | 0.2545 | 0.7589 | 0.7374 |
| early | concat | tl | raw_136 | MP | 3c | B1 | 0.5099 | 0.7241 | 0.7158 |
| early | concat | tl | raw_136 | MP | 7c | B1 | 0.2906 | 0.8266 | 0.8256 |
| early | concat | tl | raw_136 | MP | 3c | B2 | 0.6609 | 0.8334 | 0.8288 |
| early | concat | tl | raw_136 | MP | 7c | B2 | 0.2882 | 0.7796 | 0.7621 |
| early | concat | tl | raw_136 | MP | 3c | B3 | 0.6903 | 0.8216 | 0.8019 |
| early | concat | tl | raw_136 | MP | 7c | B3 | 0.2907 | 0.8282 | 0.8235 |
| early | concat | tl | raw_136 | FA | 3c | B1 | 0.6494 | 0.7920 | 0.7858 |
| early | concat | tl | raw_136 | FA | 7c | B1 | 0.2655 | 0.8182 | 0.8278 |
| early | concat | tl | raw_136 | FA | 3c | B2 | 0.6942 | 0.8281 | 0.8159 |
| early | concat | tl | raw_136 | FA | 7c | B2 | 0.2973 | 0.8250 | 0.8181 |
| early | concat | tl | raw_136 | FA | 3c | B3 | 0.6663 | 0.8152 | 0.7987 |
| early | concat | tl | raw_136 | FA | 7c | B3 | 0.3024 | 0.8530 | 0.8396 |
| early | gated | scratch | raw_136 | MP | 3c | B1 | 0.4112 | 0.7094 | 0.7664 |
| early | gated | scratch | raw_136 | MP | 7c | B1 | 0.2179 | 0.7674 | 0.7966 |
| early | gated | scratch | raw_136 | MP | 3c | B2 | 0.3852 | 0.4242 | 0.4144 |
| early | gated | scratch | raw_136 | MP | 7c | B2 | 0.2129 | 0.6897 | 0.6771 |
| early | gated | scratch | raw_136 | MP | 3c | B3 | 0.6141 | 0.7821 | 0.7567 |
| early | gated | scratch | raw_136 | MP | 7c | B3 | 0.2723 | 0.7995 | 0.7836 |
| early | gated | scratch | raw_136 | FA | 3c | B1 | 0.5838 | 0.7834 | 0.7869 |
| early | gated | scratch | raw_136 | FA | 7c | B1 | 0.2123 | 0.7534 | 0.7815 |
| early | gated | scratch | raw_136 | FA | 3c | B2 | 0.5644 | 0.7542 | 0.7449 |
| early | gated | scratch | raw_136 | FA | 7c | B2 | 0.2257 | 0.7353 | 0.7298 |
| early | gated | scratch | raw_136 | FA | 3c | B3 | 0.6128 | 0.7749 | 0.7546 |
| early | gated | scratch | raw_136 | FA | 7c | B3 | 0.2440 | 0.7398 | 0.7051 |
| early | gated | tl | raw_136 | MP | 3c | B1 | 0.6634 | 0.8176 | 0.8149 |
| early | gated | tl | raw_136 | MP | 7c | B1 | 0.2725 | 0.8075 | 0.8105 |
| early | gated | tl | raw_136 | MP | 3c | B2 | 0.6343 | 0.8110 | 0.8073 |
| early | gated | tl | raw_136 | MP | 7c | B2 | 0.2975 | 0.8333 | 0.8267 |
| early | gated | tl | raw_136 | MP | 3c | B3 | 0.6741 | 0.7863 | 0.7675 |
| early | gated | tl | raw_136 | MP | 7c | B3 | 0.2965 | 0.8107 | 0.7966 |
| early | gated | tl | raw_136 | FA | 3c | B1 | 0.5421 | 0.7771 | 0.7922 |
| early | gated | tl | raw_136 | FA | 7c | B1 | 0.2389 | 0.7377 | 0.7664 |
| early | gated | tl | raw_136 | FA | 3c | B2 | 0.6129 | 0.7715 | 0.7643 |
| early | gated | tl | raw_136 | FA | 7c | B2 | 0.3069 | 0.8424 | 0.8342 |
| early | gated | tl | raw_136 | FA | 3c | B3 | 0.7189 | 0.8471 | 0.8353 |
| early | gated | tl | raw_136 | FA | 7c | B3 | 0.2965 | 0.8073 | 0.7922 |

### 2.2 Intermediate Fusion (semua feature × variant × source)

| Fusion | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| intermediate | scratch | raw_136 | MP | 3c | B1 | 0.5050 | 0.7503 | 0.7858 |
| intermediate | scratch | raw_136 | MP | 7c | B1 | 0.1914 | 0.6589 | 0.6448 |
| intermediate | scratch | raw_136 | MP | 3c | B2 | 0.5373 | 0.7773 | 0.7675 |
| intermediate | scratch | raw_136 | MP | 7c | B2 | 0.2554 | 0.8101 | 0.8116 |
| intermediate | scratch | raw_136 | MP | 3c | B3 | 0.6041 | 0.7486 | 0.7180 |
| intermediate | scratch | raw_136 | MP | 7c | B3 | 0.2455 | 0.7809 | 0.7772 |
| intermediate | scratch | facs_28 | MP | 3c | B1 | 0.5214 | 0.7740 | 0.8052 |
| intermediate | scratch | facs_28 | MP | 7c | B1 | 0.2653 | 0.8311 | 0.8385 |
| intermediate | scratch | facs_28 | MP | 3c | B2 | 0.5415 | 0.7544 | 0.7427 |
| intermediate | scratch | facs_28 | MP | 7c | B2 | 0.2583 | 0.7291 | 0.7169 |
| intermediate | scratch | facs_28 | MP | 3c | B3 | 0.6481 | 0.8172 | 0.8073 |
| intermediate | scratch | facs_28 | MP | 7c | B3 | 0.2692 | 0.7874 | 0.7750 |
| intermediate | scratch | blendshape_52 | MP | 3c | B1 | 0.5590 | 0.8089 | 0.8267 |
| intermediate | scratch | blendshape_52 | MP | 7c | B1 | 0.2044 | 0.7367 | 0.7793 |
| intermediate | scratch | blendshape_52 | MP | 3c | B2 | 0.5502 | 0.7425 | 0.7266 |
| intermediate | scratch | blendshape_52 | MP | 7c | B2 | 0.2249 | 0.7483 | 0.7384 |
| intermediate | scratch | blendshape_52 | MP | 3c | B3 | 0.5215 | 0.7287 | 0.7061 |
| intermediate | scratch | blendshape_52 | MP | 7c | B3 | 0.2359 | 0.7221 | 0.6857 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B1 | 0.5899 | 0.8242 | 0.8439 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B1 | 0.2449 | 0.8129 | 0.8310 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B2 | 0.6106 | 0.8018 | 0.7933 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B2 | 0.2147 | 0.7358 | 0.7287 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B3 | 0.6235 | 0.8159 | 0.8149 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B3 | 0.2300 | 0.7475 | 0.7180 |
| intermediate | scratch | raw_136 | FA | 3c | B1 | 0.7370 | 0.8700 | 0.8676 |
| intermediate | scratch | raw_136 | FA | 7c | B1 | 0.2516 | 0.8270 | 0.8428 |
| intermediate | scratch | raw_136 | FA | 3c | B2 | 0.6548 | 0.8369 | 0.8299 |
| intermediate | scratch | raw_136 | FA | 7c | B2 | 0.3158 | 0.8677 | 0.8579 |
| intermediate | scratch | raw_136 | FA | 3c | B3 | 0.6390 | 0.7480 | 0.7244 |
| intermediate | scratch | raw_136 | FA | 7c | B3 | 0.2794 | 0.7904 | 0.7567 |
| intermediate | scratch | facs_28 | FA | 3c | B1 | 0.7581 | 0.8860 | 0.8848 |
| intermediate | scratch | facs_28 | FA | 7c | B1 | 0.3363 | 0.8863 | 0.8870 |
| intermediate | scratch | facs_28 | FA | 3c | B2 | 0.7107 | 0.8234 | 0.8138 |
| intermediate | scratch | facs_28 | FA | 7c | B2 | 0.2995 | 0.8386 | 0.8353 |
| intermediate | scratch | facs_28 | FA | 3c | B3 | 0.7193 | 0.8784 | 0.8805 |
| intermediate | scratch | facs_28 | FA | 7c | B3 | 0.2865 | 0.8002 | 0.7804 |
| intermediate | scratch | facs_plus_bs_80 | FA | 3c | B1 | 0.6885 | 0.8361 | 0.8342 |
| intermediate | scratch | facs_plus_bs_80 | FA | 7c | B1 | 0.3036 | 0.8495 | 0.8471 |
| intermediate | scratch | facs_plus_bs_80 | FA | 3c | B2 | 0.6509 | 0.8366 | 0.8364 |
| intermediate | scratch | facs_plus_bs_80 | FA | 7c | B2 | 0.3019 | 0.7994 | 0.7729 |
| intermediate | scratch | facs_plus_bs_80 | FA | 3c | B3 | 0.6561 | 0.8401 | 0.8439 |
| intermediate | scratch | facs_plus_bs_80 | FA | 7c | B3 | 0.2845 | 0.8023 | 0.7869 |
| intermediate | tl | raw_136 | MP | 3c | B1 | 0.6290 | 0.7323 | 0.7115 |
| intermediate | tl | raw_136 | MP | 7c | B1 | 0.2722 | 0.7939 | 0.7869 |
| intermediate | tl | raw_136 | MP | 3c | B2 | 0.6878 | 0.8228 | 0.8127 |
| intermediate | tl | raw_136 | MP | 7c | B2 | 0.2733 | 0.7573 | 0.7374 |
| intermediate | tl | raw_136 | MP | 3c | B3 | 0.6910 | 0.8518 | 0.8504 |
| intermediate | tl | raw_136 | MP | 7c | B3 | 0.3012 | 0.8167 | 0.8052 |
| intermediate | tl | facs_28 | MP | 3c | B1 | 0.7142 | 0.8463 | 0.8407 |
| intermediate | tl | facs_28 | MP | 7c | B1 | 0.2842 | 0.8104 | 0.8030 |
| intermediate | tl | facs_28 | MP | 3c | B2 | 0.6108 | 0.7522 | 0.7320 |
| intermediate | tl | facs_28 | MP | 7c | B2 | 0.2881 | 0.8366 | 0.8364 |
| intermediate | tl | facs_28 | MP | 3c | B3 | 0.7229 | 0.8539 | 0.8515 |
| intermediate | tl | facs_28 | MP | 7c | B3 | 0.3000 | 0.8468 | 0.8439 |
| intermediate | tl | blendshape_52 | MP | 3c | B1 | 0.6421 | 0.7896 | 0.7869 |
| intermediate | tl | blendshape_52 | MP | 7c | B1 | 0.2852 | 0.8337 | 0.8375 |
| intermediate | tl | blendshape_52 | MP | 3c | B2 | 0.6229 | 0.7973 | 0.7869 |
| intermediate | tl | blendshape_52 | MP | 7c | B2 | 0.2856 | 0.8131 | 0.8052 |
| intermediate | tl | blendshape_52 | MP | 3c | B3 | 0.6698 | 0.8447 | 0.8439 |
| intermediate | tl | blendshape_52 | MP | 7c | B3 | 0.2958 | 0.8367 | 0.8224 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B1 | 0.6734 | 0.8212 | 0.8095 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B1 | 0.2466 | 0.7575 | 0.7783 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B2 | 0.6018 | 0.6971 | 0.6695 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B2 | 0.2895 | 0.8200 | 0.8138 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B3 | 0.6473 | 0.8182 | 0.8267 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B3 | 0.2834 | 0.7850 | 0.7750 |
| intermediate | tl | raw_136 | FA | 3c | B1 | 0.6935 | 0.8462 | 0.8439 |
| intermediate | tl | raw_136 | FA | 7c | B1 | 0.2944 | 0.8354 | 0.8267 |
| intermediate | tl | raw_136 | FA | 3c | B2 | 0.7177 | 0.8376 | 0.8288 |
| intermediate | tl | raw_136 | FA | 7c | B2 | 0.3023 | 0.8260 | 0.8138 |
| intermediate | tl | raw_136 | FA | 3c | B3 | 0.6937 | 0.8042 | 0.7858 |
| intermediate | tl | raw_136 | FA | 7c | B3 | 0.2447 | 0.7704 | 0.7664 |
| intermediate | tl | facs_28 | FA | 3c | B1 | 0.6868 | 0.8295 | 0.8235 |
| intermediate | tl | facs_28 | FA | 7c | B1 | 0.2476 | 0.7832 | 0.8149 |
| intermediate | tl | facs_28 | FA | 3c | B2 | 0.7525 | 0.8738 | 0.8644 |
| intermediate | tl | facs_28 | FA | 7c | B2 | 0.3085 | 0.8690 | 0.8644 |
| intermediate | tl | facs_28 | FA | 3c | B3 | 0.7156 | 0.8443 | 0.8461 |
| intermediate | tl | facs_28 | FA | 7c | B3 | 0.2845 | 0.8381 | 0.8396 |
| intermediate | tl | facs_plus_bs_80 | FA | 3c | B1 | 0.6986 | 0.8552 | 0.8439 |
| intermediate | tl | facs_plus_bs_80 | FA | 7c | B1 | 0.2797 | 0.8104 | 0.8030 |
| intermediate | tl | facs_plus_bs_80 | FA | 3c | B2 | 0.6440 | 0.7493 | 0.7255 |
| intermediate | tl | facs_plus_bs_80 | FA | 7c | B2 | 0.3101 | 0.8432 | 0.8342 |
| intermediate | tl | facs_plus_bs_80 | FA | 3c | B3 | 0.6850 | 0.8413 | 0.8439 |
| intermediate | tl | facs_plus_bs_80 | FA | 7c | B3 | 0.3033 | 0.8597 | 0.8633 |

### 2.3 Late Fusion (semua feature × variant × source — saat ini hanya raw_136 MP yang implemented)

| Fusion | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| late | scratch | raw_136 | MP | 3c | B1 | 0.4962 | 0.7655 | 0.7966 |
| late | scratch | raw_136 | MP | 7c | B1 | 0.2338 | 0.7746 | 0.7858 |
| late | scratch | raw_136 | MP | 3c | B2 | 0.4725 | 0.6545 | 0.6297 |
| late | scratch | raw_136 | MP | 7c | B2 | 0.1827 | 0.6768 | 0.6416 |
| late | scratch | raw_136 | MP | 3c | B3 | 0.5974 | 0.7589 | 0.7449 |
| late | scratch | raw_136 | MP | 7c | B3 | 0.2551 | 0.7676 | 0.7330 |
| late | scratch | facs_28 | MP | 3c | B1 | 0.5877 | 0.7620 | 0.7460 |
| late | scratch | facs_28 | MP | 7c | B1 | 0.1972 | 0.7126 | 0.7449 |
| late | scratch | facs_28 | MP | 3c | B2 | 0.5849 | 0.7542 | 0.7374 |
| late | scratch | facs_28 | MP | 7c | B2 | 0.2328 | 0.6432 | 0.5414 |
| late | scratch | facs_28 | MP | 3c | B3 | 0.6218 | 0.7783 | 0.7664 |
| late | scratch | facs_28 | MP | 7c | B3 | 0.2341 | 0.6574 | 0.5576 |
| late | scratch | blendshape_52 | MP | 3c | B1 | 0.5042 | 0.7719 | 0.8019 |
| late | scratch | blendshape_52 | MP | 7c | B1 | 0.2988 | 0.8502 | 0.8547 |
| late | scratch | blendshape_52 | MP | 3c | B2 | 0.6237 | 0.7887 | 0.7761 |
| late | scratch | blendshape_52 | MP | 7c | B2 | 0.2281 | 0.7002 | 0.6620 |
| late | scratch | blendshape_52 | MP | 3c | B3 | 0.6620 | 0.8001 | 0.7847 |
| late | scratch | blendshape_52 | MP | 7c | B3 | 0.2403 | 0.7359 | 0.7061 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B1 | 0.5167 | 0.7767 | 0.8052 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B1 | 0.2475 | 0.8238 | 0.8450 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B2 | 0.5776 | 0.7859 | 0.7761 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B2 | 0.2711 | 0.7622 | 0.7341 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B3 | 0.6294 | 0.7999 | 0.7890 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B3 | 0.2730 | 0.7666 | 0.7427 |
| late | scratch | raw_136 | FA | 3c | B1 | 0.7119 | 0.8451 | 0.8471 |
| late | scratch | raw_136 | FA | 7c | B1 | 0.3122 | 0.8445 | 0.8450 |
| late | scratch | raw_136 | FA | 3c | B2 | 0.7131 | 0.8320 | 0.8127 |
| late | scratch | raw_136 | FA | 7c | B2 | 0.2487 | 0.5758 | 0.4715 |
| late | scratch | raw_136 | FA | 3c | B3 | 0.6992 | 0.8417 | 0.8095 |
| late | scratch | raw_136 | FA | 7c | B3 | 0.2989 | 0.7615 | 0.6760 |
| late | scratch | facs_28 | FA | 3c | B1 | 0.7604 | 0.8896 | 0.8881 |
| late | scratch | facs_28 | FA | 7c | B1 | 0.3100 | 0.8593 | 0.8611 |
| late | scratch | facs_28 | FA | 3c | B2 | 0.7489 | 0.8622 | 0.8536 |
| late | scratch | facs_28 | FA | 7c | B2 | 0.3262 | 0.8693 | 0.8525 |
| late | scratch | facs_28 | FA | 3c | B3 | 0.7489 | 0.8622 | 0.8536 |
| late | scratch | facs_28 | FA | 7c | B3 | 0.3309 | 0.8830 | 0.8644 |
| late | scratch | facs_plus_bs_80 | FA | 3c | B1 | 0.7345 | 0.8864 | 0.8902 |
| late | scratch | facs_plus_bs_80 | FA | 7c | B1 | 0.3042 | 0.8523 | 0.8579 |
| late | scratch | facs_plus_bs_80 | FA | 3c | B2 | 0.6717 | 0.8333 | 0.8181 |
| late | scratch | facs_plus_bs_80 | FA | 7c | B2 | 0.3043 | 0.7844 | 0.7578 |
| late | scratch | facs_plus_bs_80 | FA | 3c | B3 | 0.6513 | 0.8255 | 0.7998 |
| late | scratch | facs_plus_bs_80 | FA | 7c | B3 | 0.3171 | 0.7981 | 0.7804 |
| late | tl | raw_136 | MP | 3c | B1 | 0.5087 | 0.7867 | 0.8202 |
| late | tl | raw_136 | MP | 7c | B1 | 0.2325 | 0.7738 | 0.7804 |
| late | tl | raw_136 | MP | 3c | B2 | 0.7090 | 0.8329 | 0.8181 |
| late | tl | raw_136 | MP | 7c | B2 | 0.2828 | 0.8122 | 0.7804 |
| late | tl | raw_136 | MP | 3c | B3 | 0.7067 | 0.8190 | 0.8041 |
| late | tl | raw_136 | MP | 7c | B3 | 0.2902 | 0.8030 | 0.7890 |
| late | tl | facs_28 | MP | 3c | B1 | 0.5594 | 0.7314 | 0.7061 |
| late | tl | facs_28 | MP | 7c | B1 | 0.1972 | 0.7126 | 0.7449 |
| late | tl | facs_28 | MP | 3c | B2 | 0.7042 | 0.8296 | 0.8159 |
| late | tl | facs_28 | MP | 7c | B2 | 0.2866 | 0.7876 | 0.7449 |
| late | tl | facs_28 | MP | 3c | B3 | 0.7106 | 0.8282 | 0.8138 |
| late | tl | facs_28 | MP | 7c | B3 | 0.2792 | 0.7806 | 0.7266 |
| late | tl | blendshape_52 | MP | 3c | B1 | 0.5713 | 0.8015 | 0.8192 |
| late | tl | blendshape_52 | MP | 7c | B1 | 0.2986 | 0.8406 | 0.8439 |
| late | tl | blendshape_52 | MP | 3c | B2 | 0.6868 | 0.8170 | 0.8030 |
| late | tl | blendshape_52 | MP | 7c | B2 | 0.2813 | 0.7958 | 0.7707 |
| late | tl | blendshape_52 | MP | 3c | B3 | 0.6927 | 0.8229 | 0.8084 |
| late | tl | blendshape_52 | MP | 7c | B3 | 0.2834 | 0.8087 | 0.7879 |
| late | tl | facs_plus_bs_80 | MP | 3c | B1 | 0.6680 | 0.8390 | 0.8461 |
| late | tl | facs_plus_bs_80 | MP | 7c | B1 | 0.2471 | 0.8231 | 0.8439 |
| late | tl | facs_plus_bs_80 | MP | 3c | B2 | 0.6749 | 0.8188 | 0.8062 |
| late | tl | facs_plus_bs_80 | MP | 7c | B2 | 0.2734 | 0.7663 | 0.7374 |
| late | tl | facs_plus_bs_80 | MP | 3c | B3 | 0.6550 | 0.8084 | 0.7944 |
| late | tl | facs_plus_bs_80 | MP | 7c | B3 | 0.2778 | 0.7752 | 0.7470 |
| late | tl | raw_136 | FA | 3c | B1 | 0.7119 | 0.8451 | 0.8471 |
| late | tl | raw_136 | FA | 7c | B1 | 0.3159 | 0.8505 | 0.8525 |
| late | tl | raw_136 | FA | 3c | B2 | 0.7482 | 0.8686 | 0.8558 |
| late | tl | raw_136 | FA | 7c | B2 | 0.2970 | 0.6947 | 0.5974 |
| late | tl | raw_136 | FA | 3c | B3 | 0.7525 | 0.8806 | 0.8644 |
| late | tl | raw_136 | FA | 7c | B3 | 0.3264 | 0.8181 | 0.7664 |
| late | tl | facs_28 | FA | 3c | B1 | 0.7604 | 0.8914 | 0.8891 |
| late | tl | facs_28 | FA | 7c | B1 | 0.3095 | 0.8580 | 0.8579 |
| late | tl | facs_28 | FA | 3c | B2 | 0.7489 | 0.8622 | 0.8536 |
| late | tl | facs_28 | FA | 7c | B2 | 0.3172 | 0.8681 | 0.8418 |
| late | tl | facs_28 | FA | 3c | B3 | 0.7489 | 0.8622 | 0.8536 |
| late | tl | facs_28 | FA | 7c | B3 | 0.3173 | 0.8671 | 0.8418 |
| late | tl | facs_plus_bs_80 | FA | 3c | B1 | 0.7345 | 0.8864 | 0.8902 |
| late | tl | facs_plus_bs_80 | FA | 7c | B1 | 0.2970 | 0.8451 | 0.8504 |
| late | tl | facs_plus_bs_80 | FA | 3c | B2 | 0.6568 | 0.8262 | 0.8009 |
| late | tl | facs_plus_bs_80 | FA | 7c | B2 | 0.3043 | 0.7844 | 0.7578 |
| late | tl | facs_plus_bs_80 | FA | 3c | B3 | 0.6917 | 0.8422 | 0.8245 |
| late | tl | facs_plus_bs_80 | FA | 7c | B3 | 0.3332 | 0.8190 | 0.8019 |

---

## 3. KDEF 7c (Cross-dataset Benchmark)

Source data: `models/benchmark/kdef_7class/{3,7}class/Unified/`.

### 3.1 Landmark (MP only)

| Feature | Source | Arch | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| raw_136 | MP | FCNN | 3c | B1 | 0.6937 | 0.6885 | 0.6973 |
| raw_136 | MP | FCNN | 7c | B1 | 0.5078 | 0.5078 | 0.5476 |
| raw_136 | MP | FCNN | 3c | B2 | 0.6646 | 0.7203 | 0.7279 |
| raw_136 | MP | FCNN | 7c | B2 | 0.5485 | 0.5485 | 0.5612 |
| raw_136 | MP | FCNN | 3c | B3 | 0.6896 | 0.7136 | 0.7143 |
| raw_136 | MP | FCNN | 7c | B3 | 0.5742 | 0.5742 | 0.6020 |
| raw_136 | MP | CNN1D | 3c | B1 | 0.7099 | 0.7376 | 0.7381 |
| raw_136 | MP | CNN1D | 7c | B1 | 0.6042 | 0.6042 | 0.6190 |
| raw_136 | MP | CNN1D | 3c | B2 | 0.6041 | 0.5954 | 0.6020 |
| raw_136 | MP | CNN1D | 7c | B2 | 0.5851 | 0.5851 | 0.5952 |
| raw_136 | MP | CNN1D | 3c | B3 | 0.6936 | 0.7033 | 0.7041 |
| raw_136 | MP | CNN1D | 7c | B3 | 0.6015 | 0.6015 | 0.6259 |
| facs_28 | MP | FCNN | 3c | B1 | 0.7223 | 0.7552 | 0.7551 |
| facs_28 | MP | FCNN | 7c | B1 | 0.6066 | 0.6066 | 0.6293 |
| facs_28 | MP | FCNN | 3c | B2 | 0.7260 | 0.7363 | 0.7381 |
| facs_28 | MP | FCNN | 7c | B2 | 0.5927 | 0.5927 | 0.6122 |
| facs_28 | MP | FCNN | 3c | B3 | 0.6912 | 0.7052 | 0.7041 |
| facs_28 | MP | FCNN | 7c | B3 | 0.6087 | 0.6087 | 0.6259 |
| facs_28 | MP | CNN1D | 3c | B1 | 0.7063 | 0.7359 | 0.7381 |
| facs_28 | MP | CNN1D | 7c | B1 | 0.6341 | 0.6341 | 0.6395 |
| facs_28 | MP | CNN1D | 3c | B2 | 0.6512 | 0.6450 | 0.6565 |
| facs_28 | MP | CNN1D | 7c | B2 | 0.5836 | 0.5836 | 0.6020 |
| facs_28 | MP | CNN1D | 3c | B3 | 0.6434 | 0.6409 | 0.6463 |
| facs_28 | MP | CNN1D | 7c | B3 | 0.5303 | 0.5303 | 0.5680 |
| blendshape_52 | MP | FCNN | 3c | B1 | 0.6835 | 0.7389 | 0.7483 |
| blendshape_52 | MP | FCNN | 7c | B1 | 0.6015 | 0.6015 | 0.6122 |
| blendshape_52 | MP | FCNN | 3c | B2 | 0.7659 | 0.7730 | 0.7721 |
| blendshape_52 | MP | FCNN | 7c | B2 | 0.5982 | 0.5982 | 0.6122 |
| blendshape_52 | MP | FCNN | 3c | B3 | 0.7462 | 0.7476 | 0.7483 |
| blendshape_52 | MP | FCNN | 7c | B3 | 0.6051 | 0.6051 | 0.6190 |
| blendshape_52 | MP | CNN1D | 3c | B1 | 0.7480 | 0.7653 | 0.7653 |
| blendshape_52 | MP | CNN1D | 7c | B1 | 0.5821 | 0.5821 | 0.5918 |
| blendshape_52 | MP | CNN1D | 3c | B2 | 0.7088 | 0.7219 | 0.7211 |
| blendshape_52 | MP | CNN1D | 7c | B2 | 0.6228 | 0.6228 | 0.6395 |
| blendshape_52 | MP | CNN1D | 3c | B3 | 0.7145 | 0.7338 | 0.7313 |
| blendshape_52 | MP | CNN1D | 7c | B3 | 0.5996 | 0.5996 | 0.6156 |
| facs_plus_bs_80 | MP | FCNN | 3c | B1 | 0.7751 | 0.8094 | 0.8129 |
| facs_plus_bs_80 | MP | FCNN | 7c | B1 | 0.5966 | 0.5966 | 0.6122 |
| facs_plus_bs_80 | MP | FCNN | 3c | B2 | 0.7286 | 0.7615 | 0.7653 |
| facs_plus_bs_80 | MP | FCNN | 7c | B2 | 0.5492 | 0.5492 | 0.5680 |
| facs_plus_bs_80 | MP | FCNN | 3c | B3 | 0.7584 | 0.7730 | 0.7721 |
| facs_plus_bs_80 | MP | FCNN | 7c | B3 | 0.5884 | 0.5884 | 0.6122 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B1 | 0.7000 | 0.7610 | 0.7755 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B1 | 0.5590 | 0.5590 | 0.5714 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B2 | 0.7500 | 0.7615 | 0.7585 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B2 | 0.5654 | 0.5654 | 0.5816 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B3 | 0.7451 | 0.7579 | 0.7551 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B3 | 0.5490 | 0.5490 | 0.5714 |

### 3.2 Image

| Arch | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| CNN_SCRATCH | 3c | B1 | 0.7845 | 0.7888 | 0.7891 |
| CNN_SCRATCH | 7c | B1 | 0.7920 | 0.7920 | 0.7959 |
| CNN_SCRATCH | 3c | B2 | 0.7736 | 0.7756 | 0.7755 |
| CNN_SCRATCH | 7c | B2 | 0.8104 | 0.8104 | 0.8129 |
| CNN_SCRATCH | 3c | B3 | 0.8806 | 0.8849 | 0.8844 |
| CNN_SCRATCH | 7c | B3 | 0.8490 | 0.8490 | 0.8503 |
| CNN_TL | 3c | B1 | 0.9454 | 0.9491 | 0.9490 |
| CNN_TL | 7c | B1 | 0.8966 | 0.8966 | 0.8980 |
| CNN_TL | 3c | B2 | 0.9403 | 0.9424 | 0.9422 |
| CNN_TL | 7c | B2 | 0.8761 | 0.8761 | 0.8776 |
| CNN_TL | 3c | B3 | 0.9486 | 0.9525 | 0.9524 |
| CNN_TL | 7c | B3 | 0.8495 | 0.8495 | 0.8537 |

### 3.3 Early Fusion (raw_136, MP source) — concat vs gated

| Fusion | Mode | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| early | concat | scratch | raw_136 | MP | 3c | B1 | 0.7926 | 0.8144 | 0.8129 |
| early | concat | scratch | raw_136 | MP | 7c | B1 | 0.7305 | 0.7305 | 0.7381 |
| early | concat | scratch | raw_136 | MP | 3c | B2 | 0.7742 | 0.7869 | 0.7857 |
| early | concat | scratch | raw_136 | MP | 7c | B2 | 0.6833 | 0.6833 | 0.6837 |
| early | concat | scratch | raw_136 | MP | 3c | B3 | 0.8154 | 0.8308 | 0.8299 |
| early | concat | scratch | raw_136 | MP | 7c | B3 | 0.7366 | 0.7366 | 0.7381 |
| early | concat | tl | raw_136 | MP | 3c | B1 | 0.9441 | 0.9491 | 0.9490 |
| early | concat | tl | raw_136 | MP | 7c | B1 | 0.9140 | 0.9140 | 0.9150 |
| early | concat | tl | raw_136 | MP | 3c | B2 | 0.9626 | 0.9565 | 0.9558 |
| early | concat | tl | raw_136 | MP | 7c | B2 | 0.9066 | 0.9066 | 0.9082 |
| early | concat | tl | raw_136 | MP | 3c | B3 | 0.9485 | 0.9492 | 0.9490 |
| early | concat | tl | raw_136 | MP | 7c | B3 | 0.8789 | 0.8789 | 0.8810 |
| early | gated | scratch | raw_136 | MP | 3c | B1 | 0.7679 | 0.7942 | 0.7925 |
| early | gated | scratch | raw_136 | MP | 7c | B1 | 0.7372 | 0.7372 | 0.7449 |
| early | gated | scratch | raw_136 | MP | 3c | B2 | 0.7800 | 0.8045 | 0.8027 |
| early | gated | scratch | raw_136 | MP | 7c | B2 | 0.7323 | 0.7323 | 0.7381 |
| early | gated | scratch | raw_136 | MP | 3c | B3 | 0.8048 | 0.8335 | 0.8333 |
| early | gated | scratch | raw_136 | MP | 7c | B3 | 0.7502 | 0.7502 | 0.7585 |
| early | gated | tl | raw_136 | MP | 3c | B1 | 0.9009 | 0.9181 | 0.9184 |
| early | gated | tl | raw_136 | MP | 7c | B1 | 0.8975 | 0.8975 | 0.8980 |
| early | gated | tl | raw_136 | MP | 3c | B2 | 0.9509 | 0.9492 | 0.9490 |
| early | gated | tl | raw_136 | MP | 7c | B2 | 0.8532 | 0.8532 | 0.8537 |
| early | gated | tl | raw_136 | MP | 3c | B3 | 0.9121 | 0.9193 | 0.9184 |
| early | gated | tl | raw_136 | MP | 7c | B3 | 0.8802 | 0.8802 | 0.8810 |

### 3.4 Intermediate Fusion (semua feature × variant, MP source)

| Fusion | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| intermediate | scratch | raw_136 | MP | 3c | B1 | 0.7817 | 0.7989 | 0.7959 |
| intermediate | scratch | raw_136 | MP | 7c | B1 | 0.7305 | 0.7305 | 0.7313 |
| intermediate | scratch | raw_136 | MP | 3c | B2 | 0.7404 | 0.7640 | 0.7619 |
| intermediate | scratch | raw_136 | MP | 7c | B2 | 0.7208 | 0.7208 | 0.7279 |
| intermediate | scratch | raw_136 | MP | 3c | B3 | 0.7793 | 0.8097 | 0.8129 |
| intermediate | scratch | raw_136 | MP | 7c | B3 | 0.7947 | 0.7947 | 0.7959 |
| intermediate | scratch | facs_28 | MP | 3c | B1 | 0.7874 | 0.8134 | 0.8129 |
| intermediate | scratch | facs_28 | MP | 7c | B1 | 0.7704 | 0.7704 | 0.7721 |
| intermediate | scratch | facs_28 | MP | 3c | B2 | 0.8192 | 0.8333 | 0.8333 |
| intermediate | scratch | facs_28 | MP | 7c | B2 | 0.7677 | 0.7677 | 0.7755 |
| intermediate | scratch | facs_28 | MP | 3c | B3 | 0.7616 | 0.7779 | 0.7789 |
| intermediate | scratch | facs_28 | MP | 7c | B3 | 0.7172 | 0.7172 | 0.7279 |
| intermediate | scratch | blendshape_52 | MP | 3c | B1 | 0.7748 | 0.7987 | 0.7993 |
| intermediate | scratch | blendshape_52 | MP | 7c | B1 | 0.7211 | 0.7211 | 0.7245 |
| intermediate | scratch | blendshape_52 | MP | 3c | B2 | 0.7505 | 0.7809 | 0.7823 |
| intermediate | scratch | blendshape_52 | MP | 7c | B2 | 0.6720 | 0.6720 | 0.7007 |
| intermediate | scratch | blendshape_52 | MP | 3c | B3 | 0.8029 | 0.8139 | 0.8129 |
| intermediate | scratch | blendshape_52 | MP | 7c | B3 | 0.7106 | 0.7106 | 0.7177 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B1 | 0.7678 | 0.7888 | 0.7891 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B1 | 0.7159 | 0.7159 | 0.7245 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B2 | 0.7501 | 0.7755 | 0.7755 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B2 | 0.6992 | 0.6992 | 0.7143 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B3 | 0.7647 | 0.7849 | 0.7857 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B3 | 0.6701 | 0.6701 | 0.6871 |
| intermediate | tl | raw_136 | MP | 3c | B1 | 0.9405 | 0.9424 | 0.9422 |
| intermediate | tl | raw_136 | MP | 7c | B1 | 0.8632 | 0.8632 | 0.8639 |
| intermediate | tl | raw_136 | MP | 3c | B2 | 0.9558 | 0.9561 | 0.9558 |
| intermediate | tl | raw_136 | MP | 7c | B2 | 0.8635 | 0.8635 | 0.8673 |
| intermediate | tl | raw_136 | MP | 3c | B3 | 0.9363 | 0.9356 | 0.9354 |
| intermediate | tl | raw_136 | MP | 7c | B3 | 0.8639 | 0.8639 | 0.8639 |
| intermediate | tl | facs_28 | MP | 3c | B1 | 0.9527 | 0.9526 | 0.9524 |
| intermediate | tl | facs_28 | MP | 7c | B1 | 0.8823 | 0.8823 | 0.8844 |
| intermediate | tl | facs_28 | MP | 3c | B2 | 0.9396 | 0.9354 | 0.9354 |
| intermediate | tl | facs_28 | MP | 7c | B2 | 0.8697 | 0.8697 | 0.8673 |
| intermediate | tl | facs_28 | MP | 3c | B3 | 0.9497 | 0.9462 | 0.9456 |
| intermediate | tl | facs_28 | MP | 7c | B3 | 0.8394 | 0.8394 | 0.8435 |
| intermediate | tl | blendshape_52 | MP | 3c | B1 | 0.9176 | 0.9256 | 0.9252 |
| intermediate | tl | blendshape_52 | MP | 7c | B1 | 0.8562 | 0.8562 | 0.8571 |
| intermediate | tl | blendshape_52 | MP | 3c | B2 | 0.9313 | 0.9323 | 0.9320 |
| intermediate | tl | blendshape_52 | MP | 7c | B2 | 0.8432 | 0.8432 | 0.8503 |
| intermediate | tl | blendshape_52 | MP | 3c | B3 | 0.9468 | 0.9429 | 0.9422 |
| intermediate | tl | blendshape_52 | MP | 7c | B3 | 0.8434 | 0.8434 | 0.8435 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B1 | 0.9326 | 0.9356 | 0.9354 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B1 | 0.8799 | 0.8799 | 0.8810 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B2 | 0.9387 | 0.9424 | 0.9422 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B2 | 0.8736 | 0.8736 | 0.8776 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B3 | 0.9553 | 0.9529 | 0.9524 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B3 | 0.8591 | 0.8591 | 0.8639 |

### 3.5 Late Fusion (semua feature × variant, MP source)

| Fusion | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| late | scratch | raw_136 | MP | 3c | B1 | 0.8487 | 0.8478 | 0.8469 |
| late | scratch | raw_136 | MP | 7c | B1 | 0.7774 | 0.7774 | 0.7823 |
| late | scratch | raw_136 | MP | 3c | B2 | 0.8095 | 0.8245 | 0.8231 |
| late | scratch | raw_136 | MP | 7c | B2 | 0.7922 | 0.7922 | 0.7959 |
| late | scratch | raw_136 | MP | 3c | B3 | 0.8453 | 0.8646 | 0.8639 |
| late | scratch | raw_136 | MP | 7c | B3 | 0.8167 | 0.8167 | 0.8197 |
| late | scratch | facs_28 | MP | 3c | B1 | 0.7629 | 0.7854 | 0.7857 |
| late | scratch | facs_28 | MP | 7c | B1 | 0.7930 | 0.7930 | 0.7993 |
| late | scratch | facs_28 | MP | 3c | B2 | 0.8421 | 0.8474 | 0.8469 |
| late | scratch | facs_28 | MP | 7c | B2 | 0.7758 | 0.7758 | 0.7823 |
| late | scratch | facs_28 | MP | 3c | B3 | 0.8508 | 0.8682 | 0.8673 |
| late | scratch | facs_28 | MP | 7c | B3 | 0.8203 | 0.8203 | 0.8231 |
| late | scratch | blendshape_52 | MP | 3c | B1 | 0.8422 | 0.8506 | 0.8503 |
| late | scratch | blendshape_52 | MP | 7c | B1 | 0.8074 | 0.8074 | 0.8129 |
| late | scratch | blendshape_52 | MP | 3c | B2 | 0.8235 | 0.8343 | 0.8333 |
| late | scratch | blendshape_52 | MP | 7c | B2 | 0.7434 | 0.7434 | 0.7483 |
| late | scratch | blendshape_52 | MP | 3c | B3 | 0.8480 | 0.8680 | 0.8673 |
| late | scratch | blendshape_52 | MP | 7c | B3 | 0.7901 | 0.7901 | 0.7925 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B1 | 0.7870 | 0.8107 | 0.8129 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B1 | 0.7945 | 0.7945 | 0.7993 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B2 | 0.8123 | 0.8298 | 0.8299 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B2 | 0.7598 | 0.7598 | 0.7653 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B3 | 0.8518 | 0.8704 | 0.8707 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B3 | 0.7963 | 0.7963 | 0.7993 |
| late | tl | raw_136 | MP | 3c | B1 | 0.9343 | 0.9357 | 0.9354 |
| late | tl | raw_136 | MP | 7c | B1 | 0.8995 | 0.8995 | 0.9014 |
| late | tl | raw_136 | MP | 3c | B2 | 0.9259 | 0.9322 | 0.9320 |
| late | tl | raw_136 | MP | 7c | B2 | 0.8546 | 0.8546 | 0.8605 |
| late | tl | raw_136 | MP | 3c | B3 | 0.9508 | 0.9526 | 0.9524 |
| late | tl | raw_136 | MP | 7c | B3 | 0.8505 | 0.8505 | 0.8571 |
| late | tl | facs_28 | MP | 3c | B1 | 0.9390 | 0.9391 | 0.9388 |
| late | tl | facs_28 | MP | 7c | B1 | 0.8926 | 0.8926 | 0.8946 |
| late | tl | facs_28 | MP | 3c | B2 | 0.9303 | 0.9323 | 0.9320 |
| late | tl | facs_28 | MP | 7c | B2 | 0.8343 | 0.8343 | 0.8435 |
| late | tl | facs_28 | MP | 3c | B3 | 0.9581 | 0.9562 | 0.9558 |
| late | tl | facs_28 | MP | 7c | B3 | 0.8461 | 0.8461 | 0.8537 |
| late | tl | blendshape_52 | MP | 3c | B1 | 0.9362 | 0.9358 | 0.9354 |
| late | tl | blendshape_52 | MP | 7c | B1 | 0.8895 | 0.8895 | 0.8912 |
| late | tl | blendshape_52 | MP | 3c | B2 | 0.9324 | 0.9324 | 0.9320 |
| late | tl | blendshape_52 | MP | 7c | B2 | 0.8524 | 0.8524 | 0.8605 |
| late | tl | blendshape_52 | MP | 3c | B3 | 0.9529 | 0.9528 | 0.9524 |
| late | tl | blendshape_52 | MP | 7c | B3 | 0.8595 | 0.8595 | 0.8639 |
| late | tl | facs_plus_bs_80 | MP | 3c | B1 | 0.9313 | 0.9323 | 0.9320 |
| late | tl | facs_plus_bs_80 | MP | 7c | B1 | 0.9069 | 0.9069 | 0.9082 |
| late | tl | facs_plus_bs_80 | MP | 3c | B2 | 0.9253 | 0.9321 | 0.9320 |
| late | tl | facs_plus_bs_80 | MP | 7c | B2 | 0.8457 | 0.8457 | 0.8537 |
| late | tl | facs_plus_bs_80 | MP | 3c | B3 | 0.9558 | 0.9561 | 0.9558 |
| late | tl | facs_plus_bs_80 | MP | 7c | B3 | 0.8615 | 0.8615 | 0.8673 |

---

## 4. RAF-DB 7c (Cross-dataset Benchmark)

Source data: `models/benchmark/rafdb_7class/{3,7}class/Unified/`.

### 4.1 Landmark (MP only)

| Feature | Source | Arch | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| raw_136 | MP | FCNN | 3c | B1 | 0.6724 | 0.7037 | 0.7050 |
| raw_136 | MP | FCNN | 7c | B1 | 0.5373 | 0.6532 | 0.6625 |
| raw_136 | MP | FCNN | 3c | B2 | 0.6522 | 0.6817 | 0.6761 |
| raw_136 | MP | FCNN | 7c | B2 | 0.5197 | 0.6350 | 0.6298 |
| raw_136 | MP | FCNN | 3c | B3 | 0.6603 | 0.6892 | 0.6822 |
| raw_136 | MP | FCNN | 7c | B3 | 0.4768 | 0.5923 | 0.6124 |
| raw_136 | MP | CNN1D | 3c | B1 | 0.6720 | 0.7050 | 0.7030 |
| raw_136 | MP | CNN1D | 7c | B1 | 0.5015 | 0.6266 | 0.6488 |
| raw_136 | MP | CNN1D | 3c | B2 | 0.6515 | 0.6817 | 0.6792 |
| raw_136 | MP | CNN1D | 7c | B2 | 0.5028 | 0.6158 | 0.6063 |
| raw_136 | MP | CNN1D | 3c | B3 | 0.6489 | 0.6786 | 0.6747 |
| raw_136 | MP | CNN1D | 7c | B3 | 0.4735 | 0.5847 | 0.5746 |
| facs_28 | MP | FCNN | 3c | B1 | 0.6856 | 0.7158 | 0.7139 |
| facs_28 | MP | FCNN | 7c | B1 | 0.4734 | 0.6211 | 0.6373 |
| facs_28 | MP | FCNN | 3c | B2 | 0.6871 | 0.7129 | 0.7105 |
| facs_28 | MP | FCNN | 7c | B2 | 0.4845 | 0.5926 | 0.5841 |
| facs_28 | MP | FCNN | 3c | B3 | 0.6465 | 0.6706 | 0.6652 |
| facs_28 | MP | FCNN | 7c | B3 | 0.4942 | 0.6008 | 0.5920 |
| facs_28 | MP | CNN1D | 3c | B1 | 0.6709 | 0.7003 | 0.7027 |
| facs_28 | MP | CNN1D | 7c | B1 | 0.4722 | 0.6133 | 0.6400 |
| facs_28 | MP | CNN1D | 3c | B2 | 0.6503 | 0.6794 | 0.6822 |
| facs_28 | MP | CNN1D | 7c | B2 | 0.4762 | 0.5887 | 0.5834 |
| facs_28 | MP | CNN1D | 3c | B3 | 0.6441 | 0.6729 | 0.6744 |
| facs_28 | MP | CNN1D | 7c | B3 | 0.4762 | 0.5773 | 0.5715 |
| blendshape_52 | MP | FCNN | 3c | B1 | 0.6787 | 0.7080 | 0.7071 |
| blendshape_52 | MP | FCNN | 7c | B1 | 0.5033 | 0.6365 | 0.6526 |
| blendshape_52 | MP | FCNN | 3c | B2 | 0.6720 | 0.7004 | 0.6965 |
| blendshape_52 | MP | FCNN | 7c | B2 | 0.4717 | 0.5938 | 0.5780 |
| blendshape_52 | MP | FCNN | 3c | B3 | 0.6656 | 0.6925 | 0.6921 |
| blendshape_52 | MP | FCNN | 7c | B3 | 0.4732 | 0.5914 | 0.5763 |
| blendshape_52 | MP | CNN1D | 3c | B1 | 0.6628 | 0.6937 | 0.6928 |
| blendshape_52 | MP | CNN1D | 7c | B1 | 0.4822 | 0.6267 | 0.6396 |
| blendshape_52 | MP | CNN1D | 3c | B2 | 0.6537 | 0.6857 | 0.6856 |
| blendshape_52 | MP | CNN1D | 7c | B2 | 0.4793 | 0.6007 | 0.5841 |
| blendshape_52 | MP | CNN1D | 3c | B3 | 0.6513 | 0.6836 | 0.6826 |
| blendshape_52 | MP | CNN1D | 7c | B3 | 0.4546 | 0.5809 | 0.5664 |
| facs_plus_bs_80 | MP | FCNN | 3c | B1 | 0.6915 | 0.7233 | 0.7238 |
| facs_plus_bs_80 | MP | FCNN | 7c | B1 | 0.5418 | 0.6580 | 0.6628 |
| facs_plus_bs_80 | MP | FCNN | 3c | B2 | 0.6843 | 0.7122 | 0.7088 |
| facs_plus_bs_80 | MP | FCNN | 7c | B2 | 0.5236 | 0.6344 | 0.6192 |
| facs_plus_bs_80 | MP | FCNN | 3c | B3 | 0.6777 | 0.7062 | 0.6999 |
| facs_plus_bs_80 | MP | FCNN | 7c | B3 | 0.5122 | 0.6241 | 0.6097 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B1 | 0.6683 | 0.7029 | 0.7061 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B1 | 0.4431 | 0.6027 | 0.6260 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B2 | 0.6578 | 0.6848 | 0.6775 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B2 | 0.4983 | 0.6084 | 0.6005 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B3 | 0.6716 | 0.7009 | 0.7003 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B3 | 0.4836 | 0.6047 | 0.5991 |

### 4.2 Image

| Arch | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| CNN_SCRATCH | 3c | B1 | 0.7809 | 0.8055 | 0.8045 |
| CNN_SCRATCH | 7c | B1 | 0.6887 | 0.7816 | 0.7844 |
| CNN_SCRATCH | 3c | B2 | 0.7851 | 0.8104 | 0.8093 |
| CNN_SCRATCH | 7c | B2 | 0.6862 | 0.7802 | 0.7813 |
| CNN_SCRATCH | 3c | B3 | 0.8180 | 0.8376 | 0.8355 |
| CNN_SCRATCH | 7c | B3 | 0.7227 | 0.8092 | 0.8072 |
| CNN_TL | 3c | B1 | 0.8254 | 0.8452 | 0.8454 |
| CNN_TL | 7c | B1 | 0.7255 | 0.8137 | 0.8151 |
| CNN_TL | 3c | B2 | 0.8093 | 0.8321 | 0.8311 |
| CNN_TL | 7c | B2 | 0.7137 | 0.8026 | 0.8072 |
| CNN_TL | 3c | B3 | 0.8444 | 0.8613 | 0.8600 |
| CNN_TL | 7c | B3 | 0.7355 | 0.8232 | 0.8225 |

### 4.3 Early Fusion (raw_136, MP source) — concat vs gated

| Fusion | Mode | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| early | concat | scratch | raw_136 | MP | 3c | B1 | 0.7723 | 0.7969 | 0.7973 |
| early | concat | scratch | raw_136 | MP | 7c | B1 | 0.6595 | 0.7588 | 0.7636 |
| early | concat | scratch | raw_136 | MP | 3c | B2 | 0.7923 | 0.8150 | 0.8140 |
| early | concat | scratch | raw_136 | MP | 7c | B2 | 0.6688 | 0.7622 | 0.7646 |
| early | concat | scratch | raw_136 | MP | 3c | B3 | 0.8231 | 0.8431 | 0.8420 |
| early | concat | scratch | raw_136 | MP | 7c | B3 | 0.6977 | 0.7927 | 0.7919 |
| early | concat | tl | raw_136 | MP | 3c | B1 | 0.8041 | 0.8283 | 0.8290 |
| early | concat | tl | raw_136 | MP | 7c | B1 | 0.6743 | 0.7714 | 0.7684 |
| early | concat | tl | raw_136 | MP | 3c | B2 | 0.8150 | 0.8352 | 0.8358 |
| early | concat | tl | raw_136 | MP | 7c | B2 | 0.7006 | 0.7920 | 0.7960 |
| early | concat | tl | raw_136 | MP | 3c | B3 | 0.8216 | 0.8405 | 0.8392 |
| early | concat | tl | raw_136 | MP | 7c | B3 | 0.7183 | 0.7971 | 0.7977 |
| early | gated | scratch | raw_136 | MP | 3c | B1 | 0.7592 | 0.7853 | 0.7841 |
| early | gated | scratch | raw_136 | MP | 7c | B1 | 0.6649 | 0.7619 | 0.7616 |
| early | gated | scratch | raw_136 | MP | 3c | B2 | 0.7554 | 0.7829 | 0.7841 |
| early | gated | scratch | raw_136 | MP | 7c | B2 | 0.6557 | 0.7444 | 0.7483 |
| early | gated | scratch | raw_136 | MP | 3c | B3 | 0.8213 | 0.8426 | 0.8423 |
| early | gated | scratch | raw_136 | MP | 7c | B3 | 0.7139 | 0.8006 | 0.7970 |
| early | gated | tl | raw_136 | MP | 3c | B1 | 0.8072 | 0.8296 | 0.8280 |
| early | gated | tl | raw_136 | MP | 7c | B1 | 0.6813 | 0.7736 | 0.7772 |
| early | gated | tl | raw_136 | MP | 3c | B2 | 0.8103 | 0.8327 | 0.8324 |
| early | gated | tl | raw_136 | MP | 7c | B2 | 0.6979 | 0.7931 | 0.7946 |
| early | gated | tl | raw_136 | MP | 3c | B3 | 0.8278 | 0.8482 | 0.8481 |
| early | gated | tl | raw_136 | MP | 7c | B3 | 0.7273 | 0.8152 | 0.8185 |

### 4.4 Intermediate Fusion (semua feature × variant, MP source)

| Fusion | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| intermediate | scratch | raw_136 | MP | 3c | B1 | 0.7675 | 0.7946 | 0.7919 |
| intermediate | scratch | raw_136 | MP | 7c | B1 | 0.6774 | 0.7742 | 0.7762 |
| intermediate | scratch | raw_136 | MP | 3c | B2 | 0.7645 | 0.7904 | 0.7905 |
| intermediate | scratch | raw_136 | MP | 7c | B2 | 0.6383 | 0.7300 | 0.7193 |
| intermediate | scratch | raw_136 | MP | 3c | B3 | 0.8025 | 0.8279 | 0.8277 |
| intermediate | scratch | raw_136 | MP | 7c | B3 | 0.7088 | 0.7909 | 0.7899 |
| intermediate | scratch | facs_28 | MP | 3c | B1 | 0.7710 | 0.8000 | 0.8014 |
| intermediate | scratch | facs_28 | MP | 7c | B1 | 0.6736 | 0.7633 | 0.7653 |
| intermediate | scratch | facs_28 | MP | 3c | B2 | 0.7667 | 0.7914 | 0.7909 |
| intermediate | scratch | facs_28 | MP | 7c | B2 | 0.6666 | 0.7591 | 0.7616 |
| intermediate | scratch | facs_28 | MP | 3c | B3 | 0.7686 | 0.7954 | 0.7939 |
| intermediate | scratch | facs_28 | MP | 7c | B3 | 0.6624 | 0.7577 | 0.7592 |
| intermediate | scratch | blendshape_52 | MP | 3c | B1 | 0.7667 | 0.7946 | 0.7950 |
| intermediate | scratch | blendshape_52 | MP | 7c | B1 | 0.6663 | 0.7683 | 0.7715 |
| intermediate | scratch | blendshape_52 | MP | 3c | B2 | 0.7637 | 0.7901 | 0.7881 |
| intermediate | scratch | blendshape_52 | MP | 7c | B2 | 0.6677 | 0.7639 | 0.7670 |
| intermediate | scratch | blendshape_52 | MP | 3c | B3 | 0.7655 | 0.7923 | 0.7943 |
| intermediate | scratch | blendshape_52 | MP | 7c | B3 | 0.6699 | 0.7603 | 0.7609 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B1 | 0.7578 | 0.7860 | 0.7851 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B1 | 0.6625 | 0.7706 | 0.7694 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B2 | 0.7617 | 0.7874 | 0.7875 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B2 | 0.6745 | 0.7626 | 0.7653 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B3 | 0.7675 | 0.7944 | 0.7929 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B3 | 0.6514 | 0.7503 | 0.7500 |
| intermediate | tl | raw_136 | MP | 3c | B1 | 0.8273 | 0.8473 | 0.8474 |
| intermediate | tl | raw_136 | MP | 7c | B1 | 0.7204 | 0.8112 | 0.8110 |
| intermediate | tl | raw_136 | MP | 3c | B2 | 0.8019 | 0.8248 | 0.8239 |
| intermediate | tl | raw_136 | MP | 7c | B2 | 0.7222 | 0.8151 | 0.8164 |
| intermediate | tl | raw_136 | MP | 3c | B3 | 0.8362 | 0.8547 | 0.8546 |
| intermediate | tl | raw_136 | MP | 7c | B3 | 0.7377 | 0.8185 | 0.8188 |
| intermediate | tl | facs_28 | MP | 3c | B1 | 0.8295 | 0.8508 | 0.8508 |
| intermediate | tl | facs_28 | MP | 7c | B1 | 0.7188 | 0.8051 | 0.8042 |
| intermediate | tl | facs_28 | MP | 3c | B2 | 0.8218 | 0.8415 | 0.8420 |
| intermediate | tl | facs_28 | MP | 7c | B2 | 0.7263 | 0.8171 | 0.8202 |
| intermediate | tl | facs_28 | MP | 3c | B3 | 0.8192 | 0.8403 | 0.8403 |
| intermediate | tl | facs_28 | MP | 7c | B3 | 0.7234 | 0.8054 | 0.8052 |
| intermediate | tl | blendshape_52 | MP | 3c | B1 | 0.8151 | 0.8380 | 0.8389 |
| intermediate | tl | blendshape_52 | MP | 7c | B1 | 0.7200 | 0.8072 | 0.8103 |
| intermediate | tl | blendshape_52 | MP | 3c | B2 | 0.8179 | 0.8393 | 0.8406 |
| intermediate | tl | blendshape_52 | MP | 7c | B2 | 0.7270 | 0.8124 | 0.8157 |
| intermediate | tl | blendshape_52 | MP | 3c | B3 | 0.8212 | 0.8423 | 0.8423 |
| intermediate | tl | blendshape_52 | MP | 7c | B3 | 0.7168 | 0.8006 | 0.7987 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B1 | 0.8131 | 0.8338 | 0.8324 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B1 | 0.7264 | 0.8169 | 0.8202 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B2 | 0.8187 | 0.8392 | 0.8399 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B2 | 0.7264 | 0.8167 | 0.8178 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B3 | 0.8243 | 0.8432 | 0.8430 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B3 | 0.7309 | 0.8106 | 0.8120 |

### 4.5 Late Fusion (semua feature × variant, MP source)

| Fusion | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| late | scratch | raw_136 | MP | 3c | B1 | 0.7893 | 0.8131 | 0.8113 |
| late | scratch | raw_136 | MP | 7c | B1 | 0.6976 | 0.7939 | 0.7990 |
| late | scratch | raw_136 | MP | 3c | B2 | 0.7967 | 0.8202 | 0.8195 |
| late | scratch | raw_136 | MP | 7c | B2 | 0.6968 | 0.7865 | 0.7892 |
| late | scratch | raw_136 | MP | 3c | B3 | 0.8340 | 0.8534 | 0.8525 |
| late | scratch | raw_136 | MP | 7c | B3 | 0.7217 | 0.8107 | 0.8072 |
| late | scratch | facs_28 | MP | 3c | B1 | 0.7914 | 0.8155 | 0.8137 |
| late | scratch | facs_28 | MP | 7c | B1 | 0.6953 | 0.7913 | 0.7963 |
| late | scratch | facs_28 | MP | 3c | B2 | 0.7956 | 0.8203 | 0.8202 |
| late | scratch | facs_28 | MP | 7c | B2 | 0.6918 | 0.7810 | 0.7834 |
| late | scratch | facs_28 | MP | 3c | B3 | 0.8340 | 0.8532 | 0.8525 |
| late | scratch | facs_28 | MP | 7c | B3 | 0.7221 | 0.8081 | 0.8042 |
| late | scratch | blendshape_52 | MP | 3c | B1 | 0.7967 | 0.8198 | 0.8188 |
| late | scratch | blendshape_52 | MP | 7c | B1 | 0.6982 | 0.7944 | 0.7997 |
| late | scratch | blendshape_52 | MP | 3c | B2 | 0.7967 | 0.8215 | 0.8212 |
| late | scratch | blendshape_52 | MP | 7c | B2 | 0.6935 | 0.7825 | 0.7844 |
| late | scratch | blendshape_52 | MP | 3c | B3 | 0.8297 | 0.8487 | 0.8481 |
| late | scratch | blendshape_52 | MP | 7c | B3 | 0.7274 | 0.8124 | 0.8089 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B1 | 0.7945 | 0.8182 | 0.8164 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B1 | 0.6955 | 0.7938 | 0.8007 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B2 | 0.8020 | 0.8243 | 0.8243 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B2 | 0.6964 | 0.7876 | 0.7895 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B3 | 0.8358 | 0.8552 | 0.8546 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B3 | 0.7221 | 0.8081 | 0.8042 |
| late | tl | raw_136 | MP | 3c | B1 | 0.8094 | 0.8315 | 0.8317 |
| late | tl | raw_136 | MP | 7c | B1 | 0.7460 | 0.8303 | 0.8314 |
| late | tl | raw_136 | MP | 3c | B2 | 0.8190 | 0.8388 | 0.8379 |
| late | tl | raw_136 | MP | 7c | B2 | 0.7302 | 0.8142 | 0.8171 |
| late | tl | raw_136 | MP | 3c | B3 | 0.8400 | 0.8587 | 0.8573 |
| late | tl | raw_136 | MP | 7c | B3 | 0.7520 | 0.8269 | 0.8273 |
| late | tl | facs_28 | MP | 3c | B1 | 0.8063 | 0.8287 | 0.8290 |
| late | tl | facs_28 | MP | 7c | B1 | 0.7399 | 0.8261 | 0.8266 |
| late | tl | facs_28 | MP | 3c | B2 | 0.8164 | 0.8360 | 0.8351 |
| late | tl | facs_28 | MP | 7c | B2 | 0.7336 | 0.8154 | 0.8178 |
| late | tl | facs_28 | MP | 3c | B3 | 0.8417 | 0.8596 | 0.8583 |
| late | tl | facs_28 | MP | 7c | B3 | 0.7478 | 0.8259 | 0.8263 |
| late | tl | blendshape_52 | MP | 3c | B1 | 0.8057 | 0.8283 | 0.8287 |
| late | tl | blendshape_52 | MP | 7c | B1 | 0.7415 | 0.8270 | 0.8277 |
| late | tl | blendshape_52 | MP | 3c | B2 | 0.8191 | 0.8387 | 0.8379 |
| late | tl | blendshape_52 | MP | 7c | B2 | 0.7376 | 0.8153 | 0.8174 |
| late | tl | blendshape_52 | MP | 3c | B3 | 0.8416 | 0.8590 | 0.8580 |
| late | tl | blendshape_52 | MP | 7c | B3 | 0.7429 | 0.8257 | 0.8260 |
| late | tl | facs_plus_bs_80 | MP | 3c | B1 | 0.8084 | 0.8303 | 0.8307 |
| late | tl | facs_plus_bs_80 | MP | 7c | B1 | 0.7344 | 0.8212 | 0.8239 |
| late | tl | facs_plus_bs_80 | MP | 3c | B2 | 0.8204 | 0.8395 | 0.8389 |
| late | tl | facs_plus_bs_80 | MP | 7c | B2 | 0.7353 | 0.8167 | 0.8188 |
| late | tl | facs_plus_bs_80 | MP | 3c | B3 | 0.8388 | 0.8571 | 0.8559 |
| late | tl | facs_plus_bs_80 | MP | 7c | B3 | 0.7479 | 0.8263 | 0.8266 |

---

## 5. CK+ 7c (Cross-dataset Benchmark)

Source data: `models/benchmark/ckplus_7class/{3,7}class/Unified/`.

### 5.1 Landmark (MP only)

| Feature | Source | Arch | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| raw_136 | MP | FCNN | 3c | B1 | 0.6840 | 0.6513 | 0.6441 |
| raw_136 | MP | FCNN | 7c | B1 | 0.4074 | 0.6078 | 0.6949 |
| raw_136 | MP | FCNN | 3c | B2 | 0.4744 | 0.4403 | 0.4746 |
| raw_136 | MP | FCNN | 7c | B2 | 0.3693 | 0.4120 | 0.4068 |
| raw_136 | MP | FCNN | 3c | B3 | 0.5506 | 0.5966 | 0.6610 |
| raw_136 | MP | FCNN | 7c | B3 | 0.4652 | 0.3588 | 0.3898 |
| raw_136 | MP | CNN1D | 3c | B1 | 0.6375 | 0.6483 | 0.6441 |
| raw_136 | MP | CNN1D | 7c | B1 | 0.4594 | 0.6241 | 0.6610 |
| raw_136 | MP | CNN1D | 3c | B2 | 0.4817 | 0.5143 | 0.5593 |
| raw_136 | MP | CNN1D | 7c | B2 | 0.3638 | 0.3370 | 0.3220 |
| raw_136 | MP | CNN1D | 3c | B3 | 0.5677 | 0.6004 | 0.6271 |
| raw_136 | MP | CNN1D | 7c | B3 | 0.3611 | 0.2466 | 0.2881 |
| facs_28 | MP | FCNN | 3c | B1 | 0.9109 | 0.9126 | 0.9153 |
| facs_28 | MP | FCNN | 7c | B1 | 0.6033 | 0.7935 | 0.8136 |
| facs_28 | MP | FCNN | 3c | B2 | 0.8188 | 0.8004 | 0.7966 |
| facs_28 | MP | FCNN | 7c | B2 | 0.6359 | 0.6878 | 0.6610 |
| facs_28 | MP | FCNN | 3c | B3 | 0.7869 | 0.7833 | 0.7797 |
| facs_28 | MP | FCNN | 7c | B3 | 0.6179 | 0.7531 | 0.7458 |
| facs_28 | MP | CNN1D | 3c | B1 | 0.6453 | 0.6727 | 0.6780 |
| facs_28 | MP | CNN1D | 7c | B1 | 0.6360 | 0.8065 | 0.8305 |
| facs_28 | MP | CNN1D | 3c | B2 | 0.8250 | 0.8266 | 0.8305 |
| facs_28 | MP | CNN1D | 7c | B2 | 0.5780 | 0.7148 | 0.7119 |
| facs_28 | MP | CNN1D | 3c | B3 | 0.6603 | 0.6880 | 0.6949 |
| facs_28 | MP | CNN1D | 7c | B3 | 0.6523 | 0.7551 | 0.7458 |
| blendshape_52 | MP | FCNN | 3c | B1 | 0.8605 | 0.8641 | 0.8644 |
| blendshape_52 | MP | FCNN | 7c | B1 | 0.6839 | 0.8515 | 0.8644 |
| blendshape_52 | MP | FCNN | 3c | B2 | 0.8337 | 0.8320 | 0.8305 |
| blendshape_52 | MP | FCNN | 7c | B2 | 0.6203 | 0.6980 | 0.6610 |
| blendshape_52 | MP | FCNN | 3c | B3 | 0.8493 | 0.8625 | 0.8644 |
| blendshape_52 | MP | FCNN | 7c | B3 | 0.6761 | 0.6948 | 0.6780 |
| blendshape_52 | MP | CNN1D | 3c | B1 | 0.8391 | 0.8549 | 0.8644 |
| blendshape_52 | MP | CNN1D | 7c | B1 | 0.5881 | 0.7736 | 0.8136 |
| blendshape_52 | MP | CNN1D | 3c | B2 | 0.8044 | 0.8115 | 0.8136 |
| blendshape_52 | MP | CNN1D | 7c | B2 | 0.6177 | 0.7863 | 0.7797 |
| blendshape_52 | MP | CNN1D | 3c | B3 | 0.7834 | 0.7826 | 0.7797 |
| blendshape_52 | MP | CNN1D | 7c | B3 | 0.5612 | 0.6838 | 0.6610 |
| facs_plus_bs_80 | MP | FCNN | 3c | B1 | 0.8921 | 0.9115 | 0.9153 |
| facs_plus_bs_80 | MP | FCNN | 7c | B1 | 0.7323 | 0.8730 | 0.8814 |
| facs_plus_bs_80 | MP | FCNN | 3c | B2 | 0.8637 | 0.8661 | 0.8644 |
| facs_plus_bs_80 | MP | FCNN | 7c | B2 | 0.5822 | 0.7451 | 0.7288 |
| facs_plus_bs_80 | MP | FCNN | 3c | B3 | 0.8605 | 0.8798 | 0.8814 |
| facs_plus_bs_80 | MP | FCNN | 7c | B3 | 0.6120 | 0.7628 | 0.7458 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B1 | 0.7963 | 0.7986 | 0.7966 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B1 | 0.6559 | 0.8081 | 0.8136 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B2 | 0.7449 | 0.7335 | 0.7288 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B2 | 0.6006 | 0.7551 | 0.7458 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B3 | 0.7705 | 0.7664 | 0.7627 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B3 | 0.5777 | 0.6822 | 0.6610 |

### 5.2 Image

| Arch | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| CNN_SCRATCH | 3c | B1 | 0.3007 | 0.3942 | 0.5085 |
| CNN_SCRATCH | 7c | B1 | 0.1168 | 0.3560 | 0.4746 |
| CNN_SCRATCH | 3c | B2 | 0.8014 | 0.7981 | 0.7966 |
| CNN_SCRATCH | 7c | B2 | 0.3335 | 0.3302 | 0.3390 |
| CNN_SCRATCH | 3c | B3 | 0.7305 | 0.7667 | 0.7797 |
| CNN_SCRATCH | 7c | B3 | 0.6070 | 0.6530 | 0.6271 |
| CNN_TL | 3c | B1 | 0.9184 | 0.9179 | 0.9153 |
| CNN_TL | 7c | B1 | 0.7906 | 0.8795 | 0.8814 |
| CNN_TL | 3c | B2 | 0.8587 | 0.8752 | 0.8814 |
| CNN_TL | 7c | B2 | 0.8390 | 0.8690 | 0.8644 |
| CNN_TL | 3c | B3 | 0.9175 | 0.9167 | 0.9153 |
| CNN_TL | 7c | B3 | 0.8474 | 0.9021 | 0.8983 |

### 5.3 Early Fusion (raw_136, MP source) — concat vs gated

| Fusion | Mode | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| early | concat | scratch | raw_136 | MP | 3c | B1 | 0.7538 | 0.7657 | 0.7797 |
| early | concat | scratch | raw_136 | MP | 7c | B1 | 0.5058 | 0.6932 | 0.7288 |
| early | concat | scratch | raw_136 | MP | 3c | B2 | 0.7943 | 0.7856 | 0.7797 |
| early | concat | scratch | raw_136 | MP | 7c | B2 | 0.5555 | 0.6903 | 0.6949 |
| early | concat | scratch | raw_136 | MP | 3c | B3 | 0.7606 | 0.7629 | 0.7627 |
| early | concat | scratch | raw_136 | MP | 7c | B3 | 0.5716 | 0.5988 | 0.5763 |
| early | concat | tl | raw_136 | MP | 3c | B1 | 0.8610 | 0.8503 | 0.8475 |
| early | concat | tl | raw_136 | MP | 7c | B1 | 0.6545 | 0.7006 | 0.6780 |
| early | concat | tl | raw_136 | MP | 3c | B2 | 0.9017 | 0.8993 | 0.8983 |
| early | concat | tl | raw_136 | MP | 7c | B2 | 0.7446 | 0.8462 | 0.8644 |
| early | concat | tl | raw_136 | MP | 3c | B3 | 0.9407 | 0.9330 | 0.9322 |
| early | concat | tl | raw_136 | MP | 7c | B3 | 0.8850 | 0.9296 | 0.9322 |
| early | gated | scratch | raw_136 | MP | 3c | B1 | 0.8023 | 0.7921 | 0.7966 |
| early | gated | scratch | raw_136 | MP | 7c | B1 | 0.5481 | 0.6672 | 0.6610 |
| early | gated | scratch | raw_136 | MP | 3c | B2 | 0.7444 | 0.7587 | 0.7797 |
| early | gated | scratch | raw_136 | MP | 7c | B2 | 0.4233 | 0.4433 | 0.3898 |
| early | gated | scratch | raw_136 | MP | 3c | B3 | 0.7868 | 0.7871 | 0.7966 |
| early | gated | scratch | raw_136 | MP | 7c | B3 | 0.5473 | 0.6708 | 0.6441 |
| early | gated | tl | raw_136 | MP | 3c | B1 | 0.9315 | 0.9332 | 0.9322 |
| early | gated | tl | raw_136 | MP | 7c | B1 | 0.7446 | 0.8264 | 0.8305 |
| early | gated | tl | raw_136 | MP | 3c | B2 | 0.8466 | 0.8162 | 0.8136 |
| early | gated | tl | raw_136 | MP | 7c | B2 | 0.8571 | 0.9154 | 0.9153 |
| early | gated | tl | raw_136 | MP | 3c | B3 | 0.8398 | 0.8492 | 0.8475 |
| early | gated | tl | raw_136 | MP | 7c | B3 | 0.8510 | 0.9346 | 0.9322 |

### 5.4 Intermediate Fusion (semua feature × variant, MP source)

| Fusion | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| intermediate | scratch | raw_136 | MP | 3c | B1 | 0.4946 | 0.5416 | 0.6102 |
| intermediate | scratch | raw_136 | MP | 7c | B1 | 0.3826 | 0.5997 | 0.6441 |
| intermediate | scratch | raw_136 | MP | 3c | B2 | 0.6040 | 0.6285 | 0.6271 |
| intermediate | scratch | raw_136 | MP | 7c | B2 | 0.3658 | 0.5043 | 0.5085 |
| intermediate | scratch | raw_136 | MP | 3c | B3 | 0.7068 | 0.6696 | 0.6610 |
| intermediate | scratch | raw_136 | MP | 7c | B3 | 0.3662 | 0.3972 | 0.4068 |
| intermediate | scratch | facs_28 | MP | 3c | B1 | 0.8932 | 0.8938 | 0.8983 |
| intermediate | scratch | facs_28 | MP | 7c | B1 | 0.6360 | 0.8007 | 0.8136 |
| intermediate | scratch | facs_28 | MP | 3c | B2 | 0.8289 | 0.8156 | 0.8136 |
| intermediate | scratch | facs_28 | MP | 7c | B2 | 0.7140 | 0.7983 | 0.7797 |
| intermediate | scratch | facs_28 | MP | 3c | B3 | 0.9026 | 0.9130 | 0.9153 |
| intermediate | scratch | facs_28 | MP | 7c | B3 | 0.7137 | 0.7701 | 0.7288 |
| intermediate | scratch | blendshape_52 | MP | 3c | B1 | 0.9026 | 0.9130 | 0.9153 |
| intermediate | scratch | blendshape_52 | MP | 7c | B1 | 0.6913 | 0.8321 | 0.8305 |
| intermediate | scratch | blendshape_52 | MP | 3c | B2 | 0.8369 | 0.8336 | 0.8305 |
| intermediate | scratch | blendshape_52 | MP | 7c | B2 | 0.6460 | 0.7070 | 0.6780 |
| intermediate | scratch | blendshape_52 | MP | 3c | B3 | 0.8705 | 0.8777 | 0.8814 |
| intermediate | scratch | blendshape_52 | MP | 7c | B3 | 0.6568 | 0.7862 | 0.7797 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B1 | 0.8882 | 0.8965 | 0.8983 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B1 | 0.8646 | 0.9313 | 0.9322 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B2 | 0.9053 | 0.9150 | 0.9153 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B2 | 0.7251 | 0.8223 | 0.8136 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B3 | 0.9026 | 0.9130 | 0.9153 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B3 | 0.6476 | 0.7754 | 0.7458 |
| intermediate | tl | raw_136 | MP | 3c | B1 | 0.8601 | 0.8335 | 0.8305 |
| intermediate | tl | raw_136 | MP | 7c | B1 | 0.7538 | 0.8157 | 0.7797 |
| intermediate | tl | raw_136 | MP | 3c | B2 | 0.7901 | 0.7669 | 0.7627 |
| intermediate | tl | raw_136 | MP | 7c | B2 | 0.7231 | 0.8165 | 0.8136 |
| intermediate | tl | raw_136 | MP | 3c | B3 | 0.9251 | 0.9158 | 0.9153 |
| intermediate | tl | raw_136 | MP | 7c | B3 | 0.8582 | 0.9015 | 0.8983 |
| intermediate | tl | facs_28 | MP | 3c | B1 | 0.9560 | 0.9499 | 0.9492 |
| intermediate | tl | facs_28 | MP | 7c | B1 | 0.8586 | 0.9075 | 0.8983 |
| intermediate | tl | facs_28 | MP | 3c | B2 | 0.9174 | 0.9296 | 0.9322 |
| intermediate | tl | facs_28 | MP | 7c | B2 | 0.8568 | 0.9007 | 0.8983 |
| intermediate | tl | facs_28 | MP | 3c | B3 | 0.9219 | 0.9331 | 0.9322 |
| intermediate | tl | facs_28 | MP | 7c | B3 | 0.7693 | 0.8979 | 0.9153 |
| intermediate | tl | blendshape_52 | MP | 3c | B1 | 0.9077 | 0.9166 | 0.9153 |
| intermediate | tl | blendshape_52 | MP | 7c | B1 | 0.8167 | 0.8744 | 0.8644 |
| intermediate | tl | blendshape_52 | MP | 3c | B2 | 0.9315 | 0.9332 | 0.9322 |
| intermediate | tl | blendshape_52 | MP | 7c | B2 | 0.8410 | 0.9071 | 0.9153 |
| intermediate | tl | blendshape_52 | MP | 3c | B3 | 0.9219 | 0.9331 | 0.9322 |
| intermediate | tl | blendshape_52 | MP | 7c | B3 | 0.7622 | 0.8270 | 0.8305 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B1 | 0.9550 | 0.9495 | 0.9492 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B1 | 0.7983 | 0.8662 | 0.8475 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B2 | 0.9135 | 0.9143 | 0.9153 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B2 | 0.8598 | 0.9218 | 0.9153 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B3 | 0.9036 | 0.9003 | 0.8983 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B3 | 0.7453 | 0.8815 | 0.8983 |

### 5.5 Late Fusion (semua feature × variant, MP source)

| Fusion | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| late | scratch | raw_136 | MP | 3c | B1 | 0.7778 | 0.7966 | 0.7966 |
| late | scratch | raw_136 | MP | 7c | B1 | 0.4418 | 0.6533 | 0.7288 |
| late | scratch | raw_136 | MP | 3c | B2 | 0.5934 | 0.5927 | 0.5932 |
| late | scratch | raw_136 | MP | 7c | B2 | 0.4861 | 0.5801 | 0.5424 |
| late | scratch | raw_136 | MP | 3c | B3 | 0.7473 | 0.7545 | 0.7627 |
| late | scratch | raw_136 | MP | 7c | B3 | 0.4254 | 0.3182 | 0.3898 |
| late | scratch | facs_28 | MP | 3c | B1 | 0.8605 | 0.8641 | 0.8644 |
| late | scratch | facs_28 | MP | 7c | B1 | 0.6394 | 0.7999 | 0.8305 |
| late | scratch | facs_28 | MP | 3c | B2 | 0.8188 | 0.8004 | 0.7966 |
| late | scratch | facs_28 | MP | 7c | B2 | 0.6886 | 0.7258 | 0.6949 |
| late | scratch | facs_28 | MP | 3c | B3 | 0.8320 | 0.8168 | 0.8136 |
| late | scratch | facs_28 | MP | 7c | B3 | 0.6797 | 0.6811 | 0.6610 |
| late | scratch | blendshape_52 | MP | 3c | B1 | 0.8912 | 0.8986 | 0.8983 |
| late | scratch | blendshape_52 | MP | 7c | B1 | 0.6620 | 0.8321 | 0.8475 |
| late | scratch | blendshape_52 | MP | 3c | B2 | 0.8337 | 0.8320 | 0.8305 |
| late | scratch | blendshape_52 | MP | 7c | B2 | 0.6358 | 0.7288 | 0.6949 |
| late | scratch | blendshape_52 | MP | 3c | B3 | 0.8337 | 0.8320 | 0.8305 |
| late | scratch | blendshape_52 | MP | 7c | B3 | 0.6025 | 0.6113 | 0.5932 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B1 | 0.8921 | 0.9115 | 0.9153 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B1 | 0.7323 | 0.8730 | 0.8814 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B2 | 0.8637 | 0.8661 | 0.8644 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B2 | 0.5745 | 0.7158 | 0.7119 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B3 | 0.8702 | 0.8805 | 0.8814 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B3 | 0.6746 | 0.7074 | 0.6949 |
| late | tl | raw_136 | MP | 3c | B1 | 0.7679 | 0.7694 | 0.7627 |
| late | tl | raw_136 | MP | 7c | B1 | 0.7325 | 0.8509 | 0.8475 |
| late | tl | raw_136 | MP | 3c | B2 | 0.9754 | 0.9832 | 0.9831 |
| late | tl | raw_136 | MP | 7c | B2 | 0.7020 | 0.8160 | 0.8136 |
| late | tl | raw_136 | MP | 3c | B3 | 0.8498 | 0.8657 | 0.8644 |
| late | tl | raw_136 | MP | 7c | B3 | 0.7994 | 0.8369 | 0.8305 |
| late | tl | facs_28 | MP | 3c | B1 | 0.8432 | 0.8512 | 0.8475 |
| late | tl | facs_28 | MP | 7c | B1 | 0.6937 | 0.8471 | 0.8644 |
| late | tl | facs_28 | MP | 3c | B2 | 0.9299 | 0.9322 | 0.9322 |
| late | tl | facs_28 | MP | 7c | B2 | 0.7365 | 0.7725 | 0.7458 |
| late | tl | facs_28 | MP | 3c | B3 | 0.9129 | 0.9002 | 0.8983 |
| late | tl | facs_28 | MP | 7c | B3 | 0.8331 | 0.8806 | 0.8814 |
| late | tl | blendshape_52 | MP | 3c | B1 | 0.8166 | 0.8185 | 0.8136 |
| late | tl | blendshape_52 | MP | 7c | B1 | 0.7117 | 0.8679 | 0.8814 |
| late | tl | blendshape_52 | MP | 3c | B2 | 0.9365 | 0.9497 | 0.9492 |
| late | tl | blendshape_52 | MP | 7c | B2 | 0.7198 | 0.7847 | 0.7627 |
| late | tl | blendshape_52 | MP | 3c | B3 | 0.8663 | 0.8675 | 0.8644 |
| late | tl | blendshape_52 | MP | 7c | B3 | 0.8012 | 0.8667 | 0.8644 |
| late | tl | facs_plus_bs_80 | MP | 3c | B1 | 0.9174 | 0.9296 | 0.9322 |
| late | tl | facs_plus_bs_80 | MP | 7c | B1 | 0.8008 | 0.9039 | 0.8983 |
| late | tl | facs_plus_bs_80 | MP | 3c | B2 | 0.9754 | 0.9832 | 0.9831 |
| late | tl | facs_plus_bs_80 | MP | 7c | B2 | 0.6508 | 0.7980 | 0.7966 |
| late | tl | facs_plus_bs_80 | MP | 3c | B3 | 0.9280 | 0.9170 | 0.9153 |
| late | tl | facs_plus_bs_80 | MP | 7c | B3 | 0.7297 | 0.8362 | 0.8305 |

---

## 6. JAFFE 7c (Cross-dataset Benchmark)

Source data: `models/benchmark/jaffe_7class/{3,7}class/Unified/`.

### 6.1 Landmark (MP only)

| Feature | Source | Arch | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| raw_136 | MP | FCNN | 3c | B1 | 0.1630 | 0.1344 | 0.2750 |
| raw_136 | MP | FCNN | 7c | B1 | 0.1608 | 0.1620 | 0.2750 |
| raw_136 | MP | FCNN | 3c | B2 | 0.3814 | 0.5454 | 0.5750 |
| raw_136 | MP | FCNN | 7c | B2 | 0.1081 | 0.1071 | 0.2000 |
| raw_136 | MP | FCNN | 3c | B3 | 0.6269 | 0.6839 | 0.6750 |
| raw_136 | MP | FCNN | 7c | B3 | 0.0948 | 0.0905 | 0.2000 |
| raw_136 | MP | CNN1D | 3c | B1 | 0.2295 | 0.3959 | 0.5250 |
| raw_136 | MP | CNN1D | 7c | B1 | 0.0985 | 0.0966 | 0.2000 |
| raw_136 | MP | CNN1D | 3c | B2 | 0.4275 | 0.5660 | 0.6000 |
| raw_136 | MP | CNN1D | 7c | B2 | 0.0764 | 0.0724 | 0.1750 |
| raw_136 | MP | CNN1D | 3c | B3 | 0.5308 | 0.5148 | 0.5250 |
| raw_136 | MP | CNN1D | 7c | B3 | 0.2232 | 0.2165 | 0.2750 |
| facs_28 | MP | FCNN | 3c | B1 | 0.2917 | 0.4571 | 0.5750 |
| facs_28 | MP | FCNN | 7c | B1 | 0.2440 | 0.2255 | 0.2750 |
| facs_28 | MP | FCNN | 3c | B2 | 0.2917 | 0.4571 | 0.5750 |
| facs_28 | MP | FCNN | 7c | B2 | 0.4116 | 0.3968 | 0.4000 |
| facs_28 | MP | FCNN | 3c | B3 | 0.4841 | 0.6351 | 0.7000 |
| facs_28 | MP | FCNN | 7c | B3 | 0.1079 | 0.0994 | 0.2000 |
| facs_28 | MP | CNN1D | 3c | B1 | 0.2330 | 0.2614 | 0.3500 |
| facs_28 | MP | CNN1D | 7c | B1 | 0.2763 | 0.2702 | 0.3500 |
| facs_28 | MP | CNN1D | 3c | B2 | 0.2128 | 0.2235 | 0.3250 |
| facs_28 | MP | CNN1D | 7c | B2 | 0.1143 | 0.1075 | 0.2000 |
| facs_28 | MP | CNN1D | 3c | B3 | 0.1838 | 0.1766 | 0.3000 |
| facs_28 | MP | CNN1D | 7c | B3 | 0.3481 | 0.3393 | 0.4000 |
| blendshape_52 | MP | FCNN | 3c | B1 | 0.3101 | 0.4849 | 0.5500 |
| blendshape_52 | MP | FCNN | 7c | B1 | 0.1418 | 0.1489 | 0.2250 |
| blendshape_52 | MP | FCNN | 3c | B2 | 0.4046 | 0.5221 | 0.5500 |
| blendshape_52 | MP | FCNN | 7c | B2 | 0.3110 | 0.3123 | 0.3500 |
| blendshape_52 | MP | FCNN | 3c | B3 | 0.6793 | 0.7270 | 0.7250 |
| blendshape_52 | MP | FCNN | 7c | B3 | 0.4304 | 0.4364 | 0.4500 |
| blendshape_52 | MP | CNN1D | 3c | B1 | 0.3029 | 0.4724 | 0.6000 |
| blendshape_52 | MP | CNN1D | 7c | B1 | 0.2491 | 0.2462 | 0.2750 |
| blendshape_52 | MP | CNN1D | 3c | B2 | 0.3539 | 0.5182 | 0.6250 |
| blendshape_52 | MP | CNN1D | 7c | B2 | 0.3029 | 0.2958 | 0.3000 |
| blendshape_52 | MP | CNN1D | 3c | B3 | 0.3539 | 0.5182 | 0.6250 |
| blendshape_52 | MP | CNN1D | 7c | B3 | 0.2989 | 0.2888 | 0.3000 |
| facs_plus_bs_80 | MP | FCNN | 3c | B1 | 0.2434 | 0.4198 | 0.5750 |
| facs_plus_bs_80 | MP | FCNN | 7c | B1 | 0.2544 | 0.2528 | 0.3000 |
| facs_plus_bs_80 | MP | FCNN | 3c | B2 | 0.3773 | 0.4914 | 0.5000 |
| facs_plus_bs_80 | MP | FCNN | 7c | B2 | 0.2912 | 0.2807 | 0.3250 |
| facs_plus_bs_80 | MP | FCNN | 3c | B3 | 0.3922 | 0.5120 | 0.5250 |
| facs_plus_bs_80 | MP | FCNN | 7c | B3 | 0.1081 | 0.1135 | 0.1750 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B1 | 0.4025 | 0.5531 | 0.6250 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B1 | 0.1561 | 0.1483 | 0.2250 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B2 | 0.3397 | 0.5002 | 0.6000 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B2 | 0.1145 | 0.1046 | 0.2000 |
| facs_plus_bs_80 | MP | CNN1D | 3c | B3 | 0.4370 | 0.5444 | 0.6000 |
| facs_plus_bs_80 | MP | CNN1D | 7c | B3 | 0.2416 | 0.2390 | 0.3000 |

### 6.2 Image

| Arch | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| CNN_SCRATCH | 3c | B1 | 0.4986 | 0.6602 | 0.7000 |
| CNN_SCRATCH | 7c | B1 | 0.3424 | 0.3359 | 0.3500 |
| CNN_SCRATCH | 3c | B2 | 0.3052 | 0.2898 | 0.3500 |
| CNN_SCRATCH | 7c | B2 | 0.4063 | 0.4195 | 0.4750 |
| CNN_SCRATCH | 3c | B3 | 0.2434 | 0.4198 | 0.5750 |
| CNN_SCRATCH | 7c | B3 | 0.3536 | 0.3642 | 0.3750 |
| CNN_TL | 3c | B1 | 0.4640 | 0.4217 | 0.4500 |
| CNN_TL | 7c | B1 | 0.2684 | 0.2718 | 0.3250 |
| CNN_TL | 3c | B2 | 0.4530 | 0.4604 | 0.4750 |
| CNN_TL | 7c | B2 | 0.4791 | 0.4732 | 0.4750 |
| CNN_TL | 3c | B3 | 0.5121 | 0.5170 | 0.4750 |
| CNN_TL | 7c | B3 | 0.2747 | 0.2813 | 0.4000 |

### 6.3 Early Fusion (raw_136, MP source) — concat vs gated

| Fusion | Mode | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| early | concat | scratch | raw_136 | MP | 3c | B1 | 0.4505 | 0.5986 | 0.6250 |
| early | concat | scratch | raw_136 | MP | 7c | B1 | 0.3631 | 0.3573 | 0.4000 |
| early | concat | scratch | raw_136 | MP | 3c | B2 | 0.2200 | 0.2295 | 0.3250 |
| early | concat | scratch | raw_136 | MP | 7c | B2 | 0.1239 | 0.1136 | 0.1750 |
| early | concat | scratch | raw_136 | MP | 3c | B3 | 0.3079 | 0.3701 | 0.4250 |
| early | concat | scratch | raw_136 | MP | 7c | B3 | 0.0807 | 0.0776 | 0.1750 |
| early | concat | tl | raw_136 | MP | 3c | B1 | 0.2785 | 0.3369 | 0.4000 |
| early | concat | tl | raw_136 | MP | 7c | B1 | 0.1266 | 0.1288 | 0.2250 |
| early | concat | tl | raw_136 | MP | 3c | B2 | 0.6430 | 0.6477 | 0.6500 |
| early | concat | tl | raw_136 | MP | 7c | B2 | 0.1204 | 0.1193 | 0.2000 |
| early | concat | tl | raw_136 | MP | 3c | B3 | 0.5417 | 0.5174 | 0.5250 |
| early | concat | tl | raw_136 | MP | 7c | B3 | 0.2430 | 0.2440 | 0.3250 |
| early | gated | scratch | raw_136 | MP | 3c | B1 | 0.3426 | 0.4375 | 0.4750 |
| early | gated | scratch | raw_136 | MP | 7c | B1 | 0.3044 | 0.3013 | 0.3750 |
| early | gated | scratch | raw_136 | MP | 3c | B2 | 0.4358 | 0.5862 | 0.6250 |
| early | gated | scratch | raw_136 | MP | 7c | B2 | 0.1796 | 0.1754 | 0.2750 |
| early | gated | scratch | raw_136 | MP | 3c | B3 | 0.4166 | 0.5494 | 0.5750 |
| early | gated | scratch | raw_136 | MP | 7c | B3 | 0.2767 | 0.2642 | 0.3750 |
| early | gated | tl | raw_136 | MP | 3c | B1 | 0.3614 | 0.4696 | 0.5000 |
| early | gated | tl | raw_136 | MP | 7c | B1 | 0.2347 | 0.2339 | 0.3000 |
| early | gated | tl | raw_136 | MP | 3c | B2 | 0.4377 | 0.5950 | 0.6750 |
| early | gated | tl | raw_136 | MP | 7c | B2 | 0.4226 | 0.4104 | 0.4500 |
| early | gated | tl | raw_136 | MP | 3c | B3 | 0.5841 | 0.6862 | 0.7250 |
| early | gated | tl | raw_136 | MP | 7c | B3 | 0.2315 | 0.2360 | 0.3000 |

### 6.4 Intermediate Fusion (semua feature × variant, MP source)

| Fusion | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| intermediate | scratch | raw_136 | MP | 3c | B1 | 0.5410 | 0.7263 | 0.7750 |
| intermediate | scratch | raw_136 | MP | 7c | B1 | 0.3012 | 0.3030 | 0.4000 |
| intermediate | scratch | raw_136 | MP | 3c | B2 | 0.6157 | 0.7184 | 0.7000 |
| intermediate | scratch | raw_136 | MP | 7c | B2 | 0.0816 | 0.0786 | 0.1750 |
| intermediate | scratch | raw_136 | MP | 3c | B3 | 0.3344 | 0.4159 | 0.4500 |
| intermediate | scratch | raw_136 | MP | 7c | B3 | 0.1002 | 0.1052 | 0.2250 |
| intermediate | scratch | facs_28 | MP | 3c | B1 | 0.3397 | 0.5002 | 0.6000 |
| intermediate | scratch | facs_28 | MP | 7c | B1 | 0.2574 | 0.2435 | 0.2750 |
| intermediate | scratch | facs_28 | MP | 3c | B2 | 0.3397 | 0.5002 | 0.6000 |
| intermediate | scratch | facs_28 | MP | 7c | B2 | 0.2701 | 0.2555 | 0.3500 |
| intermediate | scratch | facs_28 | MP | 3c | B3 | 0.4841 | 0.6351 | 0.7000 |
| intermediate | scratch | facs_28 | MP | 7c | B3 | 0.3561 | 0.3345 | 0.4000 |
| intermediate | scratch | blendshape_52 | MP | 3c | B1 | 0.2529 | 0.4362 | 0.5500 |
| intermediate | scratch | blendshape_52 | MP | 7c | B1 | 0.3379 | 0.3464 | 0.4000 |
| intermediate | scratch | blendshape_52 | MP | 3c | B2 | 0.4068 | 0.5318 | 0.5500 |
| intermediate | scratch | blendshape_52 | MP | 7c | B2 | 0.4312 | 0.4305 | 0.4500 |
| intermediate | scratch | blendshape_52 | MP | 3c | B3 | 0.4068 | 0.5318 | 0.5500 |
| intermediate | scratch | blendshape_52 | MP | 7c | B3 | 0.4261 | 0.4260 | 0.4500 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B1 | 0.5117 | 0.6644 | 0.7250 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B1 | 0.4489 | 0.4354 | 0.4500 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B2 | 0.3773 | 0.4914 | 0.5000 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B2 | 0.5287 | 0.5199 | 0.5000 |
| intermediate | scratch | facs_plus_bs_80 | MP | 3c | B3 | 0.5114 | 0.6752 | 0.7250 |
| intermediate | scratch | facs_plus_bs_80 | MP | 7c | B3 | 0.1315 | 0.1297 | 0.2250 |
| intermediate | tl | raw_136 | MP | 3c | B1 | 0.4057 | 0.4615 | 0.4500 |
| intermediate | tl | raw_136 | MP | 7c | B1 | 0.1766 | 0.1854 | 0.2500 |
| intermediate | tl | raw_136 | MP | 3c | B2 | 0.3483 | 0.2597 | 0.3750 |
| intermediate | tl | raw_136 | MP | 7c | B2 | 0.1982 | 0.1884 | 0.2500 |
| intermediate | tl | raw_136 | MP | 3c | B3 | 0.3798 | 0.3927 | 0.3750 |
| intermediate | tl | raw_136 | MP | 7c | B3 | 0.1426 | 0.1414 | 0.2500 |
| intermediate | tl | facs_28 | MP | 3c | B1 | 0.3403 | 0.2745 | 0.3750 |
| intermediate | tl | facs_28 | MP | 7c | B1 | 0.2038 | 0.2069 | 0.2750 |
| intermediate | tl | facs_28 | MP | 3c | B2 | 0.7734 | 0.7972 | 0.8000 |
| intermediate | tl | facs_28 | MP | 7c | B2 | 0.1526 | 0.1602 | 0.2500 |
| intermediate | tl | facs_28 | MP | 3c | B3 | 0.7109 | 0.7510 | 0.7500 |
| intermediate | tl | facs_28 | MP | 7c | B3 | 0.3061 | 0.3064 | 0.3500 |
| intermediate | tl | blendshape_52 | MP | 3c | B1 | 0.4259 | 0.4676 | 0.4500 |
| intermediate | tl | blendshape_52 | MP | 7c | B1 | 0.2737 | 0.2763 | 0.3250 |
| intermediate | tl | blendshape_52 | MP | 3c | B2 | 0.4841 | 0.4530 | 0.4750 |
| intermediate | tl | blendshape_52 | MP | 7c | B2 | 0.2236 | 0.2277 | 0.3000 |
| intermediate | tl | blendshape_52 | MP | 3c | B3 | 0.5541 | 0.6132 | 0.5500 |
| intermediate | tl | blendshape_52 | MP | 7c | B3 | 0.5255 | 0.5304 | 0.5500 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B1 | 0.1550 | 0.1279 | 0.2500 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B1 | 0.2437 | 0.2487 | 0.3250 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B2 | 0.3857 | 0.3879 | 0.4000 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B2 | 0.1197 | 0.1186 | 0.2000 |
| intermediate | tl | facs_plus_bs_80 | MP | 3c | B3 | 0.6168 | 0.6352 | 0.6000 |
| intermediate | tl | facs_plus_bs_80 | MP | 7c | B3 | 0.0914 | 0.0960 | 0.2000 |

### 6.5 Late Fusion (semua feature × variant, MP source)

| Fusion | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| late | scratch | raw_136 | MP | 3c | B1 | 0.3598 | 0.4557 | 0.5000 |
| late | scratch | raw_136 | MP | 7c | B1 | 0.3694 | 0.3682 | 0.4500 |
| late | scratch | raw_136 | MP | 3c | B2 | 0.5317 | 0.7011 | 0.7500 |
| late | scratch | raw_136 | MP | 7c | B2 | 0.3267 | 0.3377 | 0.3750 |
| late | scratch | raw_136 | MP | 3c | B3 | 0.6269 | 0.6839 | 0.6750 |
| late | scratch | raw_136 | MP | 7c | B3 | 0.1944 | 0.1979 | 0.2750 |
| late | scratch | facs_28 | MP | 3c | B1 | 0.2917 | 0.4571 | 0.5750 |
| late | scratch | facs_28 | MP | 7c | B1 | 0.2440 | 0.2255 | 0.2750 |
| late | scratch | facs_28 | MP | 3c | B2 | 0.2917 | 0.4571 | 0.5750 |
| late | scratch | facs_28 | MP | 7c | B2 | 0.4116 | 0.3968 | 0.4000 |
| late | scratch | facs_28 | MP | 3c | B3 | 0.2917 | 0.4571 | 0.5750 |
| late | scratch | facs_28 | MP | 7c | B3 | 0.4116 | 0.3968 | 0.4000 |
| late | scratch | blendshape_52 | MP | 3c | B1 | 0.4464 | 0.5826 | 0.6250 |
| late | scratch | blendshape_52 | MP | 7c | B1 | 0.1899 | 0.1994 | 0.2500 |
| late | scratch | blendshape_52 | MP | 3c | B2 | 0.6848 | 0.7905 | 0.7750 |
| late | scratch | blendshape_52 | MP | 7c | B2 | 0.4033 | 0.4135 | 0.4750 |
| late | scratch | blendshape_52 | MP | 3c | B3 | 0.4026 | 0.3634 | 0.4000 |
| late | scratch | blendshape_52 | MP | 7c | B3 | 0.3110 | 0.3123 | 0.3500 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B1 | 0.4424 | 0.5832 | 0.6250 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B1 | 0.2356 | 0.2331 | 0.2750 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B2 | 0.6482 | 0.7482 | 0.7250 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B2 | 0.2912 | 0.2807 | 0.3250 |
| late | scratch | facs_plus_bs_80 | MP | 3c | B3 | 0.4026 | 0.3634 | 0.4000 |
| late | scratch | facs_plus_bs_80 | MP | 7c | B3 | 0.2912 | 0.2807 | 0.3250 |
| late | tl | raw_136 | MP | 3c | B1 | 0.4052 | 0.3604 | 0.4000 |
| late | tl | raw_136 | MP | 7c | B1 | 0.3722 | 0.3647 | 0.4000 |
| late | tl | raw_136 | MP | 3c | B2 | 0.7303 | 0.8058 | 0.8000 |
| late | tl | raw_136 | MP | 7c | B2 | 0.3656 | 0.3548 | 0.4250 |
| late | tl | raw_136 | MP | 3c | B3 | 0.5835 | 0.6077 | 0.5750 |
| late | tl | raw_136 | MP | 7c | B3 | 0.2570 | 0.2627 | 0.3500 |
| late | tl | facs_28 | MP | 3c | B1 | 0.2917 | 0.4571 | 0.5750 |
| late | tl | facs_28 | MP | 7c | B1 | 0.2440 | 0.2255 | 0.2750 |
| late | tl | facs_28 | MP | 3c | B2 | 0.2917 | 0.4571 | 0.5750 |
| late | tl | facs_28 | MP | 7c | B2 | 0.4116 | 0.3968 | 0.4000 |
| late | tl | facs_28 | MP | 3c | B3 | 0.2917 | 0.4571 | 0.5750 |
| late | tl | facs_28 | MP | 7c | B3 | 0.4116 | 0.3968 | 0.4000 |
| late | tl | blendshape_52 | MP | 3c | B1 | 0.6650 | 0.6971 | 0.6750 |
| late | tl | blendshape_52 | MP | 7c | B1 | 0.3181 | 0.3269 | 0.3500 |
| late | tl | blendshape_52 | MP | 3c | B2 | 0.7522 | 0.7863 | 0.7750 |
| late | tl | blendshape_52 | MP | 7c | B2 | 0.3110 | 0.3123 | 0.3500 |
| late | tl | blendshape_52 | MP | 3c | B3 | 0.6051 | 0.6877 | 0.6500 |
| late | tl | blendshape_52 | MP | 7c | B3 | 0.2915 | 0.2989 | 0.4000 |
| late | tl | facs_plus_bs_80 | MP | 3c | B1 | 0.7185 | 0.7758 | 0.7500 |
| late | tl | facs_plus_bs_80 | MP | 7c | B1 | 0.4584 | 0.4605 | 0.4750 |
| late | tl | facs_plus_bs_80 | MP | 3c | B2 | 0.7107 | 0.7644 | 0.7500 |
| late | tl | facs_plus_bs_80 | MP | 7c | B2 | 0.2912 | 0.2807 | 0.3250 |
| late | tl | facs_plus_bs_80 | MP | 3c | B3 | 0.5397 | 0.6010 | 0.5500 |
| late | tl | facs_plus_bs_80 | MP | 7c | B3 | 0.3311 | 0.3322 | 0.4000 |

---

## 7. Summary: Runs DONE vs Expected

| Dataset | Landmark expected | Landmark done | Image expected | Image done | Fusion expected | Fusion done |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Primer | 84 | 84 | 12 | 12 | 216 | 216 |
| KDEF 7c | 48 | 48 | 12 | 12 | 120 | 120 |
| RAF-DB 7c | 48 | 48 | 12 | 12 | 120 | 120 |
| CK+ 7c | 48 | 48 | 12 | 12 | 120 | 120 |
| JAFFE 7c | 48 | 48 | 12 | 12 | 120 | 120 |

> Catatan: Primer expected includes MP+FA sources. Sekunder hanya MP (FA tidak tersedia karena image sudah pre-cropped, butuh JS pipeline).

---

*Regenerate dengan: `python scripts/build_results_tables.py`*