# Lomy
Binární klasifikace lomů (štěpný x tvárný)

| Model       | Accuracy    |
| ----------- | ----------- |
| SVM (SVC)   | 79.8%       |
| CNN         | 85%         |
| KNN hist    | 82%         |
| KNN raw     | 74%         |

### Instalace:
1. Repozitář obsahuje velké soubory spravované `git lfs`

	```
	git lfs install
	```
2. Nainstalovat requirements (doporučeno do venv nebo Conda enviromentu)

	```
	pip install -r requirements.txt
	```

## KDO CO DĚLAL

Andrii - Appka

Dan, Honza - Model

Aleš, Vašek - Data

## Link na dataset

Link: https://drive.google.com/file/d/1k1I_AF1FbsyWtccinHeDZ2C5mzqOiktP/view?usp=sharing

## TODO:

- [x] Dataset preprocessing
- [x] Model
- [x] Data augmentation (moc to nezlepšovalo)
- [x] Early stopping? (potřebuje potunit)
- [x] Grafy
- [x] App na PC
- [x] Inference na apku
- [x] Ensemble CNN (blbne)
