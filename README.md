# Lomy
Binární klasifikace lomů (štěpný x tvárný)

| Model       | Accuracy    | Optimizer   |
| ----------- | ----------- | ----------- |
| SVM         | 79.8%       |             |
| CNN         | 85%         | Adam        |
| KNN hist    | 82%         |             |
| KNN raw     | 74%         |             |
| Ensemble CNN* | 25%       | SGD         | 

*nedodělaný model, koncept

***Použité balíčky:***
    **Aplikace:**
        - `tkinter`
        - `customtkinter`
    **Modely (Neuronové sítě a algoritmy strojového učení):**
        - `torch`
        - `python-opencv`
        - `numpy`
    **Data preprocessing:**
        - `numpy`
        - `python-opencv`
    **Webová Aplikace:**
        - `flask`

## Link na dataset

Link: https://drive.google.com/file/d/1k1I_AF1FbsyWtccinHeDZ2C5mzqOiktP/view?usp=sharing

## TODO:

- [x] choose theme
- [x] dataset preprocessing
- [x] dataset processing fix
- [x] model
- [x] Data augmentation
- [x] Early stopping?
- [x] Grafy
- [x] App na PC
- [x] Inference na apku
- [ ] Ensemble CNN
- [ ] Webová aplikace