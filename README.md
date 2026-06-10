# CASCADE — Detección automática de lesiones del LCA en RM de rodilla

**Trabajo Fin de Grado** · Grado en Ciencia e Ingeniería de Datos · Universidade da Coruña (UDC)

**Autor:** Jacobo Cousillas Taboada  
**Tutores:** Laura Morán Fernández · Daniel Vila de la Cruz

---

## Descripción

CASCADE (*Clinical Assessment System for Computer Aided Detection and Evaluation*) es un sistema de detección automática de lesiones del ligamento cruzado anterior (LCA) en resonancia magnética de rodilla basado en aprendizaje profundo.

El sistema procesa los tres planos estándar de adquisición (sagital, coronal y axial) mediante seis redes ResNet-18 con *attention pooling*, organizadas en dos fases secuenciales:

- **Fase 1 (Triaje):** detecta la presencia de anomalías generales en la rodilla.
- **Fase 2 (Clasificación):** identifica lesiones del LCA y del menisco de forma simultánea, combinando la información de los tres planos mediante fusión ponderada.

El preprocesamiento incluye CLAHE para la mejora del contraste de tejidos blandos y una máscara gaussiana anatómica en el plano sagital. La interpretabilidad se proporciona mediante mapas GradCAM++ sobre el corte de máxima atención de cada plano y reportes clínicos estructurados.

## Resultados principales

Evaluación mediante validación cruzada estratificada de 5 folds sobre el conjunto de datos [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/) (1 250 casos, Stanford University Medical Center):

| Métrica | Media 5-fold | IC 95 % |
|---|---|---|
| AUC | 0,838 | [0,811; 0,865] |
| Sensibilidad | 84,7 % | [77,4 %; 92,0 %] |
| Especificidad | 64,0 % | [52,1 %; 75,8 %] |
| F1-Score | 0,533 | [0,472; 0,594] |

El preprocesamiento con CLAHE mejora el AUC de 0,758 a 0,838 (test *t* pareado: *t* = 6,567, *p* = 0,003, *d* de Cohen = 2,94).

## Estructura del repositorio

```
proyecto_tfg/
├── notebook/
│   └── tfg-224-resnet-18 (4).ipynb   # Notebook principal: entrenamiento, evaluación e interpretabilidad
├── resultados/
│   ├── results.json                   # Métricas por fold (AUC, sensibilidad, especificidad, F1)
│   └── interpretability.json          # Análisis de cortes de atención vs. corte central
└── memoria/
    ├── main.txt                       # Documento LaTeX principal (plantilla FIC-UDC)
    ├── contido/                       # Capítulos de la memoria
    │   ├── Introducción.txt
    │   ├── Estado del arte.txt
    │   ├── preparacion_datos.txt
    │   ├── cascade.txt
    │   ├── Resultados.txt
    │   └── Conclusiones.txt
    └── bibliografia/
        └── bibliografía.txt           # Entradas BibTeX
```

## Conjunto de datos

El sistema se entrena y evalúa sobre **MRNet** (Bien et al., 2018), un conjunto de datos público de resonancia magnética de rodilla con 1 250 estudios y etiquetas para anomalía general, lesión del LCA y lesión meniscal.

El conjunto de datos debe descargarse directamente desde [stanfordmlgroup.github.io/competitions/mrnet](https://stanfordmlgroup.github.io/competitions/mrnet/) y no se incluye en este repositorio.

## Requisitos

El desarrollo se realizó en **Kaggle Notebooks** con GPU Tesla T4. Las dependencias principales son:

- Python 3.x
- PyTorch
- torchvision
- OpenCV (`cv2`)
- NumPy / Pandas
- Matplotlib
- SciPy
- scikit-learn

## Memoria

La memoria completa se compila con **XeLaTeX** mediante la plantilla oficial de la FIC (UDC) y está disponible en Overleaf. Los ficheros fuente en LaTeX se encuentran en el directorio `memoria/`.

## Licencia

Este repositorio contiene únicamente código y documentación académica. El conjunto de datos MRNet tiene su propia licencia de uso definida por Stanford ML Group.
