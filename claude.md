# Contexto del proyecto: TFG Detección de lesiones del LCA en RM de rodilla

## Quién soy y qué es esto
- Alumno: Jacobo Cousillas Taboada, Grado en Ciencia e Ingeniería de Datos (UDC).
- Tutores: Laura Morán Fernández y Daniel Vila de la Cruz.
- TFG sobre detección automática de lesiones del ligamento cruzado anterior (LCA)
  en resonancia magnética de rodilla mediante aprendizaje profundo.
- La memoria está en LaTeX (plantilla FIC de la UDC), compila con XeLaTeX en Overleaf.

## Reglas de redacción (MUY IMPORTANTE, respetar siempre)
- Escribir en español, registro académico, en tercera persona impersonal ("se evalúa",
  "se propone"), nunca en primera persona salvo en la sección de motivación personal.
- EVITAR ANGLICISMOS. Usar: "conjunto de datos" (no dataset), "flujo de preprocesamiento"
  (no pipeline), "aprendizaje profundo" (no deep learning), "corte" (no slice),
  "lote" (no batch), "época" (no epoch), "umbral" (no threshold), "aprendizaje por
  transferencia" (no transfer learning). Términos técnicos sin traducción clara
  (backbone, attention pooling, max pooling, dropout, softmax) van en \textit{}.
- NO traducir nombres de código (MRNetDataset, torch.utils.data.Dataset, conv1, layer4).
- Usar "LCA" de forma consistente, nunca "ACL", salvo en nombres de variables del código.
- Números: miles con \, (1\,250 no 1.250). Porcentajes con espacio fino: 80,6\,\%.
- Decimales con coma, no punto (0,838 no 0.838).
- La plantilla usa "Cuadro" para tablas (no "Tabla") salvo que se haya redefinido.
- No pasar de un título de sección directamente a una subsección: poner siempre un
  párrafo introductorio entre medias.
- Tono prudente, no categórico. "Ha mostrado resultados prometedores", no "es superior".
- Citar solo fuentes académicas (no webs educativas tipo kenhub).
- Evitar autorreferencia excesiva ("en el presente trabajo"): usar formas impersonales.

## Resultados definitivos (NO inventar, usar SIEMPRE estos)
Ejecución final: 224x224, ResNet-18, target_sensitivity=0.85, validación cruzada 5-fold.

Detección de LCA (Fase 2), media 5-fold:
- AUC: 0,838  (IC95% [0,811; 0,865], std 0,022)
- Sensibilidad: 84,7 %  (IC95% [77,4; 92,0])
- Especificidad: 64,0 %  (IC95% [52,1; 75,8])
- F1: 0,533  (IC95% [0,472; 0,594])
- Umbral medio: 0,292

Por fold (Sens / Spec / F1 / AUC / Umbral):
- Fold 1: 86,5 / 46,0 / 0,441 / 0,826 / 0,210
- Fold 2: 88,5 / 64,6 / 0,548 / 0,857 / 0,270
- Fold 3: 73,1 / 73,2 / 0,531 / 0,812 / 0,390
- Fold 4: 86,8 / 70,6 / 0,586 / 0,870 / 0,330  (mejor fold)
- Fold 5: 88,7 / 65,5 / 0,560 / 0,824 / 0,260

Fase 1 (triaje, detección de anomalías), media 5-fold:
- AUC 0,888, Sens 81,1 %, Spec 80,0 %.
- Con umbral de triaje 0,30: sensibilidad 99,0 %, filtra el 4,4 % de casos normales.

Ablación por plano (mejor fold), AUC:
- Solo sagital: 0,893 (Sens 86,8 / Spec 77,7 / F1 0,643)
- Solo coronal: 0,555 (apenas mejor que el azar)
- Solo axial: 0,790
- Sagital+coronal: 0,863
- Completo: 0,870
Esto valida los pesos de fusión: sagital 0,55 / coronal 0,35 / axial 0,10 para LCA.

Análisis estadístico:
- CLAHE: AUC 0,758 -> 0,838. Test t pareado t=6,567, p=0,003. Cohen d=2,94 (grande).
  SIGNIFICATIVO. (Atención: en versiones viejas de la memoria pone p=0,030, es ERROR,
  el valor correcto es p=0,003).
- Attention pooling vs max pooling: AUC 0,793 -> 0,838. t=2,659, p=0,056. Cohen d=1,19.
  NO significativo (potencia limitada con n=5), pero efecto grande.

Interpretabilidad (slice de atención vs slice central):
- Sagital: offset medio 7,0 cortes, máximo 14, difiere del centro en 99,2 % de casos.
- Coronal: offset medio 9,7, 99,2 %.
- Axial: offset medio 7,5, 96,8 %.

Comparativa de backbones:
- ResNet-18: AUC 0,838 (elegido).
- ResNet-50 (ejecutado en GPU del tutor): AUC 0,823, peor pese a tener más parámetros.
- EfficientNet-B0: convergencia muy lenta, descartado.

Comparativa de resoluciones: 128px AUC 0,819 -> 224px AUC 0,838.

Análisis de errores (mejor fold): 7 falsos negativos, 58 falsos positivos.
- FN: prob media 21,5 %, 6/7 cerca del umbral, solo 1 error con alta confianza.
- FP: prob media 53,0 %, 33/58 cerca del umbral.

Eficiencia: entrenamiento 8 h 40 min en Tesla T4. Inferencia 0,08 s/caso.
Consumo 0,61 kWh por ejecución.

## Caso de estudio personal (la guinda del TFG)
RM real propia, de un hospital, secuencia PD con supresión grasa (distinta de MRNet).
Diagnóstico clínico real: rotura parcial del LCA + lesión del menisco externo
(la del menisco no se detectó en el informe radiológico inicial, tardaron 7 meses).
Resultado del modelo:
- Anomalía (Fase 1): 0,651 -> detectada.
- LCA (Fase 2): 0,847 -> sospecha de lesión. (sagital 0,998, coronal 0,596, axial 0,896)
- Menisco (Fase 2): 0,816 -> sospecha de lesión.
El modelo detectó AMBAS lesiones, incluida la que los radiólogos pasaron por alto.
Limitación: en coronal el corte de máxima atención cae en el primero (posible efecto
del rellenado de bordes).

## Configuración del modelo (CONFIG)
- Resolución 224x224, 32 cortes por volumen, 1 canal (escala de grises).
- Backbone ResNet-18 preentrenada en ImageNet, 6 modelos (3 por fase).
- CLAHE clip_limit=2.0, grid 8x8. Máscara gaussiana anatómica solo en sagital
  (centro 0,40/0,50, sigma 0,30/0,35, peso mínimo 0,3).
- Fusión ponderada en espacio de probabilidades. Pesos LCA 0,55/0,35/0,10,
  menisco 0,30/0,55/0,15.
- Fase 1: Focal Loss (alpha 0,25, gamma 2,0). Fase 2: WeightedBCE (pos_weight 2,0).
- Adam, LR 1e-4, batch 2, dropout 0,5, weight decay 1e-4.
- Congeladas layer1-3, se entrena layer4 + fc + attention. 25,2M params entrenables.
- Early stopping patience 7, ReduceLROnPlateau patience 3. Semilla 42.
- Validación cruzada 5-fold estratificada: 800 train / 200 val / 250 test por fold.
- Umbral clínico: maximiza especificidad con Sens >= 85 %, calculado en validación.

## Estructura de la memoria (6 capítulos)
1. Introducción (con motivación personal, objetivos, alcance, planificación,
   costes, herramientas, estructura).
2. Estado del arte.
3. Preparación y análisis de datos.
4. Sistema CASCADE: arquitectura y entrenamiento.
5. Resultados.
6. Conclusiones (PENDIENTE de escribir si no existe).

## Cosas que el tribunal/tutores ya pidieron (no repetir errores)
- La introducción no debe entrar en detalles técnicos concretos (esos van a
  metodología/resultados). Debe contextualizar, motivar y presentar objetivos.
- Los objetivos se formulan de forma genérica (el QUÉ, no el CÓMO), sin nombrar
  técnicas concretas. No separar objetivo general y específicos en subsecciones.
- El estado del arte habla de la literatura, no de decisiones propias del trabajo.
- Cuidado con tablas y figuras que se salen de márgenes (usar \resizebox).
- Resolver referencias rotas (?? en el PDF) creando los capítulos/labels que falten.
- No adelantar sucesos que voy a explicar en otras posteriores secciones, es decir no hacer spoiler

## Tareas típicas que pediré
- Revisar coherencia de cifras entre capítulos (deben coincidir con los resultados
  de arriba).
- Corregir anglicismos.
- Escribir o pulir secciones concretas.
- Verificar que toda \cite{} tenga su entrada en bibliografia.bib.
- Comprobar que no haya referencias \ref{} rotas.