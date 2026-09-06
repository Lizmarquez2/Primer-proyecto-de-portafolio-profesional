# 📊 Teen Mental Health Dataset — Análisis Exploratorio de Datos (EDA)

**Caso de Estudio N°4 — Especialización Python for Analytics (2026)**
**Autora:** Liz Esthefanny Marquez Panuera

---

## 📝 Descripción del proyecto

Aplicación interactiva construida en **Python + Streamlit** para realizar un
Análisis Exploratorio de Datos (EDA) del dataset `Teen_Mental_Health_Dataset.csv`,
que contiene **1,200 registros y 13 variables** sobre adolescentes de 13 a 19
años (hábitos digitales, descanso, actividad física, interacción social y
variables de bienestar).

El objetivo del proyecto **no es predictivo ni clínico**: se enfoca en
explorar, limpiar, transformar y visualizar los datos para identificar
patrones exploratorios que apoyen la toma de decisiones informadas. Ninguno
de los resultados aquí presentados constituye un diagnóstico clínico ni
sustituye la valoración de profesionales de la salud mental.

La aplicación está organizada en 3 módulos navegables desde el sidebar:
**Home**, **Carga del dataset** y **Análisis Exploratorio (EDA)**, este
último con 10 ítems de análisis distribuidos en pestañas (`st.tabs`).

## ⚙️ Instrucciones de ejecución

### Ejecutar en local

```bash
git clone (https://github.com/Lizmarquez2/Primer-proyecto-de-portafolio-profesional)
cd Primer-proyecto-de-portafolio-profesional
pip install -r requirements.txt
streamlit run app.py
```

Luego, dentro de la aplicación, sube el archivo `Teen_Mental_Health_Dataset.csv`
en el módulo **📁 Carga de datos** para habilitar el análisis exploratorio.

### Usar la versión desplegada

No requiere instalación. Accede directamente a la aplicación publicada en
Streamlit Community Cloud.

---

## 📋 Descripción de las variables principales

| Variable | Descripción |
| --- | --- |
| `age` | Edad del adolescente, entre 13 y 19 años |
| `gender` | Género registrado |
| `daily_social_media_hours` | Horas diarias de uso de redes sociales |
| `platform_usage` | Plataforma utilizada: Instagram, TikTok o ambas |
| `sleep_hours` | Horas de sueño por día |
| `screen_time_before_sleep` | Tiempo de pantalla antes de dormir, en horas |
| `academic_performance` | Indicador de rendimiento académico |
| `physical_activity` | Horas de actividad física |
| `social_interaction_level` | Nivel de interacción social: bajo, medio o alto |
| `stress_level` | Nivel de estrés en escala de 1 a 10 |
| `anxiety_level` | Nivel de ansiedad en escala de 1 a 10 |
| `addiction_level` | Nivel de dependencia o uso problemático en escala de 1 a 10 |
| `depression_label` | Etiqueta binaria del dataset: 0 = ausencia, 1 = presencia de la condición etiquetada |

El dataset no presenta valores nulos ni registros duplicados.

---

## ✅ Conclusiones finales

Cinco conclusiones basadas en los análisis realizados, cada una vinculada a
una evidencia visual o estadística concreta dentro de la aplicación:

1. **El descanso se asocia con el bienestar reportado.** Los adolescentes con
   `depression_label = 1` muestran, en promedio, menos horas de sueño que
   quienes tienen `depression_label = 0`.
   *Evidencia: boxplot "Horas de sueño según depression_label" — Ítem 7 (Análisis bivariado numérico vs categórico).*

2. **El uso de redes sociales es mayor en el grupo con etiqueta positiva.**
   El promedio de horas diarias en redes sociales es más alto entre quienes
   tienen `depression_label = 1`.
   *Evidencia: boxplot "Horas de redes sociales según depression_label" — Ítem 7.*

3. **La baja interacción social se asocia a una mayor proporción de la etiqueta.**
   Los niveles bajos de interacción social muestran, de forma descriptiva,
   una proporción mayor de `depression_label = 1` frente a niveles medio y alto.
   *Evidencia: gráfico de barras apiladas "Interacción social vs depression_label" — Ítem 8 (Análisis bivariado categórico vs categórico).*

4. **Las escalas de estrés y ansiedad son consistentes entre sí.** Ambas
   escalas se desplazan hacia valores más altos en el grupo con etiqueta
   positiva, reforzando la coherencia interna del dataset.
   *Evidencia: histogramas comparativos de `stress_level`, `anxiety_level` y `addiction_level` — Ítem 5 (Distribución de variables numéricas).*

5. **La plataforma utilizada no es, por sí sola, un factor diferenciador
   relevante.** No se observan diferencias marcadas en la proporción de
   `depression_label` entre quienes usan Instagram, TikTok o ambas.
   *Evidencia: gráfico de barras apiladas "Plataforma vs depression_label" — Ítem 8.*

*Estas conclusiones tienen carácter exploratorio y educativo, orientado a la
toma de decisiones informadas — no constituyen predicciones ni diagnósticos.*

---

## 🛠️ Tecnologías utilizadas

Python 3 · Pandas · NumPy · Matplotlib · Seaborn · Streamlit
