"""
Caso de Estudio N°4 - Teen Mental Health Dataset
Especialización Python for Analytics (2026)
Autora: Liz Esthefanny Marquez Panuera

Aplicación interactiva en Streamlit para el Análisis Exploratorio de Datos (EDA)
del dataset Teen_Mental_Health_Dataset.csv.

IMPORTANTE: Este análisis es exploratorio y educativo. NO constituye un
diagnóstico clínico ni sustituye la valoración de profesionales de la salud.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# Configuración general de la página
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Teen Mental Health - EDA",
    page_icon="📊",
    layout="wide",
)

sns.set_style("whitegrid")


# ====================================================================
# CLASE PRINCIPAL: DataAnalyzer
# Encapsula toda la lógica de carga, validación, clasificación,
# estadísticas, visualizaciones y filtros del dataset.
# ====================================================================
class DataAnalyzer:
    """Encapsula la carga, validación y análisis del dataset de salud
    mental adolescente."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.n_rows, self.n_cols = self.df.shape

    # ---------- Carga y validación ----------
    @staticmethod
    def cargar_csv(archivo_subido):
        """Lee un archivo CSV subido por el usuario y devuelve un
        DataFrame. Devuelve None si falla la lectura."""
        try:
            df = pd.read_csv(archivo_subido)
            return df
        except Exception as error:
            st.error(f"No se pudo leer el archivo: {error}")
            return None

    def validar_columnas_esperadas(self, columnas_esperadas: list) -> list:
        """Devuelve la lista de columnas esperadas que NO están
        presentes en el dataset cargado."""
        return [c for c in columnas_esperadas if c not in self.df.columns]

    # ---------- Clasificación de variables ----------
    def clasificar_variables(self):
        """Clasifica las columnas del DataFrame en numéricas y
        categóricas usando sus tipos de dato (dtype)."""
        numericas = self.df.select_dtypes(include=np.number).columns.tolist()
        categoricas = self.df.select_dtypes(exclude=np.number).columns.tolist()
        return numericas, categoricas

    # ---------- Estadísticas descriptivas ----------
    def estadisticas_descriptivas(self, columnas_numericas: list) -> pd.DataFrame:
        return self.df[columnas_numericas].describe().T

    def moda_variables(self, columnas: list) -> pd.Series:
        return self.df[columnas].mode().iloc[0]

    # ---------- Valores faltantes y duplicados ----------
    def resumen_nulos(self) -> pd.DataFrame:
        nulos = self.df.isnull().sum()
        porcentaje = (nulos / len(self.df)) * 100
        resumen = pd.DataFrame({"nulos": nulos, "porcentaje_%": porcentaje.round(2)})
        return resumen

    def contar_duplicados(self) -> int:
        return int(self.df.duplicated().sum())

    # ---------- Filtros dinámicos ----------
    def filtrar(self, age_range, gender_sel, platform_sel, interaction_sel):
        df_filtrado = self.df[
            (self.df["age"] >= age_range[0]) & (self.df["age"] <= age_range[1])
        ]
        if gender_sel:
            df_filtrado = df_filtrado[df_filtrado["gender"].isin(gender_sel)]
        if platform_sel:
            df_filtrado = df_filtrado[df_filtrado["platform_usage"].isin(platform_sel)]
        if interaction_sel:
            df_filtrado = df_filtrado[
                df_filtrado["social_interaction_level"].isin(interaction_sel)
            ]
        return df_filtrado

    # ---------- Visualizaciones ----------
    def histograma(self, columna: str, ax=None, bins: int = 20):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 3.5))
        else:
            fig = ax.figure
        sns.histplot(self.df[columna].dropna(), bins=bins, kde=True, ax=ax, color="#4C72B0")
        ax.set_title(f"Distribución de {columna}")
        ax.set_xlabel(columna)
        ax.set_ylabel("Frecuencia")
        return fig

    def barras_categorica(self, columna: str):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        conteo = self.df[columna].value_counts()
        sns.barplot(x=conteo.index, y=conteo.values, ax=ax, palette="viridis")
        ax.set_title(f"Frecuencia de {columna}")
        ax.set_ylabel("Conteo")
        ax.set_xlabel(columna)
        return fig

    def boxplot_numerico_vs_categorico(self, num_col: str, cat_col: str):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.boxplot(data=self.df, x=cat_col, y=num_col, ax=ax, palette="Set2")
        ax.set_title(f"{num_col} según {cat_col}")
        return fig

    def barras_apiladas_categorico_vs_categorico(self, col_a: str, col_b: str):
        tabla = pd.crosstab(self.df[col_a], self.df[col_b], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        tabla.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        ax.set_ylabel("Porcentaje (%)")
        ax.set_title(f"{col_a} vs {col_b} (proporciones)")
        ax.legend(title=col_b, bbox_to_anchor=(1.02, 1), loc="upper left")
        return fig


# ====================================================================
# FUNCIONES AUXILIARES
# ====================================================================
def clasificar_columna(serie: pd.Series) -> str:
    """Función personalizada (Ítem 2) que clasifica una sola columna
    como 'Numérica' o 'Categórica'."""
    if pd.api.types.is_numeric_dtype(serie):
        return "Numérica"
    return "Categórica"


COLUMNAS_ESPERADAS = [
    "age", "gender", "daily_social_media_hours", "platform_usage",
    "sleep_hours", "screen_time_before_sleep", "academic_performance",
    "physical_activity", "social_interaction_level", "stress_level",
    "anxiety_level", "addiction_level", "depression_label",
]


# ====================================================================
# SIDEBAR - Navegación principal
# ====================================================================
st.sidebar.title("📊 Navegación")
modulo = st.sidebar.radio(
    "Selecciona un módulo:",
    ["🏠 Home", "📁 Carga de datos", "🔎 Análisis Exploratorio (EDA)"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Análisis exploratorio y educativo. No constituye diagnóstico clínico."
)

# Estado persistente entre módulos
if "df" not in st.session_state:
    st.session_state.df = None


# ====================================================================
# MÓDULO 1: HOME
# ====================================================================
if modulo == "🏠 Home":
    st.title("📊 Teen Mental Health Dataset - Análisis Exploratorio de Datos")

    st.markdown("""
    ### Objetivo del análisis
    Esta aplicación explora, limpia, transforma y visualiza el dataset
    **Teen_Mental_Health_Dataset.csv** con el fin de identificar patrones
    entre hábitos digitales, descanso, actividad física, interacción social
    y variables de bienestar en adolescentes. **No** se construyen modelos
    predictivos ni se emiten diagnósticos clínicos: el enfoque es
    exploratorio y orientado a la toma de decisiones informadas.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👩‍🎓 Datos de la autora")
        st.markdown("""
        - **Nombre:** Liz Esthefanny Marquez Panuera
        - **Curso / Especialización:** Python for Analytics
        - **Año:** 2026
        """)
    with col2:
        st.subheader("🛠️ Tecnologías utilizadas")
        st.markdown("""
        - Python 3
        - Pandas / NumPy
        - Matplotlib / Seaborn
        - Streamlit
        """)

    st.subheader("📁 Sobre el dataset")
    st.markdown("""
    El dataset contiene **1,200 registros** y **13 variables** sobre
    adolescentes de **13 a 19 años**, incluyendo uso diario de redes
    sociales, plataforma utilizada, horas de sueño, tiempo de pantalla
    antes de dormir, rendimiento académico, actividad física e interacción
    social, además de escalas de estrés, ansiedad, adicción y la etiqueta
    binaria `depression_label`. No presenta valores nulos ni duplicados.
    """)

    st.info("Dirígete al módulo **📁 Carga de datos** en la barra lateral para comenzar.")


# ====================================================================
# MÓDULO 2: CARGA DEL DATASET
# ====================================================================
elif modulo == "📁 Carga de datos":
    st.title("📁 Carga del dataset")

    archivo = st.file_uploader("Sube el archivo Teen_Mental_Health_Dataset.csv", type=["csv"])

    if archivo is not None:
        df_cargado = DataAnalyzer.cargar_csv(archivo)

        if df_cargado is not None:
            analyzer_temp = DataAnalyzer(df_cargado)
            faltantes = analyzer_temp.validar_columnas_esperadas(COLUMNAS_ESPERADAS)

            if faltantes:
                st.error(
                    "El archivo fue leído, pero faltan columnas esperadas: "
                    f"{', '.join(faltantes)}"
                )
            else:
                st.session_state.df = df_cargado
                st.success("✅ Archivo cargado y validado correctamente.")

                st.subheader("Vista previa del dataset")
                st.dataframe(df_cargado.head())

                col1, col2 = st.columns(2)
                col1.metric("Número de filas", df_cargado.shape[0])
                col2.metric("Número de columnas", df_cargado.shape[1])
    else:
        st.warning("⬆️ Sube el archivo CSV para habilitar el análisis exploratorio.")


# ====================================================================
# MÓDULO 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ====================================================================
elif modulo == "🔎 Análisis Exploratorio (EDA)":
    st.title("🔎 Análisis Exploratorio de Datos (EDA)")

    if st.session_state.df is None:
        st.error("⚠️ Debes cargar el dataset en el módulo '📁 Carga de datos' antes de continuar.")
        st.stop()

    df = st.session_state.df
    analyzer = DataAnalyzer(df)
    numericas, categoricas = analyzer.clasificar_variables()

    # Quitamos depression_label de las "numéricas" a efectos de gráficos,
    # ya que en realidad es una etiqueta categórica binaria.
    numericas_analisis = [c for c in numericas if c != "depression_label"]

    tabs = st.tabs([
        "1. Info general", "2. Clasificación", "3. Descriptivas",
        "4. Valores faltantes", "5. Distribuciones", "6. Categóricas",
        "7. Bivariado num-cat", "8. Bivariado cat-cat",
        "9. Análisis dinámico", "10. Hallazgos clave",
    ])

    # ---------------- Ítem 1: Información general ----------------
    with tabs[0]:
        st.header("1️⃣ Información general del dataset")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Tipos de datos (.info())")
            info_df = pd.DataFrame({
                "columna": df.columns,
                "tipo_dato": [str(t) for t in df.dtypes],
                "no_nulos": df.notnull().sum().values,
            })
            st.dataframe(info_df, use_container_width=True)

        with col2:
            st.subheader("Valores nulos y duplicados")
            st.metric("Total de valores nulos", int(df.isnull().sum().sum()))
            st.metric("Registros duplicados", analyzer.contar_duplicados())
            if df.isnull().sum().sum() == 0 and analyzer.contar_duplicados() == 0:
                st.success("El dataset no presenta nulos ni duplicados.")

    # ---------------- Ítem 2: Clasificación de variables ----------------
    with tabs[1]:
        st.header("2️⃣ Clasificación de variables")
        st.caption("Clasificación realizada con la función personalizada `clasificar_columna()`.")

        clasificacion = pd.DataFrame({
            "columna": df.columns,
            "clasificación": [clasificar_columna(df[c]) for c in df.columns],
        })

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(clasificacion, use_container_width=True)
        with col2:
            conteo_tipo = clasificacion["clasificación"].value_counts()
            st.metric("Variables numéricas", int(conteo_tipo.get("Numérica", 0)))
            st.metric("Variables categóricas", int(conteo_tipo.get("Categórica", 0)))

    # ---------------- Ítem 3: Estadísticas descriptivas ----------------
    with tabs[2]:
        st.header("3️⃣ Estadísticas descriptivas")
        desc = analyzer.estadisticas_descriptivas(numericas_analisis)
        st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

        st.markdown("""
        **Lectura rápida:** la media y la mediana (percentil 50%) permiten
        comparar el "centro" de cada variable; una diferencia grande entre
        ambas sugiere asimetría. El rango intercuartílico (percentil 75%
        menos percentil 25%) muestra la dispersión típica, y valores
        mínimos/máximos muy alejados del resto pueden indicar posibles
        valores extremos a revisar (no necesariamente errores).
        """)

    # ---------------- Ítem 4: Análisis de valores faltantes ----------------
    with tabs[3]:
        st.header("4️⃣ Análisis de valores faltantes")
        resumen_nulos = analyzer.resumen_nulos()
        st.dataframe(resumen_nulos, use_container_width=True)

        if resumen_nulos["nulos"].sum() == 0:
            st.success(
                "El dataset no presenta valores faltantes. Por ello, no se "
                "requiere imputación ni eliminación de registros: el análisis "
                "puede centrarse en distribuciones y comparaciones entre grupos."
            )
        else:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x=resumen_nulos.index, y=resumen_nulos["porcentaje_%"], ax=ax)
            ax.set_ylabel("% de nulos")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

    # ---------------- Ítem 5: Distribución de variables numéricas ----------------
    with tabs[4]:
        st.header("5️⃣ Distribución de variables numéricas")

        col_sel = st.selectbox("Selecciona una variable numérica:", numericas_analisis)
        fig = analyzer.histograma(col_sel)
        st.pyplot(fig)

        st.subheader("Comparación de escalas: estrés, ansiedad y adicción")
        st.caption(
            "Comparación puramente descriptiva de las escalas registradas en el "
            "dataset; no debe interpretarse como un análisis o diagnóstico clínico."
        )
        col1, col2, col3 = st.columns(3)
        for col, colname in zip([col1, col2, col3], ["stress_level", "anxiety_level", "addiction_level"]):
            with col:
                fig_mini = analyzer.histograma(colname, bins=10)
                st.pyplot(fig_mini)

    # ---------------- Ítem 6: Análisis de variables categóricas ----------------
    with tabs[5]:
        st.header("6️⃣ Análisis de variables categóricas")

        cat_sel = st.selectbox("Selecciona una variable categórica:", categoricas)
        col1, col2 = st.columns([1, 1])

        with col1:
            conteo = df[cat_sel].value_counts()
            proporcion = (df[cat_sel].value_counts(normalize=True) * 100).round(2)
            tabla_cat = pd.DataFrame({"conteo": conteo, "proporción_%": proporcion})
            st.dataframe(tabla_cat, use_container_width=True)

        with col2:
            st.pyplot(analyzer.barras_categorica(cat_sel))

    # ---------------- Ítem 7: Bivariado numérico vs categórico ----------------
    with tabs[6]:
        st.header("7️⃣ Análisis bivariado: numérico vs categórico")
        st.caption("Todas las comparaciones se realizan frente a `depression_label`.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Horas de redes sociales según depression_label")
            st.pyplot(analyzer.boxplot_numerico_vs_categorico(
                "daily_social_media_hours", "depression_label"))
        with col2:
            st.subheader("Horas de sueño según depression_label")
            st.pyplot(analyzer.boxplot_numerico_vs_categorico(
                "sleep_hours", "depression_label"))

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Rendimiento académico según depression_label")
            st.pyplot(analyzer.boxplot_numerico_vs_categorico(
                "academic_performance", "depression_label"))
        with col4:
            st.subheader("Actividad física según depression_label")
            st.pyplot(analyzer.boxplot_numerico_vs_categorico(
                "physical_activity", "depression_label"))

    # ---------------- Ítem 8: Bivariado categórico vs categórico ----------------
    with tabs[7]:
        st.header("8️⃣ Análisis bivariado: categórico vs categórico")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Plataforma vs depression_label")
            st.pyplot(analyzer.barras_apiladas_categorico_vs_categorico(
                "platform_usage", "depression_label"))
        with col2:
            st.subheader("Interacción social vs depression_label")
            st.pyplot(analyzer.barras_apiladas_categorico_vs_categorico(
                "social_interaction_level", "depression_label"))

        st.subheader("Género vs plataforma utilizada")
        st.pyplot(analyzer.barras_apiladas_categorico_vs_categorico(
            "gender", "platform_usage"))

    # ---------------- Ítem 9: Análisis basado en parámetros seleccionados ----------------
    with tabs[8]:
        st.header("9️⃣ Análisis dinámico según parámetros seleccionados")

        st.subheader("Filtros")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            edad_min, edad_max = int(df["age"].min()), int(df["age"].max())
            rango_edad = st.slider("Rango de edad", edad_min, edad_max, (edad_min, edad_max))
        with c2:
            generos_sel = st.multiselect("Género", df["gender"].unique().tolist())
        with c3:
            plataformas_sel = st.multiselect("Plataforma", df["platform_usage"].unique().tolist())
        with c4:
            interaccion_sel = st.multiselect(
                "Nivel de interacción social", df["social_interaction_level"].unique().tolist())

        mostrar_tabla = st.checkbox("Mostrar tabla de datos filtrados")

        df_filtrado = analyzer.filtrar(rango_edad, generos_sel, plataformas_sel, interaccion_sel)
        st.caption(f"Registros que cumplen los filtros: **{len(df_filtrado)}**")

        if mostrar_tabla:
            st.dataframe(df_filtrado, use_container_width=True)

        st.markdown("---")
        st.subheader("Comparación dinámica de variables")
        colv1, colv2 = st.columns(2)
        with colv1:
            var_bienestar = st.selectbox(
                "Variable de bienestar", ["stress_level", "anxiety_level", "addiction_level"])
        with colv2:
            var_habito = st.selectbox(
                "Variable de hábito digital",
                ["daily_social_media_hours", "screen_time_before_sleep", "sleep_hours"])

        if len(df_filtrado) > 0:
            analyzer_filtrado = DataAnalyzer(df_filtrado)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                data=df_filtrado, x=var_habito, y=var_bienestar,
                hue="depression_label", palette="Set1", ax=ax)
            ax.set_title(f"{var_bienestar} vs {var_habito} (según filtros aplicados)")
            st.pyplot(fig)
        else:
            st.warning("No hay registros que cumplan con los filtros seleccionados.")

    # ---------------- Ítem 10: Hallazgos clave ----------------
    with tabs[9]:
        st.header("🔟 Hallazgos clave")

        promedio_sueño_dep = df.groupby("depression_label")["sleep_hours"].mean()
        promedio_redes_dep = df.groupby("depression_label")["daily_social_media_hours"].mean()
        promedio_ansiedad_dep = df.groupby("depression_label")["anxiety_level"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Horas de sueño promedio (label=1)", f"{promedio_sueño_dep.get(1, np.nan):.2f}")
        col2.metric("Horas de redes promedio (label=1)", f"{promedio_redes_dep.get(1, np.nan):.2f}")
        col3.metric("Ansiedad promedio (label=1)", f"{promedio_ansiedad_dep.get(1, np.nan):.2f}")

        st.pyplot(analyzer.boxplot_numerico_vs_categorico("sleep_hours", "depression_label"))

        st.subheader("📌 Conclusiones")
        st.markdown("""
        1. **Sueño y bienestar:** los adolescentes con `depression_label = 1`
           muestran, en promedio, menos horas de sueño que quienes tienen
           `depression_label = 0`, lo que sugiere una asociación exploratoria
           entre el descanso y el bienestar reportado, sin implicar causalidad.
        2. **Uso de redes sociales:** el grupo con etiqueta positiva presenta
           un mayor promedio de horas diarias en redes sociales, patrón
           consistente con lo observado en el análisis bivariado del Ítem 7.
        3. **Interacción social:** los niveles bajos de interacción social
           se asocian, de forma descriptiva, con una mayor proporción de
           `depression_label = 1` frente a los niveles medio y alto.
        4. **Escalas de estrés y ansiedad:** ambas escalas muestran
           distribuciones más desplazadas hacia valores altos en el grupo
           con etiqueta positiva, reforzando la coherencia interna del
           dataset entre variables relacionadas.
        5. **Plataforma utilizada:** no se observan diferencias marcadas en
           la proporción de `depression_label` entre quienes usan Instagram,
           TikTok o ambas plataformas, lo que sugiere que la plataforma en
           sí misma no sería, en este dataset, un factor diferenciador tan
           relevante como el tiempo de uso o el descanso.

        *Estas observaciones son de carácter exploratorio y educativo;
        no constituyen un diagnóstico clínico ni deben usarse para tomar
        decisiones médicas. Cualquier preocupación real sobre bienestar
        adolescente debe canalizarse hacia profesionales de la salud
        mental calificados.*
        """)
