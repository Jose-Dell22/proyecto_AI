# Requerimientos


---

## Página 1


Campo Descripción
Título del 
Requerimiento
Carga y validación de imágenes MRI
Código de 
identificación
RF-01
Tipo de 
requerimiento
Funcional
Versión 1.0
Fuente Historia Usuario 1
Descripción El sistema debe permitir al usuario cargar una imagen de resonancia 
magnética cerebral (MRI) en secuencia T1 desde su dispositivo. 
Antes de habilitar el análisis, el sistema valida automáticamente el 
formato del archivo (JPG o PNG), su tamaño máximo (10 MB) y su 
integridad como imagen válida. La imagen debe corresponder a un 
corte axial T1, ya que el modelo fue entrenado exclusivamente con 
imágenes en dicho protocolo. Cualquier archivo que no cumpla estas 
condiciones es rechazado con un mensaje descriptivo en español.
Justificación Garantizar la integridad de los datos de entrada es crítico en un 
sistema de diagnóstico asistido. El procesamiento de archivos 
corruptos, en formato incorrecto o que no correspondan a imágenes 
MRI reales puede producir predicciones erróneas con consecuencias 
clínicas graves.
Precondiciones• El usuario ha iniciado la sesión y se encuentra en la interfaz de 
análisis.
• La interfaz de carga de imágenes está operativa y accesible.
Restricciones • Solo se aceptan archivos con extensión .jpg o .png.
• El tamaño máximo del archivo es de 10 MB.
• No es posible iniciar la predicción sin una imagen válida cargada.
• Solo se puede tener una imagen activa a la vez; cargar una nueva 
reemplaza la anterior.
Prioridad Alta/Must
Dificultad Normal
Dependencia Ninguna
Actores Médico especialista, Profesional de la salud

---

## Página 2


Campo Descripción
Entradas • Archivo de imagen seleccionado por el usuario (JPG o PNG), 
correspondiente a una secuencia MRI T1.
Proceso • El usuario presiona el botón 'Cargar imagen'.
• El sistema abre el selector de archivos del dispositivo.
• El usuario selecciona el archivo.
• El sistema verifica la extensión del archivo (.jpg, .png).
• El sistema verifica que el tamaño no supere 10 MB.
• El sistema verifica que el contenido sea una imagen válida (no 
corrupta).
• Si todas las validaciones pasan, el sistema muestra la vista previa 
de la imagen y habilita el botón 'Predecir'.
• Si alguna validación falla, el sistema muestra el mensaje de error 
correspondiente y no habilita el botón 'Predecir'.
Salida • Vista previa de la imagen MRI cargada, visible en menos de 2 
segundos.
• Botón 'Predecir' habilitado.
• Mensaje de error específico en caso de fallo de validación.
Postcondiciones• La imagen validada queda disponible en memoria para ser enviada 
al backend en la siguiente acción del usuario.
Criterios de 
aceptación
• CA-01.1: El sistema acepta sin error archivos con extensión .jpg y 
.png.
• CA-01.2: El sistema rechaza archivos con extensión no permitida 
mostrando: 'Formato no soportado. Use JPG o PNG.'
• CA-01.3: El sistema rechaza archivos mayores a 10 MB mostrando: 
'El archivo excede el tamaño máximo de 10 MB.'
• CA-01.4: El sistema rechaza archivos corruptos o no legibles como 
imagen mostrando: 'Archivo no válido o corrupto.'
• CA-01.5: La vista previa de la imagen se renderiza en menos de 2 
segundos tras la validación exitosa.

---

## Página 3


Campo Descripción
• CA-01.6: El botón 'Predecir' permanece deshabilitado si no hay una 
imagen válida cargada.
Requerimientos 
no funcionales
• Usabilidad: los mensajes de error son claros, específicos y en 
español.
• Rendimiento: la validación del archivo se completa en menos de 1 
segundo para archivos de hasta 10 MB.
• Fiabilidad: el sistema no falla silenciosamente; cualquier error de 
validación genera retroalimentación visible al usuario.
Estado Pendiente
Observaciones La imagen debe corresponder a una secuencia MRI T1 (corte axial), 
ya que el modelo DenseNet121 + CBAM fue entrenado 
exclusivamente con imágenes en dicho protocolo. El uso de 
secuencias distintas (T2, FLAIR, etc.) puede producir predicciones 
incorrectas. Esta validación de protocolo es responsabilidad del 
usuario; el sistema no puede verificar automáticamente la secuencia 
MRI desde un JPG o PNG.
Campo Descripción
Título del 
Requerimiento
Ejecución de predicción con DenseNet121 + CBAM
Código de 
identificación
RF-02
Tipo de 
requerimiento
Funcional
Versión 1.0
Fuente Historia Usuario 2
Descripción El sistema debe permitir al usuario ejecutar el análisis de clasificación 
de la imagen MRI cargada mediante un único clic. El frontend envía la 
imagen al backend a través de la API REST, el backend ejecuta la 
inferencia exclusivamente con el modelo DenseNet121+CBAM, y 
retorna el resultado al frontend. Durante el procesamiento, el sistema 
muestra un indicador de progreso visible que bloquea el envío de 
solicitudes duplicadas.
Justificación La predicción es la funcionalidad central del sistema. Debe ser simple 
de invocar, rápida (máx. 5 s) y robusta ante errores de red o del 
servidor. El bloqueo de solicitudes simultáneas evita condiciones de 
carrera y resultados inconsistentes.

---

## Página 4


Campo Descripción
Precondiciones• RF-01 completado: existe una imagen MRI válida cargada en 
memoria.
• El backend está activo y accesible en la red.
Restricciones • El modelo de inferencia utilizado es exclusivamente DenseNet121 + 
CBAM; no se permite sustituirlo por otro.
• No se permiten múltiples solicitudes simultáneas de la misma sesión.
• El tiempo de respuesta del backend no debe superar 5 segundos.
Prioridad Alta/Must
Dificultad Difícil
Dependencia RF-01-
Actores Profesional de la salud, Sistema (frontend/backend)
Entradas • Imagen MRI T1 validada (formato JPG o PNG), proveniente de RF-
01.
Proceso • El usuario presiona el botón 'Predecir'.
• El sistema verifica que existe una imagen válida en memoria (guard 
clause).
• El frontend deshabilita el botón 'Predecir' y muestra el indicador de 
progreso.
• El frontend envía una solicitud HTTP POST al endpoint /predict con 
la imagen como cuerpo de la petición (multipart/form-data).
• El backend recibe la imagen, aplica el preprocesamiento requerido 
por el modelo y ejecuta la inferencia con DenseNet121 + CBAM.
• El backend retorna una respuesta JSON con la clase predicha, 
probabilidades por clase y el mapa de activación Grad-CAM.
• El frontend oculta el indicador de progreso, habilita nuevamente el 
botón 'Predecir' y pasa el resultado a RF-03 y RF-04.

---

## Página 5


Campo Descripción
Salida • Objeto JSON con: clase predicha, nivel de confianza, probabilidades 
de las 4 clases y mapa Grad-CAM.
• Indicador de progreso visible durante el procesamiento.
• Mensaje de error específico en caso de fallo de conexión o error 
interno del servidor.
Postcondiciones• El resultado de la predicción queda almacenado temporalmente en 
el estado de la aplicación y disponible para RF-03 y RF-04.
Criterios de 
aceptación
• CA-02.1: El botón 'Predecir' está deshabilitado si no hay imagen 
cargada.
• CA-02.2: Al presionar 'Predecir', aparece un indicador de progreso 
visible en menos de 200 ms.
• CA-02.3: El tiempo total entre el envío y la recepción del resultado 
es ≤ 5 segundos bajo condiciones normales de red.
• CA-02.4: Si el backend no responde, el frontend muestra: 'Error de 
conexión con el servidor. Intente nuevamente.'
• CA-02.5: Si el backend retorna un error interno (HTTP 500), el 
frontend muestra: 'Error al procesar la imagen. Contacte al 
administrador.'
• CA-02.6: No es posible enviar una segunda solicitud mientras una 
predicción está en curso.
Requerimientos 
no funcionales
• Rendimiento: tiempo de inferencia del modelo ≤ 5 s en hardware 
con GPU; ≤ 15 s en CPU.
• Fiabilidad: el backend retorna códigos HTTP semánticamente 
correctos (200, 400, 500).
• Seguridad: la imagen se transmite sobre HTTPS; no se persiste en 
el servidor tras la inferencia.
• Usabilidad: el indicador de progreso es visible y no bloquea 
completamente la interfaz.
Estado Pendiente
Observaciones Si el backend utiliza GPU, el tiempo de respuesta puede reducirse 
significativamente. Considerar timeout de 15 s para entornos con 
CPU.

---

## Página 6


Campo Descripción
Título del 
Requerimiento
Visualización de resultados de clasificación
Código de 
identificación
RF-03
Tipo de 
requerimiento
Funcional
Versión 1.0
Fuente Historia de Usuario 3
Descripción El sistema debe presentar los resultados del análisis de forma clara e 
inmediata: clase predicha (estadio de Alzheimer detectado), 
porcentaje de confianza de la predicción y las probabilidades 
individuales de las cuatro clases posibles (Non Demented, Very Mild 
Dementia, Mild Dementia, Moderate Dementia). Los datos se 
muestran en una tabla y/o gráfico de barras, con formato legible y sin 
tecnicismos innecesarios.
Justificación La visualización clara y sin ambigüedad de los resultados es esencial 
para que el especialista pueda interpretar correctamente el 
diagnóstico asistido y tomar decisiones clínicas informadas. La 
presentación de las cuatro probabilidades permite evaluar el grado de 
certeza del modelo.
Precondiciones• RF-02 completado con éxito: el backend retornó una respuesta 
JSON válida con los resultados.
Restricciones • Siempre se muestran las probabilidades de las cuatro clases, sin 
excepción.
• Los porcentajes se presentan con exactamente 2 decimales.
• La visualización debe ser legible tanto en pantallas de escritorio 
como en tablets.
Prioridad Alta/Must
Dificultad Facil
Dependencia RF-02
Actores Médico tratante, Neurorradiólogo
Entradas • Objeto JSON retornado por el backend con: clase predicha, nivel de 
confianza y probabilidades por clase.
Proceso • El frontend recibe la respuesta JSON del backend.
• El sistema extrae la clase predicha, el porcentaje de confianza y las

---

## Página 7


Campo Descripción
probabilidades de las 4 clases.
• El sistema renderiza los datos en una tabla con las columnas: 
Clase, Probabilidad (%).
• El sistema resalta visualmente la clase predicha (fila destacada o 
indicador gráfico).
• Opcionalmente, el sistema renderiza un gráfico de barras horizontal 
con las 4 probabilidades.
• Si la respuesta JSON está incompleta, muestra: 'Resultados 
parciales. Algunos datos no están disponibles.'
Salida • Tabla con la clase predicha destacada y las probabilidades de las 4 
clases con 2 decimales.
• Gráfico de barras horizontal (opcional pero recomendado).
• Indicador visual del nivel de confianza.
Postcondiciones• Los resultados permanecen visibles en pantalla hasta que el usuario 
cargue una nueva imagen o navegue fuera de la sección.
Criterios de 
aceptación
• CA-03.1: La clase detectada se muestra en texto legible (ej. 'Mild 
Dementia') con su nivel de confianza (ej. '87.50%').
• CA-03.2: Se muestran las probabilidades individuales de las 4 
clases con exactamente 2 decimales.
• CA-03.3: La clase predicha está visualmente diferenciada del resto 
(color, negrita o ícono).
• CA-03.4: La visualización completa aparece en menos de 1 
segundo tras recibir la respuesta del backend.
• CA-03.5: Si los datos del JSON están incompletos, se muestra un 
mensaje de advertencia sin bloquear la interfaz.
Requerimientos 
no funcionales
• Usabilidad: los nombres de las clases se muestran en inglés con 
una nota explicativa en español.
• Rendimiento: el renderizado de la tabla y/o gráfico ocurre en < 1 s.
• Accesibilidad: el contraste de colores cumple con WCAG 2.1 nivel 
AA.

---

## Página 8


Campo Descripción
• Internacionalización: los valores numéricos usan punto como 
separador decimal, consistente con el formato del JSON.
Estado Pendiente
Observaciones Considerar colas de procesamiento
Campo Descripción
Título del 
Requerimiento
Generación y visualización del mapa de calor Grad-CAM
Código de 
identificación
RF-04
Tipo de 
requerimiento
Funcional
Descripción El sistema debe generar y mostrar un mapa de calor Grad-CAM 
superpuesto sobre la imagen MRI original, resaltando las regiones 
cerebrales que mayor influencia tuvieron en la decisión del modelo. 
El mapa se genera sobre la última capa CBAM del modelo 
DenseNet121, usando la paleta JET con transparencia 0.4. El usuario 
puede alternar entre la imagen original y la imagen con el mapa 
superpuesto.
Justificación La interpretabilidad del modelo es un requisito crítico en aplicaciones 
clínicas de IA. Grad-CAM permite a los especialistas verificar que el 
modelo toma decisiones basándose en regiones anatómicamente 
relevantes, aumentando la confianza clínica y permitiendo auditar el 
comportamiento del modelo.
Precondiciones• RF-02 completado: la predicción fue ejecutada exitosamente.
Restricciones • La imagen MRI original está disponible en memoria del frontend.
Prioridad Alta
Dificultad Dificil
Dependencia RF-03
Actores Neurorradiólogo, Investigador clínico, Médico especialista
Entradas • Mapa de activación codificado en base64 retornado por el backend 
en la respuesta JSON.
• Imagen MRI original disponible en memoria del frontend.
Proceso • El frontend recibe el mapa de activación desde la respuesta JSON 
del backend.
• El sistema decodifica el mapa de activación y lo redimensiona para

---

## Página 9


Campo Descripción
coincidir con las dimensiones de la imagen original.
• El sistema aplica la paleta de colores JET al mapa de activación.
• El sistema superpone el mapa de calor sobre la imagen original con 
transparencia 0.4.
• El sistema renderiza la imagen resultante junto a los resultados de 
RF-03.
• El sistema habilita un control de alternancia (toggle) entre la imagen 
original y la imagen con el mapa superpuesto.
• Si ocurre un error en la generación, muestra: 'No se pudo generar el 
mapa explicativo.'
Salida • Imagen MRI con mapa de calor Grad-CAM superpuesto (paleta JET, 
transparencia 0.4).
• Control de alternancia entre imagen original y mapa Grad-CAM.
• Mensaje de error en caso de fallo.
Postcondiciones• El mapa Grad-CAM queda disponible para descarga (RF-08) y para 
inclusión en el informe PDF (RF-09).
Criterios de 
aceptación
• CA-04.1: El mapa de calor se genera automáticamente tras la 
predicción, sin acción adicional del usuario.
• CA-04.2: La superposición es visualmente correcta: el mapa cubre 
la imagen original con transparencia 0.4.
• CA-04.3: Las regiones de mayor relevancia se representan en 
colores cálidos (rojo/amarillo) y las de menor relevancia en colores 
fríos (azul).
• CA-04.4: El control de alternancia funciona correctamente en 
ambas direcciones (original ↔ Grad-CAM).
• CA-04.5: El mapa se renderiza en ≤ 3 segundos adicionales tras 
recibir la respuesta del backend.
• CA-04.6: Si el backend no retorna el mapa de activación, se 
muestra el mensaje de error correspondiente sin bloquear la 
visualización de resultados.

---

## Página 10


Campo Descripción
Requerimientos 
no funcionales
• Usabilidad: el toggle entre imagen original y mapa es intuitivo y está 
claramente etiquetado.
• Rendimiento: el procesamiento de superposición en el frontend 
ocurre en < 1 s.
• Fiabilidad: un error en la generación del mapa no impide visualizar 
los resultados de clasificación (RF-03).
• Rendimiento (backend): la generación del mapa puede beneficiarse 
de aceleración por GPU.
Estado Pendiente
Observaciones El backend puede retornar el mapa Grad-CAM como imagen PNG 
codificada en base64 dentro del JSON, o como endpoint separado 
GET /gradcam/{id}. Definir en contrato de API.
Campo Descripción
Título del 
Requerimiento
Visualización de métricas de rendimiento del modelo
Código RF-05
Tipo Funcional
Descripción El sistema debe mostrar en la sección 'Modelo' las métricas de 
rendimiento del modelo DenseNet121 + CBAM obtenidas sobre el 
conjunto de prueba: Accuracy, Precision (por clase y macro-
average), Recall (por clase y macro-average) y F1-Score (por clase y 
macro-average). Los valores se muestran en formato tabular con 2 
decimales.
Justificación Transparentar el rendimiento del modelo ante los usuarios clínicos es 
esencial para establecer confianza y permitir una evaluación crítica 
del sistema. El especialista debe poder juzgar la validez científica del 
diagnóstico asistido antes de incorporarlo a su práctica clínica.
Precondiciones • Las métricas del modelo han sido calculadas sobre el conjunto de 
prueba y están almacenadas en el sistema (archivo de configuración 
o endpoint del backend).
Restricciones • Se muestran únicamente las métricas del conjunto de prueba; no 
del conjunto de entrenamiento ni validación.
• Los valores se presentan con exactamente 2 decimales en formato 
porcentual.

---

## Página 11


Campo Descripción
• Las métricas son estáticas (precalculadas); no se recalculan en 
tiempo real.
Prioridad Alta
Dificultad Facil
Dependencia Investigador clínico, Médico especialista
Actores • Métricas precalculadas almacenadas en el sistema: Accuracy, 
Precision, Recall, F1-Score por clase y macro-average.
Entradas • El usuario navega a la sección 'Modelo' mediante la barra de 
navegación (RF-06).
• El sistema recupera las métricas del almacenamiento local (archivo 
JSON o endpoint GET /metrics).
• El sistema renderiza una tabla con las columnas: Clase | Precision 
(%) | Recall (%) | F1-Score (%).
• El sistema muestra una fila adicional con los valores macro-
average.
• El sistema muestra el Accuracy global del modelo.
• Si los datos no están disponibles, muestra: 'Métricas no disponibles 
en este momento.'
Proceso • Tabla de métricas por clase y macro-average.
• Valor de Accuracy global del modelo.
• Mensaje informativo si los datos no están disponibles.
Salida • Las métricas quedan visibles para el usuario hasta que navegue a 
otra sección.
Postcondiciones• CA-05.1: La tabla muestra Precision, Recall y F1-Score para cada 
una de las 4 clases.
• CA-05.2: La tabla incluye una fila de macro-average.
• CA-05.3: El Accuracy global del modelo se muestra de forma 
prominente (ej. 99.38%).
• CA-05.4: Todos los valores se muestran con exactamente 2 
decimales en formato porcentual.

---

## Página 12


Campo Descripción
• CA-05.5: La sección se carga completamente en menos de 1 
segundo.
Criterios de 
aceptación
• Usabilidad: cada métrica incluye un tooltip o leyenda breve con su 
definición.
• Rendimiento: carga en < 1 s desde almacenamiento local o cache.
• Fiabilidad: si el endpoint de métricas falla, el sistema muestra datos 
desde un archivo estático de respaldo.
• Trazabilidad: la sección indica la fecha de evaluación del modelo y 
el dataset utilizado.
Requerimientos 
no funcionales
• Usabilidad: cada métrica incluye un tooltip o leyenda breve con su 
definición.
• Rendimiento: carga en < 1 s desde almacenamiento local o cache.
• Fiabilidad: si el endpoint de métricas falla, el sistema muestra datos 
desde un archivo estático de respaldo.
• Trazabilidad: la sección indica la fecha de evaluación del modelo y 
el dataset utilizado.
Estado Pendiente
Observaciones Valores de referencia del modelo DenseNet121 + CBAM sobre el 
conjunto de prueba: Accuracy 99.38%, Precision macro 99.55%. 
Estos valores deben actualizarse si el modelo es reentrenado.
Campo Descripción
Título del 
Requerimiento
Navegación entre secciones del sistema
Código de 
identificación
RF-06
Tipo de 
requerimiento
Funcional
Versión 1.0
Fuente Historia de Usuario 6
Descripción El sistema debe proporcionar una barra de navegación superior 
persistente que permita al usuario desplazarse entre las secciones 
'Inicio' (carga, predicción y resultados) y 'Modelo' (métricas de 
rendimiento). La sección activa debe estar visualmente diferenciada.

---

## Página 13


Campo Descripción
Al regresar a 'Inicio', el último resultado de predicción debe 
permanecer visible si existe.
Justificación Una navegación clara y persistente es fundamental para la 
usabilidad del sistema. La pérdida de resultados al navegar entre 
secciones genera frustración y obliga al usuario a repetir análisis, 
impactando negativamente el flujo de trabajo clínico.
Precondiciones • El sistema está cargado correctamente en el navegador del usuario.
Restricciones • La barra de navegación debe estar visible en todas las 
páginas/secciones de la aplicación.
• La sección activa debe estar resaltada visualmente en todo 
momento.
• La navegación no debe causar pérdida del último resultado de 
predicción.
Prioridad Alta/Must
Dificultad Facil
Dependencia Ninguna
Actores Cualquier usuario del sistema
Entradas • Clic del usuario sobre un ítem de la barra de navegación ('Inicio' o 
'Modelo').
Proceso • El sistema renderiza la barra de navegación superior en todas las 
vistas.
• El usuario hace clic en un ítem de la barra de navegación.
• El sistema actualiza la vista activa y resalta el ítem correspondiente.
• Si el usuario regresa a 'Inicio' y existe un resultado de predicción en 
el estado de la aplicación, lo muestra.
• Si ocurre un error al cargar una sección, muestra: 'Error al cargar la 
página. Intente nuevamente.'
Salida • Vista correspondiente a la sección seleccionada.
• Barra de navegación con el ítem activo resaltado.
• Persistencia del último resultado de predicción al regresar a 'Inicio'.
Postcondiciones• El usuario se encuentra en la sección solicitada con el estado de la 
aplicación intacto.

---

## Página 14


Campo Descripción
Criterios de 
aceptación
• CA-06.1: La barra de navegación es visible en todas las secciones 
de la aplicación.
• CA-06.2: La barra incluye al menos los ítems 'Inicio' y 'Modelo'.
• CA-06.3: El ítem de la sección activa está visualmente diferenciado 
(color, subrayado o indicador).
• CA-06.4: El cambio de sección se completa en menos de 2 
segundos.
• CA-06.5: Al regresar a 'Inicio' desde 'Modelo', el último resultado de 
predicción (si existe) permanece visible.
Requerimientos 
no funcionales
• Usabilidad: la barra de navegación es responsive y funciona 
correctamente en pantallas de 768px o más de ancho.
• Rendimiento: el cambio de sección no requiere recarga completa 
de la página (SPA o navegación por estado).
• Accesibilidad: los ítems de navegación son accesibles mediante 
teclado y tienen atributos aria-label correctos.
Estado Pendiente
Observaciones
Campo Descripción
Título del 
Requerimiento
Gestión centralizada de errores del sistema
Código de 
identificación
RF-07
Tipo de 
requerimiento
Funcional
Versión 1.0
Fuente Historia de Usuario 7
Descripción El sistema debe detectar y clasificar los errores que ocurran durante 
su operación (errores de validación de entrada, errores de conexión 
con el backend y errores internos del servidor) y mostrar al usuario 
un mensaje descriptivo, específico y en español, con una sugerencia 
de acción cuando sea posible. Los mensajes no deben exponer 
detalles técnicos internos (stack traces, códigos HTTP crudos).

---

## Página 15


Campo Descripción
Justificación Un manejo de errores claro y consistente reduce la frustración del 
usuario, minimiza las consultas de soporte técnico y evita que el 
sistema quede en un estado irrecuperable ante fallos parciales. En un 
contexto clínico, la ambigüedad ante un error puede llevar a 
decisiones equivocadas.
Precondiciones• El sistema está en ejecución y el usuario está interactuando con 
alguna funcionalidad.
Restricciones • Todos los mensajes de error deben estar en español claro y sin 
tecnicismos.
• No se debe mostrar el código HTTP crudo ni el stack trace al 
usuario final.
• Los errores no deben bloquear la aplicación permanentemente; el 
usuario debe poder recuperarse.
• Los mensajes de error deben poder cerrarse manualmente.
Prioridad Alta/Must
Dificultad Nominal
Dependencia RF-01, RF-02, RF-04, RF-05, RF-06, RF-08, RF-09
Actores Cualquier usuario del sistema
Entradas • Evento de error generado por cualquier módulo del sistema 
(validación, red, servidor).
Proceso • El sistema detecta el error (excepción capturada, respuesta HTTP 
con código de error, timeout).
• El sistema clasifica el error según su tipo: validación, conexión o 
interno.
• El sistema muestra un componente de notificación no bloqueante 
con el mensaje correspondiente y la sugerencia de acción.
• El usuario puede cerrar el mensaje manualmente o el sistema lo 
oculta automáticamente tras un tiempo configurable.
• Para errores críticos (ej. fallo total de carga), se muestra un 
mensaje reforzado con la opción 'Recargar página'.
Salida • Notificación visible con mensaje de error específico en español.
• Sugerencia de acción correctiva.

---

## Página 16


Campo Descripción
• Opción de cierre manual del mensaje.
• Para errores críticos: botón 'Recargar página'.
Postcondiciones• El usuario está informado del error y puede continuar usando la 
aplicación o tomar la acción sugerida.
Criterios de 
aceptación
• CA-07.1: Cada tipo de error muestra un mensaje diferenciado (no 
genérico).
• CA-07.2: Todos los mensajes están en español y son comprensibles 
por un usuario no técnico.
• CA-07.3: Los mensajes incluyen una sugerencia de acción (ej. 
'Intente nuevamente' o 'Contacte al administrador').
• CA-07.4: Los mensajes de error pueden cerrarse manualmente 
mediante un botón visible.
• CA-07.5: Un error en un módulo (ej. Grad-CAM) no bloquea la 
funcionalidad de otros módulos (ej. visualización de resultados).
• CA-07.6: No se muestra al usuario ningún stack trace, código HTTP 
crudo ni mensaje de error interno del sistema.
Requerimientos 
no funcionales
• Fiabilidad: el sistema maneja excepciones no controladas sin 
colapsar la aplicación completa.
• Usabilidad: los mensajes de error tienen contraste suficiente y son 
visibles sin desplazar la vista del usuario.
• Trazabilidad: los errores se registran en el log del sistema (sin datos 
sensibles) para diagnóstico posterior.
• Seguridad: los mensajes no revelan información de la arquitectura 
interna del sistema.
Estado Pendiente
Observaciones Clasificación de errores: (1) Validación: archivo inválido, campos 
faltantes. (2) Conexión: timeout, red no disponible, CORS. (3) Interno: 
error 500, excepción no controlada en el backend.
Campo Descripción
Título del 
Requerimiento
Descarga del mapa de calor Grad-CAM como imagen PNG

---

## Página 17


Campo Descripción
Código de 
identificación
RF-08
Tipo de 
requerimiento
Funcional
Versión 1.0
Fuente Historia de Usuario 8
Descripción El sistema debe permitir al usuario descargar la imagen resultante de 
la superposición del mapa de calor Grad-CAM sobre la imagen MRI 
original, en formato PNG. La imagen descargada debe ser idéntica a 
la visualizada en pantalla. La descarga se inicia desde un botón 
visible junto al mapa Grad-CAM.
Justificación El mapa Grad-CAM es evidencia visual del proceso de decisión del 
modelo. Permitir su descarga facilita su inclusión en informes 
clínicos, historias médicas electrónicas o publicaciones científicas, 
extendiendo la utilidad del sistema más allá de la sesión web.
Precondiciones • RF-04 completado: el mapa Grad-CAM ha sido generado y está 
visible en la interfaz.
• El navegador del usuario admite la descarga de archivos 
(funcionalidad estándar).
Restricciones • El formato de descarga es exclusivamente PNG.
• La imagen descargada debe ser pixel-perfect respecto a la 
visualizada en pantalla.
• La descarga debe iniciarse en menos de 2 segundos tras la acción 
del usuario.
Prioridad Alta/Must
Dificultad Fácil
Dependencia RF-04
Actores Médico especialista, Neurorradiólogo
Entradas • Clic del usuario sobre el botón 'Descargar mapa'.
• Imagen compuesta (MRI + Grad-CAM) disponible en memoria del 
frontend.
Proceso • El usuario presiona el botón 'Descargar mapa'.
• El sistema toma la imagen compuesta (MRI + Grad-CAM 
superpuesto) desde el estado del frontend.

---

## Página 18


Campo Descripción
• El sistema genera un archivo PNG a partir de la imagen compuesta.
• El navegador inicia la descarga con el nombre de archivo sugerido 
(ej. 'gradcam_resultado.png').
• Si la generación del archivo falla, se muestra: 'No se pudo generar 
la descarga. Intente nuevamente.'
Salida • Archivo PNG descargado en el dispositivo del usuario.
• Mensaje de error si la descarga falla.
Postcondiciones• El archivo PNG queda almacenado localmente en el dispositivo del 
usuario. El estado de la aplicación no se altera.
Criterios de 
aceptación
• CA-08.1: El botón 'Descargar mapa' es visible cuando el mapa 
Grad-CAM está disponible.
• CA-08.2: El archivo descargado está en formato PNG.
• CA-08.3: La imagen descargada es visualmente idéntica al mapa 
Grad-CAM mostrado en pantalla.
• CA-08.4: La descarga se inicia en menos de 2 segundos tras el clic 
del usuario.
• CA-08.5: La funcionalidad opera correctamente en Chrome, Edge y 
Firefox (últimas versiones estables).
Requerimientos 
no funcionales
• Usabilidad: el botón de descarga está claramente identificado con 
ícono y etiqueta de texto.
• Compatibilidad: la descarga usa la API nativa del navegador 
(elemento  con atributo download) sin dependencias externas.
• Rendimiento: la generación del PNG desde canvas ocurre en el hilo 
principal sin bloquear la interfaz.
Estado Pendiente
Observaciones La imagen puede generarse desde un elemento HTML Canvas que 
ya contenga la superposición, usando canvas.toBlob() o 
canvas.toDataURL() con tipo 'image/png'.

---

## Página 19


Campo Descripción
Título del 
Requerimiento
Generación y descarga de informe de resultados en PDF
Código de 
identificación
RF-09
Tipo de 
requerimiento
Funcional
Versión 1.0
Fuente Médico tratante / Especialista clínico
Descripción El sistema debe permitir al usuario generar y descargar un informe 
en formato PDF que consolide los resultados del análisis. El informe 
debe incluir: fecha y hora del análisis, imagen MRI original, clase 
predicha, nivel de confianza, tabla de probabilidades por clase y el 
mapa de calor Grad-CAM. El tamaño máximo del PDF es 5 MB.
Justificación Un informe descargable y portable permite integrar el diagnóstico 
asistido en los flujos de trabajo clínicos existentes: historia clínica 
electrónica, interconsultas, presentaciones de caso o archivos de 
investigación. El PDF es el formato estándar para documentación 
clínica formal.
Precondiciones• RF-02 completado: la predicción fue ejecutada exitosamente.
• RF-04 completado: el mapa Grad-CAM está disponible.
• Todos los datos necesarios están en el estado de la aplicación.
Restricciones • El PDF debe incluir obligatoriamente: fecha/hora, imagen original, 
clase predicha, confianza, probabilidades por clase y mapa Grad-
CAM.
• Tamaño máximo del archivo PDF: 5 MB.
• La descarga debe iniciarse en menos de 3 segundos tras la 
solicitud del usuario.
• El PDF debe ser legible en cualquier visor de PDF estándar (Adobe 
Acrobat, navegadores modernos).
Prioridad Alta/Must
Dificultad Nominal
Dependencia RF-02, RF-04
Actores Médico tratante, Especialista clínico

---

## Página 20


Campo Descripción
Entradas • Datos del estado de la aplicación: fecha/hora del análisis, imagen 
original, clase predicha, nivel de confianza, probabilidades por clase, 
imagen Grad-CAM.
Proceso • El usuario presiona el botón 'Descargar informe'.
• El sistema recopila todos los datos necesarios del estado de la 
aplicación.
• El sistema genera el documento PDF con la estructura definida: 
encabezado, imagen original, resultados tabulares, mapa Grad-CAM 
y pie de página.
• El sistema verifica que el tamaño del PDF no supere 5 MB.
• El navegador inicia la descarga con el nombre sugerido (ej. 
'informe_alzheimer_YYYY-MM-DD.pdf').
• Si la generación falla, muestra: 'No se pudo generar el informe. 
Intente nuevamente.'
Salida • Archivo PDF descargado en el dispositivo del usuario.
• Mensaje de error si la generación falla.
Postcondiciones• El archivo PDF queda almacenado localmente en el dispositivo del 
usuario. El estado de la aplicación no se altera.
Criterios de 
aceptación
• CA-09.1: El PDF incluye todos los elementos requeridos: 
fecha/hora, imagen original, clase predicha, confianza, 
probabilidades por clase y mapa Grad-CAM.
• CA-09.2: El PDF es legible y bien estructurado en Adobe Acrobat y 
en Chrome/Edge.
• CA-09.3: La descarga se inicia en menos de 3 segundos.
• CA-09.4: El tamaño del PDF no supera 5 MB.
• CA-09.5: La funcionalidad opera correctamente en los navegadores 
Chrome, Edge y Firefox (últimas versiones estables).
Requerimientos 
no funcionales
• Usabilidad: el PDF incluye un pie de página con texto: 'Este informe 
es un apoyo diagnóstico. No reemplaza el criterio médico.'
• Rendimiento: la generación del PDF ocurre en el cliente (ej. jsPDF 
+ html2canvas) sin enviar datos al servidor.

---

## Página 21


Campo Descripción
• Privacidad: la imagen MRI del paciente no se envía a servidores 
externos durante la generación del PDF.
• Compatibilidad: el PDF cumple con el estándar PDF/A para 
archivado de largo plazo.
Estado Pendiente
Observaciones Usar librería jsPDF con html2canvas para la generación en cliente. Si 
las imágenes son de alta resolución, comprimir antes de incluirlas 
para no superar el límite de 5 MB.
Campo Descripción
Título del 
Requerimiento
Integración frontend–backend mediante API REST
Código de 
identificación
RF-10
Tipo de 
requerimiento
Funcional
Versión 1.0
Fuente Historia de Usuario 10
Descripción El sistema debe garantizar la comunicación bidireccional entre el 
frontend y el backend mediante una API REST sobre HTTPS. El 
frontend envía la imagen MRI al endpoint POST /predict y recibe una 
respuesta JSON estructurada con la clase predicha, las 
probabilidades por clase y el mapa de activación Grad-CAM. La API 
debe manejar CORS correctamente y retornar códigos HTTP 
semánticamente adecuados.
Justificación La integración frontend–backend es el núcleo técnico que habilita 
todas las funcionalidades de diagnóstico asistido. Un contrato de API 
bien definido, con manejo correcto de errores, CORS y formatos de 
datos, garantiza la interoperabilidad del sistema y facilita el 
mantenimiento y la escalabilidad futura.
Precondiciones• El backend está desplegado y accesible desde la red donde opera el 
frontend.
• El certificado SSL/TLS está configurado para comunicación HTTPS.
• Las variables de entorno con la URL base del backend están 
configuradas en el frontend.
Restricciones • Toda comunicación debe realizarse sobre HTTPS (HTTP sin cifrado 
no está permitido en producción).

---

## Página 22


Campo Descripción
• El formato de intercambio de datos es JSON para respuestas; 
multipart/form-data para el envío de la imagen.
• El backend debe tener CORS habilitado para el dominio del frontend.
• El tiempo de respuesta del endpoint /predict no debe superar 5 
segundos.
• Los errores del backend deben retornar códigos HTTP 
semánticamente correctos (400 para errores de cliente, 500 para 
errores de servidor).
Prioridad Alta/Must
Dificultad Normal
Dependencia RF-01, RF-02
Actores Sistema frontend, Sistema backend
Entradas • Solicitud HTTP POST al endpoint /predict con la imagen MRI como 
archivo (multipart/form-data).
Proceso • El frontend construye una solicitud HTTP POST con la imagen MRI 
adjunta como multipart/form-data.
• El frontend envía la solicitud al endpoint configurado (ej. 
https://api.dominio.com/predict).
• El backend valida el formato y tamaño del archivo recibido.
• El backend ejecuta el preprocesamiento y la inferencia con el 
modelo DenseNet121 + CBAM.
• El backend genera el mapa Grad-CAM y serializa todos los 
resultados en un objeto JSON.
• El backend retorna la respuesta con HTTP 200 y el JSON 
estructurado.
• El frontend deserializa el JSON y distribuye los datos a los módulos 
RF-03 y RF-04.
• Ante errores, el backend retorna HTTP 400 (entrada inválida) o 
HTTP 500 (error interno) con un mensaje descriptivo en el cuerpo 
JSON.

---

## Página 23


Campo Descripción
Salida • Respuesta HTTP 200 con JSON: {predicted_class, confidence, 
probabilities: {Non Demented, Very Mild Dementia, Mild Dementia, 
Moderate Dementia}, gradcam_image (base64)}.
• Respuesta HTTP 400 o 500 con JSON de error: {error_code, 
message} ante fallos.
Postcondiciones• Los datos de la respuesta están disponibles en el estado del 
frontend para su renderizado por RF-03 y RF-04.
Criterios de 
aceptación
• CA-10.1: El frontend envía correctamente la imagen al endpoint 
/predict mediante POST multipart/form-data.
• CA-10.2: El backend retorna una respuesta JSON con todos los 
campos requeridos (predicted_class, confidence, probabilities, 
gradcam_image).
• CA-10.3: CORS está configurado sin errores para el dominio del 
frontend.
• CA-10.4: El tiempo de respuesta total del endpoint es ≤ 5 segundos 
en condiciones normales de red.
• CA-10.5: Los errores retornan el código HTTP correcto (400 para 
entrada inválida, 500 para error interno).
• CA-10.6: La comunicación se realiza exclusivamente sobre HTTPS 
en el entorno de producción.
Requerimientos 
no funcionales
• Seguridad: la imagen no se almacena en el servidor tras completar 
la inferencia.
• Rendimiento: el endpoint /predict responde en ≤ 5 s (GPU) o ≤ 15 s 
(CPU).
• Mantenibilidad: la API está documentada con OpenAPI/Swagger y el 
contrato de respuesta es versionado.
• Fiabilidad: el backend implementa manejo de excepciones para 
evitar respuestas HTTP 500 ante entradas malformadas.
• Escalabilidad: la arquitectura permite agregar endpoints adicionales 
(ej. /metrics, /gradcam) sin romper el contrato existente.
Estado Pendiente
Observaciones Estructura JSON de respuesta esperada: { 'predicted_class': 'Mild 
Dementia', 'confidence': 87.50, 'probabilities': { 'Non Demented': 5.20,

---

## Página 24


Campo Descripción
'Very Mild Dementia': 4.30, 'Mild Dementia': 87.50, 'Moderate 
Dementia': 3.00 }, 'gradcam_image': '<base64_string>' }