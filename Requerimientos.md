# Requerimientos del Sistema de Diagnóstico de Alzheimer por IA

---

## REQUERIMIENTOS FUNCIONALES (RF)

### RF-01: Subida y validación de imágenes MRI
**Código de identificación:** RF-01  
**Tipo de requerimiento:** Funcional  
**Versión:** 1.0  
**Fuente:** Historia de Usuario 1  
**Prioridad:** Alta/Must  
**Dificultad:** Nominal  
**Actores:** Médico especialista  

**Descripción:** El sistema debe permitir al usuario cargar imágenes de resonancia magnética cerebral (MRI) en formatos JPG, PNG y DICOM, validando tanto el formato como la coherencia del contenido antes de permitir su uso en el análisis.

**Justificación:** Asegurar que las imágenes utilizadas sean válidas y adecuadas para el procesamiento por modelos de inteligencia artificial, evitando resultados incorrectos.

**Precondiciones:**
- Usuario dentro del sistema
- Interfaz cargada correctamente

**Restricciones:**
- Formatos permitidos: JPG, PNG, DICOM
- Tamaño máximo: 10 MB
- Solo imágenes médicas (no fotos comunes)

**Dependencia:** Módulo de validación y carga de archivos

**Entradas:** Archivo de imagen MRI

**Proceso:**
1. Usuario selecciona archivo
2. Sistema valida formato
3. Sistema valida tamaño
4. Sistema analiza contenido básico
5. Se muestra vista previa

**Salida:** Imagen válida cargada en el sistema

**Postcondiciones:** Imagen disponible para análisis posterior

**Criterios de aceptación:**
1. El sistema acepta únicamente archivos con extensiones .jpg, .png o .dcm (rechaza otros con mensaje claro).
2. Si el archivo supera los 10 MB, se muestra mensaje: "El archivo excede el tamaño permitido".
3. Se genera una vista previa visible en menos de 2 segundos tras la carga.
4. Si el archivo no corresponde a una imagen válida, el sistema muestra: "Archivo no válido o corrupto".
5. El sistema no permite continuar al análisis sin una imagen válida cargada.

**Requerimientos no funcionales:** Usabilidad, rendimiento  
**Estado:** Pendiente  
**Observaciones:** Se puede mejorar con validación automática de estructuras cerebrales

---

### RF-02: Selección de modelo de inteligencia artificial
**Código de identificación:** RF-02  
**Tipo de requerimiento:** Funcional  
**Versión:** 1.0  
**Fuente:** Historia de Usuario 2  
**Prioridad:** Alta/Must  
**Dificultad:** Nominal  
**Actores:** Investigador médico  

**Descripción:** El sistema debe permitir al usuario seleccionar entre diferentes modelos de deep learning disponibles para ejecutar el análisis.

**Justificación:** Permite comparar resultados y seleccionar el modelo más adecuado según el contexto clínico.

**Precondiciones:** Modelos previamente cargados en el sistema

**Restricciones:** Solo modelos previamente entrenados y registrados

**Dependencia:** Base de datos de modelos

**Entradas:** Selección del modelo

**Proceso:**
1. Mostrar lista de modelos
2. Mostrar métricas
3. Usuario selecciona

**Salida:** Modelo seleccionado

**Postcondiciones:** Modelo listo para predicción

**Criterios de aceptación:**
1. El sistema muestra al menos 4 modelos disponibles.
2. Cada modelo muestra métricas: accuracy, precision y recall.
3. El modelo seleccionado se resalta visualmente (color o etiqueta "activo").
4. El usuario puede cambiar de modelo antes de ejecutar la predicción.
5. Si no hay modelo seleccionado, el sistema selecciona uno por defecto.

**Requerimientos no funcionales:** Usabilidad  
**Estado:** Pendiente  
**Observaciones:** Se recomienda agregar descripción técnica por modelo

---

### RF-03: Ejecución de predicción
**Código de identificación:** RF-03  
**Tipo de requerimiento:** Funcional  
**Versión:** 1.0  
**Fuente:** Historia de Usuario 3  
**Prioridad:** Alta/Must  
**Dificultad:** Nominal  
**Actores:** Profesional de salud  

**Descripción:** El sistema debe permitir ejecutar el análisis de la imagen MRI mediante el modelo DenseNet121 predefinido.

**Justificación:** Permite obtener diagnósticos asistidos automáticamente con un modelo optimizado.

**Precondiciones:**
- Imagen cargada
- Modelo DenseNet121 disponible

**Restricciones:** Conexión activa al backend, uso exclusivo del modelo DenseNet121

**Dependencia:** API /predict

**Entradas:** Imagen MRI

**Proceso:**
1. Usuario presiona botón
2. Validación de imagen
3. Envío al backend con modelo fijo
4. Procesamiento

**Salida:** Resultado de predicción

**Postcondiciones:** Resultado almacenado temporalmente

**Criterios de aceptación:**
1. El botón "Predecir" permanece deshabilitado si no hay imagen cargada.
2. Al hacer clic, aparece un indicador de carga visible.
3. El tiempo de respuesta no supera los 5 segundos en condiciones normales.
4. Si falla la conexión, se muestra: "Error de conexión con el servidor".
5. El sistema evita múltiples envíos simultáneos del mismo análisis.
6. El sistema utiliza automáticamente el modelo DenseNet121 sin opción de selección.

**Requerimientos no funcionales:** Rendimiento, fiabilidad  
**Estado:** Pendiente  
**Observaciones:** Considerar colas de procesamiento

---

### RF-04: Visualización de resultados
**Código de identificación:** RF-04  
**Tipo de requerimiento:** Funcional  
**Prioridad:** Alta  
**Dificultad:** Fácil  
**Actores:** Médico  

**Descripción:** Mostrar los resultados del análisis incluyendo clase predicha, nivel de confianza y probabilidades por clase.

**Justificación:** Facilita la interpretación médica del resultado.

**Precondiciones:** Predicción realizada

**Restricciones:** Datos provenientes del backend

**Dependencia:** RF-03

**Entradas:** Resultado del modelo

**Proceso:**
1. Recepción
2. Formato
3. Visualización

**Salida:** Resultados interpretables

**Postcondiciones:** Datos visibles al usuario

**Criterios de aceptación:**
1. Se muestra la clase detectada (ej: "Mild Dementia").
2. Se muestra el porcentaje de confianza (ej: 87.5%).
3. Se muestran todas las clases con sus probabilidades.
4. Los datos se presentan en formato legible (tabla o gráfico).
5. La información aparece en menos de 1 segundo tras recibir respuesta.

**Requerimientos no funcionales:** Usabilidad  
**Estado:** Pendiente  
**Observaciones:** Uso de gráficos mejora interpretación

---

### RF-05: Visualización Grad-CAM
**Código:** RF-05  
**Tipo:** Funcional  
**Prioridad:** Alta  
**Dificultad:** Difícil  
**Actores:** Neurorradiólogo  

**Descripción:** Mostrar un mapa de calor que indique las regiones relevantes en la imagen MRI utilizadas por el modelo.

**Justificación:** Permite interpretar decisiones del modelo.

**Precondiciones:** Predicción completada

**Restricciones:** Compatible con modelos usados

**Dependencia:** Modelo IA

**Entradas:** Imagen + mapa

**Proceso:** Generación y superposición

**Salida:** Imagen interpretativa

**Postcondiciones:** Visualización activa

**Criterios de aceptación:**
1. El mapa se genera automáticamente tras la predicción.
2. Se muestra superpuesto a la imagen original.
3. Las zonas de mayor relevancia se destacan en colores (rojo/amarillo).
4. El usuario puede alternar entre imagen original y Grad-CAM.
5. La generación no supera los 3 segundos adicionales.

**Requerimientos no funcionales:** Rendimiento  
**Estado:** Pendiente  
**Observaciones:** Puede requerir GPU

---

### RF-06: Visualización de métricas del modelo
**Código de identificación:** RF-06  
**Tipo de requerimiento:** Funcional  
**Versión:** 1.0  
**Fuente:** Historia de Usuario 6  
**Prioridad:** Alta/Must  
**Dificultad:** Nominal  
**Actores:** Investigador clínico  

**Descripción:** El sistema debe mostrar las métricas de rendimiento del modelo seleccionado, incluyendo accuracy, precision y recall, en un formato claro y comprensible.

**Justificación:** Permite evaluar la confiabilidad del modelo y respaldar decisiones clínicas o de investigación.

**Precondiciones:**
- Modelo seleccionado
- Métricas disponibles en el sistema

**Restricciones:**
- Las métricas deben corresponder al modelo seleccionado
- Datos provenientes de fuente validada

**Dependencia:** Base de datos o configuración de modelos

**Entradas:** Datos de métricas del modelo

**Proceso:**
1. Recuperar métricas del modelo
2. Formatear datos
3. Mostrar en tabla

**Salida:** Tabla con métricas del modelo

**Postcondiciones:** Métricas visibles para el usuario

**Criterios de aceptación:**
1. Se muestran al menos las métricas: accuracy, precision y recall.
2. Cada métrica se presenta en porcentaje con máximo 2 decimales (ej: 92.45%).
3. Las métricas corresponden exactamente al modelo seleccionado.
4. La información se muestra en formato tabular claro y alineado.
5. El tiempo de carga de métricas no supera 1 segundo.

**Requerimientos no funcionales:** Usabilidad, rendimiento  
**Estado:** Pendiente  
**Observaciones:** Se pueden incluir métricas adicionales como F1-score

---

### RF-07: Comparación de modelos de IA
**Código de identificación:** RF-07  
**Tipo de requerimiento:** Funcional  
**Versión:** 1.0  
**Fuente:** Historia de Usuario 7  
**Prioridad:** Alta/Must  
**Dificultad:** Nominal  
**Actores:** Científico de datos  

**Descripción:** El sistema debe mostrar una tabla comparativa con todos los modelos disponibles y sus métricas de rendimiento.

**Justificación:** Permite seleccionar el modelo más adecuado según precisión y desempeño.

**Precondiciones:** Modelos cargados en el sistema

**Restricciones:** Información debe estar actualizada

**Dependencia:** Base de datos de modelos

**Entradas:** Datos de modelos

**Proceso:**
1. Recuperar lista de modelos
2. Obtener métricas
3. Mostrar tabla comparativa

**Salida:** Tabla comparativa

**Postcondiciones:** Información visible

**Criterios de aceptación:**
1. Se muestran todos los modelos disponibles (mínimo 4).
2. Cada modelo incluye accuracy, precision y recall.
3. La tabla permite comparación visual clara entre modelos.
4. Los datos están ordenados (ej: por accuracy descendente).
5. El usuario puede regresar a la pantalla principal en máximo 1 clic.

**Requerimientos no funcionales:** Usabilidad  
**Estado:** Pendiente  
**Observaciones:** Se puede agregar filtro o búsqueda

---

### RF-08: Navegación del sistema
**Código de identificación:** RF-08  
**Tipo de requerimiento:** Funcional  
**Versión:** 1.0  
**Fuente:** Historia de Usuario 8  
**Prioridad:** Alta/Must  
**Dificultad:** Fácil  
**Actores:** Usuario del sistema  

**Descripción:** El sistema debe permitir navegar de forma intuitiva entre las diferentes páginas del sistema (inicio, análisis, modelos).

**Justificación:** Mejora la experiencia del usuario y evita confusión.

**Precondiciones:** Sistema cargado

**Restricciones:** Navegación basada en interfaz web

**Dependencia:** Frontend

**Entradas:** Acciones del usuario (clics)

**Proceso:**
1. Mostrar menú
2. Usuario selecciona opción
3. Sistema redirige

**Salida:** Página solicitada

**Postcondiciones:** Usuario ubicado en nueva vista

**Criterios de aceptación:**
1. Existe una barra de navegación visible en todo momento.
2. Incluye al menos opciones: "Inicio" y "Modelos".
3. La página activa se resalta visualmente.
4. El cambio de página ocurre en menos de 2 segundos.
5. No se pierde información crítica al navegar (ej: imagen cargada opcionalmente persistente).

**Requerimientos no funcionales:** Usabilidad  
**Estado:** Pendiente  
**Observaciones:** Puede incluir breadcrumbs

---

### RF-09: Manejo de errores del sistema
**Código de identificación:** RF-09  
**Tipo de requerimiento:** Funcional  
**Versión:** 1.0  
**Fuente:** Historia de Usuario 9  
**Prioridad:** Alta/Must  
**Dificultad:** Nominal  
**Actores:** Usuario  

**Descripción:** El sistema debe detectar y mostrar mensajes de error claros y específicos ante fallos o entradas inválidas.

**Justificación:** Permite al usuario entender y corregir problemas rápidamente.

**Precondiciones:** Sistema en ejecución

**Restricciones:** Mensajes en español

**Dependencia:** Validaciones frontend y backend

**Entradas:** Errores del sistema

**Proceso:**
1. Detectar error
2. Clasificar error
3. Mostrar mensaje

**Salida:** Mensaje de error

**Postcondiciones:** Usuario informado

**Criterios de aceptación:**
1. Cada error muestra un mensaje específico (no genérico).
2. Los mensajes están en español claro y comprensible.
3. Se incluyen sugerencias de solución (ej: "suba una imagen válida").
4. Los errores no bloquean completamente la aplicación.
5. Los mensajes desaparecen o se pueden cerrar manualmente.
6. Mensajes específicos implementados: "Imagen no válida o corrupta", "No se pudo procesar la imagen", "Error de formato de archivo", "Error en el procesamiento del modelo".

**Requerimientos no funcionales:** Usabilidad, fiabilidad  
**Estado:** Pendiente  
**Observaciones:** Clasificar errores: validación, conexión, sistema

---

### RF-10: Integración frontend-backend
**Código de identificación:** RF-10  
**Tipo de requerimiento:** Funcional  
**Versión:** 1.0  
**Fuente:** Historia de Usuario 10  
**Prioridad:** Alta/Must  
**Dificultad:** Difícil  
**Actores:** Sistema (frontend/backend)  

**Descripción:** El sistema debe permitir la comunicación entre el frontend y el backend mediante una API REST para enviar imágenes y recibir resultados.

**Justificación:** Garantiza el funcionamiento del sistema de IA.

**Precondiciones:** Backend activo

**Restricciones:** Uso de protocolo HTTP/HTTPS

**Dependencia:** API /predict

**Entradas:** Imagen + modelo

**Proceso:**
1. Envío de solicitud HTTP
2. Procesamiento backend
3. Respuesta

**Salida:** JSON con resultados

**Postcondiciones:** Datos disponibles en frontend

**Criterios de aceptación:**
1. El frontend envía correctamente la imagen y modelo al endpoint /predict.
2. El backend responde con un JSON estructurado (clase, probabilidades, métricas).
3. Se maneja correctamente CORS sin errores de navegador.
4. El tiempo total de comunicación no supera 5 segundos.
5. En caso de error, se devuelve un código HTTP adecuado (ej: 500, 400).

**Requerimientos no funcionales:** Rendimiento, seguridad  
**Estado:** Pendiente  
**Observaciones:** Uso de estándares REST

---

## REQUERIMIENTOS NO FUNCIONALES (RNF)

### RNF-01: Rendimiento del sistema
**Código de identificación:** RNF-01  
**Tipo de requerimiento:** No funcional  
**Versión:** 1.0  
**Fuente:** Requisito de calidad del sistema  
**Prioridad:** Alta/Must  
**Dificultad:** Nominal  
**Actores:** Sistema  

**Descripción:** El sistema debe procesar las solicitudes de predicción y mostrar resultados en tiempos adecuados para uso clínico.

**Justificación:** En entornos médicos, la rapidez en la respuesta es fundamental para la toma de decisiones.

**Precondiciones:** Sistema en funcionamiento

**Restricciones:** Dependencia de recursos del servidor

**Dependencia:** Infraestructura backend

**Entradas:** Solicitudes de predicción

**Proceso:** Procesamiento de imagen y respuesta

**Salida:** Resultado en tiempo óptimo

**Postcondiciones:** Usuario recibe resultados rápidamente

**Criterios de aceptación:**
1. El tiempo de respuesta de una predicción no debe superar los 5 segundos en condiciones normales.
2. La carga de imágenes no debe superar 2 segundos.
3. El sistema soporta al menos 10 usuarios concurrentes sin degradación significativa (>20% aumento de tiempo).
4. El tiempo de visualización de resultados es menor a 1 segundo tras recibir datos.
5. El sistema mantiene tiempos estables durante múltiples ejecuciones consecutivas.

**Requerimientos no funcionales:** Rendimiento  
**Estado:** Pendiente  
**Observaciones:** Puede requerir optimización o uso de GPU

---

### RNF-02: Usabilidad del sistema
**Código de identificación:** RNF-02  
**Tipo de requerimiento:** No funcional  
**Prioridad:** Alta/Must  
**Dificultad:** Fácil  
**Actores:** Usuario  

**Descripción:** El sistema debe ser fácil de usar, intuitivo y comprensible para usuarios médicos sin necesidad de capacitación avanzada.

**Justificación:** Facilita la adopción del sistema por profesionales de la salud.

**Precondiciones:** Interfaz disponible

**Restricciones:** Idioma español

**Dependencia:** Diseño UI/UX

**Entradas:** Interacciones del usuario

**Proceso:** Navegación e interacción

**Salida:** Experiencia de usuario fluida

**Postcondiciones:** Usuario completa tareas sin dificultad

**Criterios de aceptación:**
1. La interfaz está completamente en español.
2. Un usuario nuevo puede realizar una predicción sin ayuda en menos de 2 minutos.
3. Los botones y opciones tienen etiquetas claras y comprensibles.
4. La navegación entre páginas no genera confusión.
5. No se requieren más de 3 clics para ejecutar una predicción.

**Requerimientos no funcionales:** Usabilidad  
**Estado:** Pendiente  
**Observaciones:** Validar con pruebas de usuario

---

### RNF-03: Seguridad de la información
**Código de identificación:** RNF-03  
**Tipo de requerimiento:** No funcional  
**Prioridad:** Alta/Must  
**Dificultad:** Difícil  
**Actores:** Sistema  

**Descripción:** El sistema debe garantizar la protección de los datos médicos mediante mecanismos de seguridad adecuados.

**Justificación:** Los datos médicos son sensibles y requieren protección.

**Precondiciones:** Sistema desplegado

**Restricciones:** Cumplimiento de buenas prácticas de seguridad

**Dependencia:** Infraestructura y backend

**Entradas:** Datos médicos

**Proceso:** Protección y transmisión segura

**Salida:** Datos protegidos

**Postcondiciones:** Información segura

**Criterios de aceptación:**
1. Toda comunicación se realiza mediante HTTPS.
2. Los archivos cargados no se almacenan permanentemente sin autorización.
3. No se exponen datos sensibles en el frontend.
4. El sistema previene accesos no autorizados.
5. Se validan entradas para evitar ataques (ej: archivos maliciosos).

**Requerimientos no funcionales:** Seguridad  
**Estado:** Pendiente  
**Observaciones:** Considerar anonimización de datos

---

### RNF-04: Fiabilidad del sistema
**Código de identificación:** RNF-04  
**Tipo de requerimiento:** No funcional  
**Prioridad:** Alta/Must  
**Dificultad:** Nominal  
**Actores:** Sistema  

**Descripción:** El sistema debe operar de manera continua y sin fallos frecuentes.

**Justificación:** Garantiza disponibilidad en entornos clínicos.

**Precondiciones:** Sistema desplegado

**Restricciones:** Dependencia de infraestructura

**Dependencia:** Servidor

**Entradas:** Solicitudes

**Proceso:** Operación continua

**Salida:** Servicio estable

**Postcondiciones:** Sistema disponible

**Criterios de aceptación:**
1. El sistema tiene una disponibilidad mínima del 95% mensual.
2. Los errores críticos no superan el 2% de las solicitudes.
3. El sistema se recupera automáticamente de fallos menores.
4. No se pierde información durante fallos.
5. Se registran errores en logs para análisis.

**Requerimientos no funcionales:** Fiabilidad  
**Estado:** Pendiente  
**Observaciones:** Implementar logs y monitoreo

---

### RNF-05: Compatibilidad del sistema
**Código de identificación:** RNF-05  
**Tipo de requerimiento:** No funcional  
**Prioridad:** Media/Should  
**Dificultad:** Fácil  
**Actores:** Usuario  

**Descripción:** El sistema debe funcionar correctamente en diferentes navegadores y dispositivos modernos.

**Justificación:** Permite acceso desde múltiples entornos.

**Precondiciones:** Sistema web

**Restricciones:** Navegadores modernos

**Dependencia:** Frontend

**Entradas:** Acceso web

**Proceso:** Renderizado

**Salida:** Interfaz funcional

**Postcondiciones:** Sistema usable

**Criterios de aceptación:**
1. Funciona correctamente en Chrome, Edge y Firefox.
2. No presenta errores visuales críticos.
3. La interfaz se adapta a diferentes resoluciones.
4. Las funciones principales funcionan en todos los navegadores soportados.
5. No requiere instalación adicional.

**Requerimientos no funcionales:** Compatibilidad  
**Estado:** Pendiente  
**Observaciones:** Pruebas cross-browser

---

### RNF-06: Escalabilidad del sistema
**Código de identificación:** RNF-06  
**Tipo de requerimiento:** No funcional  
**Prioridad:** Media  
**Dificultad:** Difícil  
**Actores:** Sistema  

**Descripción:** El sistema debe ser capaz de manejar un incremento en el número de usuarios y solicitudes sin degradación significativa.

**Justificación:** Permite crecimiento del sistema.

**Precondiciones:** Infraestructura disponible

**Restricciones:** Recursos del servidor

**Dependencia:** Backend

**Entradas:** Múltiples solicitudes

**Proceso:** Procesamiento concurrente

**Salida:** Respuestas estables

**Postcondiciones:** Sistema estable

**Criterios de aceptación:**
1. El sistema soporta al menos 50 usuarios concurrentes.
2. El tiempo de respuesta no aumenta más del 30% bajo carga.
3. No se generan caídas del sistema bajo carga moderada.
4. Se pueden agregar recursos (horizontal/vertical).
5. El sistema mantiene estabilidad durante pruebas de estrés.

**Requerimientos no funcionales:** Escalabilidad  
**Estado:** Pendiente  
**Observaciones:** Uso de cloud recomendado

---

### RNF-07: Mantenibilidad del sistema
**Código de identificación:** RNF-07  
**Tipo de requerimiento:** No funcional  
**Prioridad:** Media  
**Dificultad:** Nominal  
**Actores:** Desarrollador  

**Descripción:** El sistema debe ser fácil de mantener, actualizar y modificar.

**Justificación:** Reduce costos de desarrollo a largo plazo.

**Precondiciones:** Código implementado

**Restricciones:** Buenas prácticas

**Dependencia:** Código fuente

**Entradas:** Cambios en el sistema

**Proceso:** Modificación

**Salida:** Sistema actualizado

**Postcondiciones:** Código mantenible

**Criterios de aceptación:**
1. El código está documentado en al menos un 70%.
2. Se siguen estándares de codificación.
3. Los módulos están desacoplados.
4. Se pueden hacer cambios sin afectar otros módulos.
5. Existe control de versiones.

**Requerimientos no funcionales:** Mantenibilidad  
**Estado:** Pendiente  
**Observaciones:** Uso de Git recomendado

---

### RNF-08: Interoperabilidad del sistema
**Código de identificación:** RNF-08  
**Tipo de requerimiento:** No funcional  
**Prioridad:** Alta  
**Dificultad:** Nominal  
**Actores:** Sistema  

**Descripción:** El sistema debe integrarse correctamente con servicios externos mediante API REST.

**Justificación:** Permite comunicación entre sistemas.

**Precondiciones:** API disponible

**Restricciones:** Estándares REST

**Dependencia:** Backend

**Entradas:** Solicitudes HTTP

**Proceso:** Comunicación API

**Salida:** Respuestas JSON

**Postcondiciones:** Integración funcional

**Criterios de aceptación:**
1. El sistema usa métodos HTTP estándar (GET, POST).
2. Los datos se envían en formato JSON.
3. Las respuestas contienen códigos HTTP correctos.
4. Se maneja CORS correctamente.
5. La API responde en menos de 5 segundos.

**Requerimientos no funcionales:** Interoperabilidad  
**Estado:** Pendiente  
**Observaciones:** Documentar API

---

## HISTORIAS DE USUARIO (HU)

### HU-01: Subida y análisis de imágenes MRI
**ID:** HU-01  
**Nombre:** Subida y análisis de imágenes MRI  
**Actor:** Médico especialista  
**Prioridad:** Alta  
**Frecuencia de uso:** Alta  

**Descripción (Historia):** Como médico especialista, quiero poder subir imágenes de resonancia magnética cerebral (MRI) al sistema para obtener un diagnóstico asistido por inteligencia artificial sobre posibles signos de Alzheimer.

**Objetivo:** Obtener apoyo diagnóstico automatizado basado en IA

**Valor de negocio:** Reduce tiempo de diagnóstico y mejora precisión clínica

**Precondiciones:**
- Acceso al sistema mediante enlace o despliegue local
- Interfaz de análisis cargada correctamente

**Flujo principal:**
1. El usuario accede al sistema
2. Selecciona una imagen MRI
3. El sistema valida el archivo
4. Se muestra vista previa
5. La imagen queda lista para análisis

**Flujos alternos:**
- Archivo inválido → se muestra error
- Tamaño excedido → se rechaza carga

**Postcondiciones:** Imagen cargada correctamente y lista para predicción

**Reglas de negocio:**
- Solo formatos JPG, PNG, DICOM
- Tamaño máximo permitido

**Criterios de aceptación:**
1. El sistema permite cargar archivos JPG, PNG y DICOM.
2. Se muestra una vista previa en menos de 2 segundos.
3. Archivos inválidos generan mensaje claro.
4. No se permite continuar sin imagen válida.
5. La imagen queda disponible para análisis.

**Observaciones:** Acceso restringido por distribución del sistema

---

### HU-02: Selección de modelo de IA
**ID:** HU-02  
**Nombre:** Selección de modelo de IA  
**Actor:** Investigador médico  
**Prioridad:** Alta  
**Frecuencia:** Media  

**Descripción:** Como investigador médico, quiero seleccionar entre diferentes modelos de deep learning para comparar resultados y elegir el más adecuado.

**Objetivo:** Evaluar desempeño de modelos

**Valor de negocio:** Mejora la precisión del diagnóstico

**Precondiciones:**
- Sistema accesible
- Modelos cargados

**Flujo principal:**
1. Usuario visualiza modelos
2. Consulta métricas
3. Selecciona modelo

**Flujos alternos:** No hay modelos → mensaje informativo

**Postcondiciones:** Modelo seleccionado

**Reglas de negocio:** Solo modelos disponibles en el sistema

**Criterios de aceptación:**
1. Se muestran al menos 4 modelos.
2. Se visualizan métricas por modelo.
3. El modelo seleccionado se resalta.
4. Puede cambiarse antes de predecir.
5. Se guarda selección actual.

**Observaciones:** Acceso sin autenticación

---

### HU-03: Ejecución de predicción
**ID:** HU-03  
**Nombre:** Ejecución de predicción  
**Actor:** Profesional de la salud  
**Prioridad:** Alta  
**Frecuencia:** Alta  

**Descripción:** Como profesional de la salud, quiero ejecutar el análisis de la imagen MRI con un solo clic para obtener rápidamente una predicción.

**Objetivo:** Obtener diagnóstico automatizado

**Valor de negocio:** Reduce tiempo de análisis

**Precondiciones:**
- Imagen cargada
- Modelo seleccionado

**Flujo principal:**
1. Usuario presiona "Predecir"
2. Sistema valida datos
3. Envía solicitud
4. Recibe resultado

**Flujos alternos:**
- Error de conexión
- Falta de imagen

**Postcondiciones:** Resultado generado

**Reglas de negocio:** No ejecutar sin datos completos

**Criterios de aceptación:**
1. Botón deshabilitado sin imagen.
2. Indicador de carga visible.
3. Tiempo ≤ 5 segundos.
4. Error claro si falla conexión.
5. No duplicación de solicitudes.

**Observaciones:** Sistema de acceso controlado

---

### HU-04: Visualización de resultados
**ID:** HU-04  
**Nombre:** Visualización de resultados  
**Actor:** Médico tratante  
**Prioridad:** Alta  
**Frecuencia:** Alta  

**Descripción:** Como médico tratante, quiero ver los resultados del análisis para tomar decisiones informadas.

**Objetivo:** Interpretar resultados

**Valor de negocio:** Mejora decisiones clínicas

**Precondiciones:** Predicción realizada

**Flujo principal:** Mostrar resultados

**Postcondiciones:** Datos visibles

**Reglas de negocio:** Mostrar todas las clases

**Criterios de aceptación:**
1. Se muestra clase detectada.
2. Se muestra confianza.
3. Se muestran probabilidades.
4. Datos claros.
5. Tiempo < 1 segundo.

**Observaciones:** —

---

### HU-05: Visualización Grad-CAM
**ID:** HU-05  
**Nombre:** Visualización Grad-CAM  
**Actor:** Neurorradiólogo  
**Prioridad:** Alta  
**Frecuencia de uso:** Media  

**Descripción (Historia):** Como neurorradiólogo, quiero visualizar el mapa de calor Grad-CAM sobre la imagen MRI para entender qué regiones del cerebro fueron relevantes en la predicción del modelo.

**Objetivo:** Interpretar el comportamiento del modelo de IA

**Valor de negocio:** Aumenta la confianza y validación clínica del sistema

**Precondiciones:**
- Sistema accesible
- Predicción realizada previamente

**Flujo principal:**
1. El sistema recibe resultado de predicción
2. Genera mapa Grad-CAM
3. Superpone mapa sobre imagen original
4. Muestra resultado al usuario

**Flujos alternos:** Error en generación → mensaje: "No se pudo generar el mapa explicativo"

**Postcondiciones:** Imagen explicativa disponible

**Reglas de negocio:**
- Solo disponible tras predicción
- Depende del modelo seleccionado

**Criterios de aceptación:**
1. El mapa Grad-CAM se genera automáticamente tras la predicción.
2. Se muestra superpuesto a la imagen original.
3. Las zonas de mayor relevancia se visualizan en colores cálidos (rojo/amarillo).
4. El usuario puede alternar entre imagen original y mapa.
5. El tiempo de generación no supera los 3 segundos.

**Observaciones:** Puede requerir aceleración por GPU

---

### HU-06: Visualización de métricas del modelo
**ID:** HU-06  
**Nombre:** Visualización de métricas del modelo  
**Actor:** Investigador clínico  
**Prioridad:** Alta  
**Frecuencia:** Media  

**Descripción:** Como investigador clínico, quiero visualizar las métricas del modelo utilizado para evaluar su confiabilidad en el diagnóstico.

**Objetivo:** Evaluar rendimiento del modelo

**Valor de negocio:** Mejora la validación científica del sistema

**Precondiciones:**
- Sistema accesible
- Modelo seleccionado

**Flujo principal:**
1. Usuario accede a métricas
2. Sistema obtiene datos del modelo
3. Muestra métricas en tabla

**Flujos alternos:** No hay datos → mensaje informativo

**Postcondiciones:** Métricas visibles

**Reglas de negocio:** Métricas deben corresponder al modelo activo

**Criterios de aceptación:**
1. Se muestran accuracy, precision y recall.
2. Los valores se presentan en porcentaje con máximo 2 decimales.
3. Las métricas corresponden al modelo seleccionado.
4. Se muestran en formato tabular claro.
5. Se cargan en menos de 1 segundo.

**Observaciones:** Se pueden incluir métricas adicionales

---

### HU-07: Comparación de modelos
**ID:** HU-07  
**Nombre:** Comparación de modelos  
**Actor:** Científico de datos  
**Prioridad:** Alta  
**Frecuencia:** Media  

**Descripción:** Como científico de datos, quiero visualizar una tabla comparativa de los modelos disponibles para seleccionar el más adecuado según su rendimiento.

**Objetivo:** Comparar modelos

**Valor de negocio:** Optimiza selección de modelos

**Precondiciones:** Modelos disponibles

**Flujo principal:**
1. Usuario accede a vista de modelos
2. Sistema carga datos
3. Muestra tabla comparativa

**Flujos alternos:** Sin modelos → mensaje informativo

**Postcondiciones:** Comparación disponible

**Reglas de negocio:** Mostrar métricas clave

**Criterios de aceptación:**
1. Se muestran todos los modelos disponibles (mínimo 4).
2. Cada modelo incluye accuracy, precision y recall.
3. La tabla permite comparación clara entre modelos.
4. Los modelos se ordenan por accuracy descendente.
5. El usuario puede regresar a la página principal en máximo 1 clic.

**Observaciones:** Puede incluir filtros

---

### HU-08: Navegación del sistema
**ID:** HU-08  
**Nombre:** Navegación del sistema  
**Actor:** Usuario  
**Prioridad:** Alta  
**Frecuencia:** Alta  

**Descripción:** Como usuario con acceso al sistema, quiero navegar entre las diferentes páginas para utilizar todas las funcionalidades sin confusión.

**Objetivo:** Facilitar uso del sistema

**Valor de negocio:** Mejora experiencia del usuario

**Precondiciones:** Sistema accesible

**Flujo principal:**
1. Usuario visualiza menú
2. Selecciona opción
3. Sistema redirige

**Flujos alternos:** Error de navegación → mensaje

**Postcondiciones:** Usuario en nueva vista

**Reglas de negocio:** Navegación persistente

**Criterios de aceptación:**
1. Existe barra de navegación visible en todo momento.
2. Incluye al menos "Inicio" y "Modelos".
3. La página actual se resalta visualmente.
4. El cambio de página ocurre en menos de 2 segundos.
5. No se pierde información crítica durante la navegación.

**Observaciones:** Puede incluir breadcrumbs

---

### HU-09: Manejo de errores
**ID:** HU-09  
**Nombre:** Manejo de errores  
**Actor:** Usuario  
**Prioridad:** Alta  
**Frecuencia:** Alta  

**Descripción:** Como usuario, quiero recibir mensajes claros cuando ocurren errores para entender qué salió mal y cómo solucionarlo.

**Objetivo:** Mejorar interacción y solución de problemas

**Valor de negocio:** Reduce errores de uso

**Precondiciones:** Sistema en ejecución

**Flujo principal:**
1. Sistema detecta error
2. Clasifica tipo
3. Muestra mensaje

**Flujos alternos:** Error crítico → mensaje reforzado

**Postcondiciones:** Usuario informado

**Reglas de negocio:** Mensajes en español

**Criterios de aceptación:**
1. Cada error muestra un mensaje específico (no genérico).
2. Los mensajes están en español claro.
3. Se incluyen sugerencias de solución.
4. Los errores no bloquean completamente el sistema.
5. Los mensajes pueden cerrarse manualmente.

**Observaciones:** Clasificar errores por tipo

---

### HU-10: Integración del sistema
**ID:** HU-10  
**Nombre:** Integración del sistema  
**Actor:** Administrador del sistema  
**Prioridad:** Alta  
**Frecuencia:** Alta  

**Descripción:** Como administrador del sistema, quiero que el frontend se comunique correctamente con el backend para asegurar el funcionamiento del sistema de IA.

**Objetivo:** Garantizar funcionamiento técnico

**Valor de negocio:** Asegura operación del sistema

**Precondiciones:** Backend activo

**Flujo principal:**
1. Usuario envía solicitud
2. Frontend envía datos
3. Backend procesa
4. Retorna respuesta

**Flujos alternos:**
- Error de conexión
- Error de servidor

**Postcondiciones:** Datos disponibles en frontend

**Reglas de negocio:** Uso de API REST

**Criterios de aceptación:**
1. El frontend envía correctamente imagen y modelo al endpoint /predict.
2. El backend responde con JSON estructurado.
3. Se maneja correctamente CORS.
4. El tiempo de respuesta es menor a 5 segundos.
5. Se gestionan correctamente errores HTTP (400, 500).

**Observaciones:** Uso de estándares REST
