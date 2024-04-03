# Descripción

La compañía de seguros Sure Tomorrow quiere resolver varias tareas con la ayuda de machine learning y te pide que evalúes esa posibilidad.

Tarea 1: encontrar clientes que sean similares a un cliente determinado. Esto ayudará a los agentes de la compañía con el marketing.

Tarea 2: predecir si es probable que un nuevo cliente reciba un beneficio de seguro. ¿Puede un modelo de predicción funcionar mejor que un modelo ficticio?

Tarea 3: predecir la cantidad de beneficios de seguro que probablemente recibirá un nuevo cliente utilizando un modelo de regresión lineal.

Tarea 4: proteger los datos personales de los clientes sin romper el modelo de la tarea anterior.

Es necesario desarrollar un algoritmo de transformación de datos que dificulte la recuperación de la información personal si los datos caen en manos equivocadas. Esto se denomina enmascaramiento de datos u ofuscación de datos. Pero los datos deben protegerse de tal manera que la calidad de los modelos de machine learning no se vea afectada. No es necesario elegir el mejor modelo, basta con demostrar que el algoritmo funciona correctamente.

## Instrucciones del proyecto

### Carga los datos.

Verifica que los datos no tengan problemas: no faltan datos, no hay valores extremos, etc.
Trabaja en cada tarea y responde las preguntas planteadas en la plantilla del proyecto.
Saca conclusiones basadas en tu experiencia trabajando en el proyecto.
Hay algo de código previo en la plantilla del proyecto, siéntete libre de usarlo. Primero se debe terminar algo de código previo. Además, hay dos apéndices en la plantilla del proyecto con información útil.

## Descripción de datos

El dataset se almacena en el archivo /datasets/insurance_us.csv. Puedes descargar el dataset aquí.

Características: sexo, edad, salario y número de familiares de la persona asegurada.

Objetivo: número de beneficios de seguro recibidos por una persona asegurada en los últimos cinco años.