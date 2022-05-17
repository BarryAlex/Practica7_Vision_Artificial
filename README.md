# Practica7_Vision_Artificial
Eliminación de falsos negativos y falsos positivos en imágenes

# Objetivo
Remover Ruido – Lineal y morfológicamente en VIDEO.
Remover ruido de la detección F+ y F-

Falsos positivos - En el área negra lejos del objeto a detectar se ven pixeles azules porque estamos buscando el color azul
Falsos negativos - Dentro del área azul se ven pixeles negros.

# Limitaciones
Los valores máximos y mínimos empleados en mi código se tomarón de acuerdo al modelo empleado, a mi cámara, los objetos que quiero reconocer y el lugar en donde se realizaron las pruebas, así como la iluminación del entorno, además como cierta distancia en la que mi cámara los puede detectar.

No he podido hacer funcionar el video con el modelo YUV, por lo que en las funciones de dicho modelo solo imprime en consola el valor que debería identificar
