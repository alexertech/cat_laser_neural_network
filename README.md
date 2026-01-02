# Cat Laser - Red Neuronal desde Cero

Proyecto de apoyo para el articulo [Hackea tu Cerebro](https://www.greyhat.cl/posts/hackea-tu-cerebro-como-aprendi-redes-neuronales-con-gatos).

Una red neuronal feedforward simple que predice si un gato atacara un punto laser basandose en:

- Horas dormidas (0-10)
- Nivel de hambre (0-1)
- Juguete cerca (0/1)

## Requisitos

- Python 3.13+
- Poetry

## Instalacion

```bash
poetry install
```

## Uso

### Ver entrenamiento y predicciones

```bash
poetry run python src/cat_laser/cat_nn.py
```

### Modo interactivo

```bash
poetry run python tests/predictions.py
```

## Estructura

```
cat_laser/
├── src/cat_laser/
│   └── cat_nn.py      # Red neuronal con datos inline
└── tests/
    └── predictions.py # Predicciones interactivas
```

## Ejemplo

```
❯ poetry run python src/cat_laser/cat_nn.py

>>> INICIANDO ENTRENAMIENTO <<<
>>> PESOS INICIALES ALEATORIOS ESTABLECIDOS <<<

Epoch    0 | Error promedio: 0.5132
            | Ejemplo: pred=0.309 real=1 (Gato descansado)
Epoch 1000 | Error promedio: 0.0247
            | Ejemplo: pred=0.998 real=1 (Gato descansado)
Epoch 2000 | Error promedio: 0.0154
            | Ejemplo: pred=0.999 real=1 (Gato descansado)
Epoch 3000 | Error promedio: 0.0119
            | Ejemplo: pred=0.999 real=1 (Gato descansado)
Epoch 4000 | Error promedio: 0.0099
            | Ejemplo: pred=0.999 real=1 (Gato descansado)

>>> ENTRENAMIENTO COMPLETO <<<

--- PREDICCIONES EN DATASET DE ENTRENAMIENTO ---
Gato 1: ATACA (conf: 99.9%) | Real: ATACA [OK]
         Descansado, sin hambre, juguete
Gato 2: IGNORA (conf: 0.1%) | Real: IGNORA [OK]
         Agotado, hambriento, sin juguete
Gato 3: ATACA (conf: 99.4%) | Real: ATACA [OK]
         Energia media, algo hambre, juguete
Gato 4: IGNORA (conf: 1.8%) | Real: IGNORA [OK]
         Casi dormido, poca hambre, sin juguete
Gato 5: ATACA (conf: 99.9%) | Real: ATACA [OK]
         Descansado, satisfecho, sin juguete
Gato 6: IGNORA (conf: 0.1%) | Real: IGNORA [OK]
         Cansado, hambriento, juguete
Gato 7: ATACA (conf: 97.7%) | Real: ATACA [OK]
         Energia media, hambre media, juguete
Gato 8: IGNORA (conf: 1.9%) | Real: IGNORA [OK]
         Cansado, sin hambre, juguete

--- GENERALIZANDO A CASOS NUEVOS ---
Muy descansado, sin hambre, juguete cerca    : ATACA (conf: 100.0%)
Exhausto y hambriento, sin juguete           : IGNORA (conf: 0.0%)
Energia media, hambre media, sin juguete     : ATACA (conf: 99.3%)

```
