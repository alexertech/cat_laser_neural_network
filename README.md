# Cat Laser - Red Neuronal desde Cero

Proyecto de apoyo para el articulo [Hackea tu Cerebro: Como aprendi redes neuronales con gatos](https://www.greyhat.cl/posts/hackea-tu-cerebro-como-aprendi-redes-neuronales-con-gatos).

Una red neuronal feedforward simple que predice si un gato capturara un punto laser basandose en:

- Posicion del laser (x, y)
- Velocidad del laser
- Direccion del laser
- Nivel de energia del gato

## Requisitos

- Python 3.13+
- Poetry

## Instalacion

```bash
poetry install
```

## Uso

### Entrenar la red

```bash
poetry run python src/cat_laser/cat_nn.py
```

### Ejecutar predicciones interactivas

```bash
poetry run python tests/predictions.py
```

## Estructura

```
cat_laser/
├── src/cat_laser/
│   └── cat_nn.py      # Red neuronal y entrenamiento
├── tests/
│   └── predictions.py # Predicciones interactivas
└── data/
    └── data.csv       # Dataset de entrenamiento
```
