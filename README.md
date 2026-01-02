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
>>> ENTRENAMIENTO: 3 -> 4 -> 1 <<<

Epoch     0 | Error: 0.5061 | Precision: 50%
Epoch  2000 | Error: 0.2313 | Precision: 92%
Epoch  4000 | Error: 0.1432 | Precision: 92%

>>> COMPLETO <<<

 1. ATACA (92.4%) | Real: ATACA [OK]  Descansado, sin hambre, juguete
 2. IGNORA ( 0.9%) | Real: IGNORA [OK]  Agotado, hambriento, sin estimulo
 3. ATACA (93.0%) | Real: ATACA [OK]  Muy descansado, satisfecho, juguete
 4. IGNORA ( 0.0%) | Real: IGNORA [OK]  Exhausto, mucha hambre
 5. ATACA (89.3%) | Real: ATACA [OK]  Energia media, juguete
 6. IGNORA ( 4.0%) | Real: IGNORA [OK]  Energia media, SIN juguete
 7. IGNORA ( 5.0%) | Real: IGNORA [OK]  Descansado, sin juguete *
 8. ATACA (99.3%) | Real: ATACA [OK]  Cansado, hambriento, juguete *
 9. IGNORA (13.1%) | Real: IGNORA [OK]  Buena energia, juguete *
10. ATACA (94.4%) | Real: ATACA [OK]  Energia media-baja *
11. ATACA (98.0%) | Real: ATACA [OK]  Muy descansado, juguete
12. IGNORA ( 5.5%) | Real: IGNORA [OK]  Cansado, juguete

* = Contraintuitivo

--- CASOS NUEVOS ---
Descansado, algo hambre, juguete   : ATACA (72.7%)
Cansado, poca hambre               : IGNORA (6.0%)
Mucha hambre, juguete              : ATACA (99.6%)

```
