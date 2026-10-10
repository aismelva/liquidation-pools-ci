# Resultado del último backtest (v5.2) — BTCUSDT

- Ejecutado: 2026-10-10 12:51 UTC
- Modelo: **v5.2** (v4.1 + ranking por ratio cascada/absorción con order book L2)
- Datos: 14 días · ventana 1d · lookahead 24h · libro L2: últimas 120h
- Evaluaciones: 16
- Ratio cascada medio: **6.39** (usd del pool / usd absorbible)
- Captura del modelo v5.2: **21.3%** (v4.1/14d: 45.0%)
- Captura null uniforme: **43.1%**
- Captura null distancia-emparejada: **18.7%**
- Ventaja vs uniforme: **-21.8 pts** · vs emparejada: **+2.6 pts**

## Por evaluación

| Paso | Liq USD | Cap modelo | Cap uniforme | Cap emparejada |
|---|---|---|---|---|
| 10-05 17:50 | 13,308,855 | 0.0% | 33.2% | 5.6% |
| 10-05 23:50 | 10,470,665 | 0.0% | 47.4% | 0.0% |
| 10-06 05:50 | 65,759,113 | 58.4% | 61.0% | 33.3% |
| 10-06 11:50 | 71,512,311 | 47.2% | 40.5% | 32.5% |
| 10-06 17:50 | 82,871,801 | 54.0% | 49.1% | 43.4% |
| 10-06 23:50 | 83,292,859 | 4.6% | 57.0% | 22.7% |
| 10-07 05:50 | 53,078,881 | 55.1% | 46.2% | 27.5% |
| 10-07 11:50 | 55,545,380 | 0.0% | 49.0% | 0.0% |
| 10-07 17:50 | 128,900,691 | 30.3% | 43.5% | 24.3% |
| 10-07 23:50 | 133,741,025 | 0.0% | 38.6% | 22.8% |
| 10-08 05:50 | 111,041,790 | 24.2% | 50.3% | 25.2% |
| 10-08 11:50 | 104,361,152 | 33.1% | 34.9% | 31.6% |
| 10-08 17:50 | 18,608,373 | 14.7% | 37.2% | 8.8% |
| 10-08 23:50 | 13,854,525 | 19.8% | 32.2% | 16.8% |
| 10-09 05:50 | 10,647,456 | 0.0% | 46.7% | 0.0% |
| 10-09 11:50 | 5,939,881 | 0.1% | 22.7% | 4.7% |

**Veredicto:** El modelo v5.2 supera el null estricto pero no al uniforme: mejora parcial.

_Diferencial v5: pools × absorción del libro — capa que las plataformas estándar no integran._
