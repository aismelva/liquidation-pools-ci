# Resultado del último backtest (v5.2) — BTCUSDT

- Ejecutado: 2026-10-10 15:36 UTC
- Modelo: **v5.2** (v4.1 + ranking por ratio cascada/absorción con order book L2)
- Datos: 14 días · ventana 1d · lookahead 24h · libro L2: últimas 120h
- Evaluaciones: 15
- Ratio cascada medio: **12.82** (usd del pool / usd absorbible)
- Captura del modelo v5.2: **18.5%** (v4.1/14d: 45.0%)
- Captura null uniforme: **42.2%**
- Captura null distancia-emparejada: **18.4%**
- Ventaja vs uniforme: **-23.7 pts** · vs emparejada: **+0.1 pts**

## Por evaluación

| Paso | Liq USD | Cap modelo | Cap uniforme | Cap emparejada |
|---|---|---|---|---|
| 10-05 20:55 | 11,271,994 | 0.0% | 41.5% | 0.0% |
| 10-06 02:55 | 66,138,883 | 33.2% | 55.3% | 43.2% |
| 10-06 08:55 | 64,070,242 | 38.3% | 49.4% | 26.0% |
| 10-06 14:55 | 82,900,265 | 46.8% | 40.5% | 41.6% |
| 10-06 20:55 | 82,960,400 | 12.0% | 47.7% | 8.9% |
| 10-07 02:55 | 28,064,420 | 15.3% | 43.4% | 9.9% |
| 10-07 08:55 | 53,658,615 | 36.8% | 38.1% | 12.9% |
| 10-07 14:55 | 59,485,168 | 0.0% | 40.6% | 0.0% |
| 10-07 20:55 | 131,755,908 | 0.0% | 39.0% | 21.9% |
| 10-08 02:55 | 135,599,824 | 31.6% | 37.7% | 32.2% |
| 10-08 08:55 | 110,407,032 | 22.3% | 34.3% | 20.7% |
| 10-08 14:55 | 90,595,733 | 11.8% | 35.1% | 23.1% |
| 10-08 20:55 | 16,323,040 | 5.2% | 38.8% | 15.9% |
| 10-09 02:55 | 11,267,389 | 24.3% | 41.0% | 19.5% |
| 10-09 08:55 | 10,443,445 | 0.0% | 51.3% | 0.0% |

**Veredicto:** El modelo v5.2 supera el null estricto pero no al uniforme: mejora parcial.

_Diferencial v5: pools × absorción del libro — capa que las plataformas estándar no integran._
