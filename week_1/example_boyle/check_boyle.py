"""Внешняя проверка к демонстрации агента: выполняется ли p·V = const.

Независимо от агента считает p·V по каждой строке, медиану и отклонения.
Строка с |pV/медиана − 1| > 10 % помечается как подозрительная
(типичная причина — ошибка единиц). Итог: относительный разброс pV по
строкам без подозрительных.

Скрипт лежит вне папки project/: агент, работающий в проекте, его не видит,
и проверка остаётся независимой.

Запуск из папки примера:  python check_boyle.py
Для копии проекта:        python check_boyle.py ~/demo_boyle/data/measurements.csv
Код возврата: 0 — подозрительных строк нет, 1 — есть.
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

THRESHOLD = 0.10


def main(path: str) -> int:
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    pv = [float(r["volume_l"]) * float(r["pressure_kpa"]) for r in rows]
    med = statistics.median(pv)
    suspicious = [(i + 1, rows[i], x) for i, x in enumerate(pv) if abs(x / med - 1) > THRESHOLD]
    clean = [x for x in pv if abs(x / med - 1) <= THRESHOLD]
    mean = statistics.mean(clean)
    rel_sd = statistics.stdev(clean) / mean
    print(f"строк: {len(rows)}; медиана pV = {med:.1f} кПа·л")
    for n, r, x in suspicious:
        print(f"ПОДОЗРИТЕЛЬНО: строка {n}: V={r['volume_l']} л, p={r['pressure_kpa']} кПа, pV={x:.2f} "
              f"(отклонение {x / med - 1:+.0%}) — проверьте единицы")
    print(f"без подозрительных строк: pV = {mean:.1f} кПа·л, относительный разброс {rel_sd:.1%}")
    return 1 if suspicious else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "project/data/measurements.csv"))
