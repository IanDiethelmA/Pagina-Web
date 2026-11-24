"""
migrate_to_supabase.py - Migra los datos locales de SQLite hacia Supabase PostgreSQL.
Uso:
    python migrate_to_supabase.py
Lee DATABASE_URL desde '.env' (vía config.py) y copia de forma idempotente:
  - autos_historico
  - precio_historial
  - favoritos
"""

import sqlite3
from contextlib import closing
from typing import Any, List, Tuple

import config
from database import DatabaseManager


def migrate() -> None:
    if not config.DATABASE_URL:
        raise RuntimeError(
            "Falta DATABASE_URL en '.env'. Agrega la URL de Session Pooler de Supabase y vuelve a intentar."
        )
    if not config.DB_PATH.exists():
        raise FileNotFoundError(f"No se encontró SQLite local en {config.DB_PATH}")

    pg_db = DatabaseManager(database_url=config.DATABASE_URL)

    with closing(sqlite3.connect(str(config.DB_PATH))) as conn:
        conn.row_factory = sqlite3.Row
        autos = [dict(r) for r in conn.execute("SELECT * FROM autos_historico").fetchall()]
        ph = [
            dict(r)
            for r in conn.execute(
                "SELECT id_aviso, precio_antiguo, precio_nuevo, fecha_cambio FROM precio_historial ORDER BY id"
            ).fetchall()
        ]
        favs = [
            dict(r)
            for r in conn.execute("SELECT id_aviso, nota, fecha_guardado FROM favoritos").fetchall()
        ]

    cols_autos = [
        "id_aviso", "url", "marca", "modelo", "anio", "kilometraje",
        "combustible", "transmision", "comuna", "titulo", "descripcion",
        "precio", "precio_inicial", "precio_anterior", "cambios_precio",
        "estado", "fecha_creacion", "fecha_ultima_vista", "fecha_venta",
        "dias_publicado", "alerta_enviada", "categoria_alerta", "datos_corregidos",
    ]
    placeholders = ", ".join(["?"] * len(cols_autos))
    insert_autos_sql = (
        f"INSERT INTO public.autos_historico ({', '.join(cols_autos)}) "
        f"VALUES ({placeholders}) ON CONFLICT (id_aviso) DO NOTHING"
    )

    rows_autos: List[Tuple[Any, ...]] = [
        tuple(r.get(c) for c in cols_autos) for r in autos
    ]

    with pg_db.get_connection() as conn:
        cur = conn.cursor()
        for start in range(0, len(rows_autos), 500):
            cur.executemany(insert_autos_sql, rows_autos[start : start + 500])

        cur.execute("SELECT COUNT(*) AS cnt FROM public.precio_historial")
        existing_ph = int(cur.fetchone()["cnt"] or 0)
        if ph and existing_ph == 0:
            cur.executemany(
                "INSERT INTO public.precio_historial (id_aviso, precio_antiguo, precio_nuevo, fecha_cambio) VALUES (?, ?, ?, ?)",
                [
                    (r["id_aviso"], r["precio_antiguo"], r["precio_nuevo"], r["fecha_cambio"])
                    for r in ph
                ],
            )

        if favs:
            cur.executemany(
                "INSERT INTO public.favoritos (id_aviso, nota, fecha_guardado) VALUES (?, ?, ?) ON CONFLICT (id_aviso) DO NOTHING",
                [(r["id_aviso"], r.get("nota", ""), r["fecha_guardado"]) for r in favs],
            )

    stats = pg_db.get_summary_stats()
    fav_count = len(pg_db.get_favorites_map())
    print(
        f"Migración completada en Supabase -> total={stats['total_historicos']}, "
        f"activos={stats['activos']}, vendidos={stats['vendidos']}, "
        f"historial={stats['cambios_de_precio_registrados']}, favoritos={fav_count}"
    )


if __name__ == "__main__":
    migrate()
