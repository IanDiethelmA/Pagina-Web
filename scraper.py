"""
scraper.py - MÓDULO 1: Motor Multi-Portal y Buscador Segmentado (Mercado Automotriz Chileno).
Implementa:
1. Extracción real en vivo desde múltiples portales públicos en Chile:
   - Autocosmos Chile (autocosmos.cl/auto/usado)
   - MercadoLibre Autos Chile (autos.mercadolibre.cl)
   - Yapo Autos Chile (yapo.cl/autos-usados)
   - Chileautos (chileautos.cl)
2. Búsqueda Segmentada / Focalizada por Marca y Modelo:
   - Permite buscar un auto específico bajo demanda (--marca MAZDA --modelo 3) para no
     saturar servidores y construir curvas específicas de depreciación por modelo.
   - Modo Barrido Multi-Segmento (--segmented) que recorre las marcas y modelos más
     transados en Chile recopilando cientos o miles de avisos reales.
3. Bloques Try/Except independientes en cada etiqueta HTML de cada portal.
4. Fallback Realista Ampliado en caso de desconexión total de red.
"""

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup, Tag

import config
from database import DatabaseManager

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = config.setup_logger("Modulo1.Scraper")


class RealisticChileanMarketFallback:
    """
    Generador de respaldo del parque automotriz usado en Chile.
    Solo se activa si el equipo está completamente sin conexión a Internet
    o si se solicita explícitamente con --fallback.
    """

    CATALOG = [
        ("MAZDA", "3", 17_800_000, "Bencina", "Automática", "Sedan/Hatch"),
        ("MAZDA", "CX-5", 22_900_000, "Bencina", "Automática", "SUV"),
        ("TOYOTA", "RAV4", 23_500_000, "Bencina", "Automática", "SUV"),
        ("TOYOTA", "YARIS", 13_200_000, "Bencina", "Manual", "Sedan/Hatch"),
        ("TOYOTA", "COROLLA", 17_500_000, "Híbrido", "Automática", "Sedan"),
        ("TOYOTA", "HILUX", 26_500_000, "Diésel", "Manual", "Pick-up"),
        ("SUZUKI", "SWIFT", 11_800_000, "Bencina", "Manual", "Hatchback"),
        ("SUZUKI", "BALENO", 12_400_000, "Bencina", "Manual", "Hatchback"),
        ("KIA", "RIO", 12_900_000, "Bencina", "Manual", "Sedan/Hatch"),
        ("KIA", "MORNING", 9_400_000, "Bencina", "Manual", "Citycar"),
        ("KIA", "SPORTAGE", 21_800_000, "Diésel", "Automática", "SUV"),
        ("HYUNDAI", "ACCENT", 13_100_000, "Bencina", "Manual", "Sedan"),
        ("HYUNDAI", "TUCSON", 21_500_000, "Bencina", "Automática", "SUV"),
        ("CHEVROLET", "SAIL", 9_800_000, "Bencina", "Manual", "Sedan"),
        ("NISSAN", "QASHQAI", 19_600_000, "Bencina", "Automática", "SUV"),
        ("NISSAN", "VERSA", 13_800_000, "Bencina", "Automática", "Sedan"),
        ("SUBARU", "XV", 20_900_000, "Bencina", "Automática", "SUV"),
        ("VOLKSWAGEN", "GOL", 10_600_000, "Bencina", "Manual", "Hatchback"),
        ("PEUGEOT", "208", 14_200_000, "Diésel", "Manual", "Hatchback"),
        ("PEUGEOT", "3008", 23_400_000, "Diésel", "Automática", "SUV"),
        ("FORD", "RANGER", 25_900_000, "Diésel", "Automática", "Pick-up"),
        ("MITSUBISHI", "L200", 25_200_000, "Diésel", "Manual", "Pick-up"),
        ("HONDA", "CIVIC", 18_900_000, "Bencina", "Automática", "Sedan"),
        ("HONDA", "CR-V", 25_500_000, "Bencina", "Automática", "SUV"),
        ("HONDA", "HR-V", 19_800_000, "Bencina", "Automática", "SUV"),
        ("HONDA", "PILOT", 34_500_000, "Bencina", "Automática", "SUV"),
        ("VOLVO", "XC60", 36_900_000, "Híbrido", "Automática", "SUV"),
        ("VOLVO", "XC40", 29_500_000, "Bencina", "Automática", "SUV"),
        ("VOLVO", "V40", 16_500_000, "Bencina", "Automática", "Hatchback"),
        ("VOLVO", "XC90", 48_900_000, "Híbrido", "Automática", "SUV"),
        ("BMW", "320I", 33_900_000, "Bencina", "Automática", "Sedan"),
        ("BMW", "118I", 24_900_000, "Bencina", "Automática", "Hatchback"),
        ("BMW", "X1", 29_900_000, "Bencina", "Automática", "SUV"),
        ("BMW", "X3", 39_500_000, "Bencina", "Automática", "SUV"),
        ("MERCEDES-BENZ", "A200", 27_900_000, "Bencina", "Automática", "Hatchback"),
        ("MERCEDES-BENZ", "C200", 35_900_000, "Bencina", "Automática", "Sedan"),
        ("MERCEDES-BENZ", "GLA", 31_500_000, "Bencina", "Automática", "SUV"),
        ("MERCEDES-BENZ", "GLC", 42_900_000, "Diésel", "Automática", "SUV"),
    ]

    COMUNAS_CHILE = [
        "Las Condes", "Vitacura", "Lo Barnechea", "Providencia", "Ñuñoa",
        "La Reina", "La Florida", "Maipú", "Santiago", "San Miguel",
        "Huechuraba", "Colina", "Viña del Mar", "Concón", "Concepción",
        "Temuco", "Puerto Varas", "Antofagasta", "La Serena", "Rancagua",
        "Puente Alto", "Quilicura", "San Bernardo", "Talca", "Pudahuel",
    ]

    @classmethod
    def generate_listings(cls, total_records: int = 600, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Genera un lote amplio de publicaciones simuladas cuando se opera offline."""
        rng = random.Random(seed if seed is not None else 20261106)
        listings: List[Dict[str, Any]] = []

        for idx in range(1, total_records + 1):
            marca, modelo, base_price_2024, default_fuel, default_trans, _ = rng.choice(cls.CATALOG)
            anio = rng.randint(2011, 2025)
            age = max(0, 2025 - anio)
            km_mean = age * 13_800 + rng.randint(4_000, 18_000)
            kilometraje = max(1_500, int(rng.gauss(km_mean, 12_000)))

            transmision = default_trans if rng.random() < 0.72 else ("Automática" if default_trans == "Manual" else "Manual")
            combustible = default_fuel if rng.random() < 0.80 else rng.choice(["Bencina", "Diésel", "Híbrido"])
            comuna = rng.choice(cls.COMUNAS_CHILE)
            tier = config.COMUNA_WEALTH_TIER.get(comuna.upper(), config.DEFAULT_COMUNA_TIER)

            is_unico_dueno = rng.random() < 0.32
            is_mantenciones = rng.random() < 0.45
            is_cuero = rng.random() < (0.40 if tier >= 4 else 0.20)
            is_inmaculado = rng.random() < 0.28
            is_sunroof = rng.random() < 0.25
            is_full = rng.random() < 0.50
            is_laminas = rng.random() < 0.22
            is_neumaticos = rng.random() < 0.24

            depreciation_factor = 0.932 ** age
            km_delta = kilometraje - (age * 14_000)
            km_adjustment = max(0.70, min(1.12, 1.0 - (km_delta / 450_000)))

            extras_multiplier = 1.0
            if transmision == "Automática":
                extras_multiplier += 0.045
            if combustible in ("Diésel", "Híbrido"):
                extras_multiplier += 0.04
            if is_unico_dueno:
                extras_multiplier += 0.035
            if is_mantenciones:
                extras_multiplier += 0.03
            if is_cuero:
                extras_multiplier += 0.03
            if is_inmaculado:
                extras_multiplier += 0.025
            if is_full:
                extras_multiplier += 0.025
            if tier == 5:
                extras_multiplier += 0.03
            elif tier <= 2:
                extras_multiplier -= 0.025

            fair_market_price = int(
                round((base_price_2024 * depreciation_factor * km_adjustment * extras_multiplier * rng.uniform(0.96, 1.04)) / 50_000)
                * 50_000
            )

            desc_parts = [f"Vendo {marca.title()} {modelo} año {anio} ubicado en {comuna}."]
            title_extras = []
            if is_full:
                title_extras.append("Full Equipo")
                desc_parts.append("Versión top de línea full equipo.")
            if is_unico_dueno:
                title_extras.append("Único Dueño")
                desc_parts.append("Único dueño desde nuevo.")
            if is_mantenciones:
                desc_parts.append("Mantenciones al día en la marca y papeles al día.")
            if is_cuero:
                desc_parts.append("Asientos de cuero en excelente estado.")
            if is_inmaculado:
                desc_parts.append("Estado inmaculado, sin detalles.")
            if is_sunroof:
                desc_parts.append("Cuenta con sunroof / techo panorámico.")
            if is_laminas:
                desc_parts.append("Láminas de seguridad certificadas.")
            if is_neumaticos:
                desc_parts.append("Neumáticos nuevos recién cambiados.")

            precio_publicado = fair_market_price
            if idx in (15, 42, 78, 115, 164, 210):
                discount = rng.uniform(0.18, 0.27)
                precio_publicado = int(round((fair_market_price * (1.0 - discount)) / 50_000) * 50_000)
                desc_parts.append("Vende por apuro y viaje, precio conversable, oferta.")
            elif idx in (88, 195):
                discount = rng.uniform(0.68, 0.82)
                precio_publicado = int(round((fair_market_price * (1.0 - discount)) / 50_000) * 50_000)

            titulo = f"{marca.title()} {modelo} {anio} {transmision} {' '.join(title_extras[:2])}".strip()
            id_aviso = f"CL-AD-{20260000 + idx}"
            slug = f"{marca.lower()}-{modelo.lower().replace(' ', '-')}-{anio}"
            url = f"https://www.chileautos.cl/vehiculos/detalles/{slug}/{id_aviso}/"

            listings.append(
                {
                    "id_aviso": id_aviso,
                    "url": url,
                    "marca": marca.upper(),
                    "modelo": modelo.upper(),
                    "anio": anio,
                    "kilometraje": kilometraje,
                    "combustible": combustible,
                    "transmision": transmision,
                    "comuna": comuna,
                    "titulo": titulo,
                    "descripcion": " ".join(desc_parts),
                    "precio": precio_publicado,
                }
            )
        return listings


class ChileautosScraper:
    """
    Motor de Adquisición Multi-Portal y Segmentado para el Mercado Chileno:
    Extrae anuncios reales en vivo desde:
      1. Autocosmos Chile (autocosmos.cl)
      2. MercadoLibre Autos Chile (autos.mercadolibre.cl)
      3. Yapo Autos Chile (yapo.cl)
      4. Chileautos (chileautos.cl)
    Soporta tanto barrido general/multi-segmento como búsqueda focalizada (--marca y --modelo).
    """

    KNOWN_BRANDS = {
        "TOYOTA", "MAZDA", "SUZUKI", "HYUNDAI", "KIA", "CHEVROLET", "NISSAN",
        "PEUGEOT", "FORD", "VOLKSWAGEN", "SUBARU", "MITSUBISHI", "HONDA",
        "BMW", "MERCEDES-BENZ", "MERCEDES", "AUDI", "VOLVO", "JEEP", "CHERY",
        "MG", "JAC", "GREAT WALL", "HAVAL", "CHANGAN", "CITROEN", "CITROËN",
        "RENAULT", "SSANGYONG", "OPEL", "FIAT", "RAM", "DODGE", "LEXUS",
        "MAXUS", "GEELY", "DFSK", "BYD", "OMODA", "JETOUR", "CUPRA", "SEAT",
        "MINI", "PORSCHE", "LAND ROVER", "MAHINDRA", "ZX AUTO", "FOTON",
    }

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db = db_manager or DatabaseManager()
        self.session = requests.Session()

    def _build_headers(self, referer: str = "https://www.google.cl/") -> Dict[str, str]:
        """Genera cabeceras HTTP limpias con rotación de User-Agent."""
        return {
            "User-Agent": random.choice(config.USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "es-CL,es;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": referer,
            "Connection": "keep-alive",
        }

    @staticmethod
    def _clean_int(raw_value: Any, default: int = 0) -> int:
        if raw_value is None:
            return default
        try:
            digits = re.sub(r"[^\d]", "", str(raw_value))
            return int(digits) if digits else default
        except Exception:
            return default

    @classmethod
    def _normalize_brand_name(cls, raw_brand: Optional[str]) -> str:
        """Normaliza variantes de marcas compuestas como Mercedes-Benz."""
        if not raw_brand:
            return ""
        b = raw_brand.strip().upper()
        if b in ("MERCEDES", "MERCEDES BENZ", "MERCEDES-BENZ", "MB"):
            return "MERCEDES-BENZ"
        if b in ("VW", "VOLKS WAGEN"):
            return "VOLKSWAGEN"
        return b

    @classmethod
    def _normalize_model_name(cls, raw_model: Optional[str]) -> str:
        """Normaliza variantes y errores tipográficos de modelos (ej. XTRAIL -> X-TRAIL)."""
        if not raw_model:
            return ""
        m = raw_model.strip().upper()
        # Limpiar guiones o espacios múltiples para buscar en aliases
        clean_m = re.sub(r"\s+", " ", m)
        if clean_m in config.MODEL_ALIASES:
            return config.MODEL_ALIASES[clean_m]
        no_hyphen = clean_m.replace("-", "").replace(" ", "")
        if no_hyphen in config.MODEL_ALIASES:
            return config.MODEL_ALIASES[no_hyphen]
        return m

    _EXTRA_KNOWN_BRANDS = {
        "MAXUS", "OMODA", "JETOUR", "GWM", "DFSK", "BYD", "GEELY", "HAVAL",
        "CHERY", "CHANGAN", "MG", "JAC", "RAM", "DODGE", "SSANGYONG", "SEAT",
        "SKODA", "OPEL", "FIAT", "MINI", "LEXUS", "PORSCHE", "CITROEN",
        "RENAULT", "PEUGEOT", "SUBARU", "HYUNDAI", "VOLKSWAGEN", "MITSUBISHI",
        "CHEVROLET", "TOYOTA", "MAZDA", "KIA", "NISSAN", "SUZUKI", "FORD",
        "BMW", "AUDI", "VOLVO", "JEEP", "HONDA", "MERCEDES", "LAND ROVER",
        "GREAT WALL", "ALFA ROMEO",
    }

    @classmethod
    def _title_contradicts_forced_brand(
        cls, titulo: str, forced_marca: Optional[str]
    ) -> bool:
        """Detecta anuncios patrocinados o cruzados cuyo título corresponde a otra marca."""
        norm_forced = cls._normalize_brand_name(forced_marca)
        if not norm_forced or not titulo:
            return False

        clean_title = re.sub(r"[^\w\s\-]", " ", titulo).upper().strip()
        tokens = [t for t in clean_title.split() if not re.match(r"^(19|20)\d{2}$", t)]
        if not tokens:
            return False

        forced_core = norm_forced.split("-")[0].split()[0]
        if norm_forced in clean_title or (len(forced_core) >= 3 and forced_core in clean_title):
            return False

        all_known = set(cls.KNOWN_BRANDS) | cls._EXTRA_KNOWN_BRANDS
        two_tok = f"{tokens[0]} {tokens[1]}" if len(tokens) >= 2 else ""
        if two_tok in all_known and cls._normalize_brand_name(two_tok) != norm_forced:
            return True
        if tokens[0] in all_known and cls._normalize_brand_name(tokens[0]) != norm_forced:
            return True
        return False

    @classmethod
    def _infer_brand_model_from_title(
        cls,
        titulo: str,
        forced_marca: Optional[str] = None,
        forced_modelo: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Infiere Marca y Modelo normalizados desde el título o usa el filtro segmentado."""
        norm_forced_marca = cls._normalize_brand_name(forced_marca)
        norm_forced_modelo = cls._normalize_model_name(forced_modelo)
        if norm_forced_marca and norm_forced_modelo:
            return norm_forced_marca, norm_forced_modelo

        clean_title = re.sub(r"[^\w\s\-]", " ", titulo or "").upper().strip()
        tokens = [t for t in clean_title.split() if not re.match(r"^(19|20)\d{2}$", t)]

        marca = norm_forced_marca
        modelo = norm_forced_modelo

        if not marca and tokens:
            if len(tokens) >= 2 and f"{tokens[0]} {tokens[1]}" in cls.KNOWN_BRANDS:
                marca = cls._normalize_brand_name(f"{tokens[0]} {tokens[1]}")
                tokens = tokens[2:]
            else:
                marca = cls._normalize_brand_name(tokens[0])
                tokens = tokens[1:]
        elif marca and tokens and cls._normalize_brand_name(tokens[0]) == marca:
            tokens = tokens[1:]

        if not modelo and tokens:
            if len(tokens) >= 2:
                pair_space = f"{tokens[0]} {tokens[1]}"
                pair_hyphen = f"{tokens[0]}-{tokens[1]}"
                pair_compact = f"{tokens[0]}{tokens[1]}"
                if pair_space in config.MODEL_ALIASES or pair_hyphen in config.MODEL_ALIASES or pair_compact in config.MODEL_ALIASES:
                    modelo = cls._normalize_model_name(pair_space)
                    tokens = tokens[2:]
                else:
                    modelo = cls._normalize_model_name(tokens[0])
            else:
                modelo = cls._normalize_model_name(tokens[0])

        return (marca or "DESCONOCIDA"), (modelo or "GENERICO")

    _AUTO_ONLY_BRANDS = {
        "LEXUS", "PORSCHE", "TESLA", "INFINITI", "LAND ROVER", "JAGUAR",
    }
    _AUTO_ONLY_MODELS = {
        ("VOLVO", "XC60"), ("VOLVO", "XC90"), ("VOLVO", "XC40"),
        ("VOLVO", "S60"), ("VOLVO", "S90"), ("VOLVO", "V60"), ("VOLVO", "V90"),
        ("BMW", "X3"), ("BMW", "X4"), ("BMW", "X5"), ("BMW", "X6"),
        ("MERCEDES-BENZ", "GLC"), ("MERCEDES-BENZ", "GLE"),
        ("MERCEDES-BENZ", "GLB"), ("MERCEDES-BENZ", "GLA"),
        ("AUDI", "Q5"), ("AUDI", "Q7"), ("AUDI", "Q8"),
    }

    @staticmethod
    def _estimate_fallback_km(anio: int) -> int:
        """Estima un kilometraje de respaldo acorde a la antigüedad cuando el portal no lo informa."""
        age = max(1, datetime.now().year - int(anio or 2019))
        return min(260_000, max(12_000, age * 14_000))

    @classmethod
    def _infer_fuel_and_trans(
        cls,
        text_blob: str,
        default_trans: str = "Manual",
        marca: Optional[str] = None,
        modelo: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Detecta combustible y transmisión desde el texto de versión/título/descripción."""
        blob = (text_blob or "").lower()
        norm_marca = cls._normalize_brand_name(marca) if marca else ""
        norm_modelo = cls._normalize_model_name(modelo) if modelo else ""

        auto_explicit = bool(
            re.search(
                r"\b(aut|autom[aá]tic[oa]|at[4-9]?|at10|[4-9]at|cvt|ecvt|dsg|dct|pdk|"
                r"tiptronic|triptronic|geartronic|steptronic|stronic|s-tronic|"
                r"multitronic|powershift|7g-?tronic|9g-?tronic)\b",
                blob,
            )
            or re.search(
                r"\bauto\b(?:\s+(?:4wd|awd|4x4|4x2|2wd|5p|4p|fwd|rwd|diesel|bencina)\b|$)",
                blob,
            )
        )
        manual_explicit = bool(
            re.search(r"\b(man|manual|mec[aá]nic[oa]|mt|[5-6]mt|mt[5-6])\b", blob)
        )

        if auto_explicit:
            transmision = "Automática"
        elif manual_explicit:
            transmision = "Manual"
        elif norm_marca in cls._AUTO_ONLY_BRANDS or (norm_marca, norm_modelo) in cls._AUTO_ONLY_MODELS:
            transmision = "Automática"
        else:
            transmision = default_trans

        if re.search(r"\b(di[eé]sel|tdi|hdi|crdi|d4d|petr[oó]leo|bluehdi|cdi)\b", blob) or (
            norm_marca == "VOLVO" and re.search(r"\b(d[2-5])\b", blob)
        ):
            combustible = "Diésel"
        elif re.search(r"\b(h[ií]brid[oa]|hybrid|hev|phev|mhev|recharge)\b", blob) or (
            norm_marca == "VOLVO" and re.search(r"\bt8\b", blob)
        ):
            combustible = "Híbrido"
        elif re.search(r"\b(el[eé]ctric[oa]|ev)\b", blob):
            combustible = "Eléctrico"
        else:
            combustible = "Bencina"

        return combustible, transmision

    # ==========================================================================
    # PORTAL 1: AUTOCOSMOS CHILE (autocosmos.cl/auto/usado)
    # ==========================================================================
    def _parse_autocosmos_card(
        self,
        card: Tag,
        forced_marca: Optional[str] = None,
        forced_modelo: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Parsea un <article class='card listing-card'> de Autocosmos Chile con Try/Except."""
        try:
            link_tag = card.select_one("a[itemprop='url'], a[href*='/auto/usado/']")
            if not link_tag or not link_tag.get("href"):
                return None
            href = link_tag["href"].strip()
            url = urljoin("https://www.autocosmos.cl", href)
            slug_id = href.rstrip("/").split("/")[-1]
            id_aviso = f"AC-{slug_id[:16].upper()}"
        except Exception:
            return None

        marca = ""
        modelo = ""
        version = ""
        try:
            b_tag = card.select_one(".listing-card__brand, [itemprop='brand']")
            m_tag = card.select_one(".listing-card__model")
            v_tag = card.select_one(".listing-card__version")
            if b_tag:
                marca = b_tag.get_text(strip=True).upper()
            if m_tag:
                modelo = m_tag.get_text(strip=True).upper()
            if v_tag:
                version = v_tag.get_text(" ", strip=True)
        except Exception:
            pass

        if forced_marca:
            marca = forced_marca.strip().upper()
        if forced_modelo:
            modelo = forced_modelo.strip().upper()

        anio = 2018
        try:
            y_tag = card.select_one(".listing-card__year, [itemprop='modelDate']")
            if y_tag:
                anio = self._clean_int(y_tag.get_text(), default=2018)
        except Exception:
            pass

        kilometraje = 60_000
        try:
            km_tag = card.select_one(".listing-card__km, [itemprop='mileageFromOdometer']")
            if km_tag:
                raw_km = km_tag.get("content") or km_tag.get_text()
                kilometraje = self._clean_int(raw_km, default=60_000)
        except Exception:
            pass

        precio = 0
        try:
            p_tag = card.select_one(".listing-card__price-value, [itemprop='price']")
            if p_tag:
                raw_p = p_tag.get("content") or p_tag.get_text()
                precio = self._clean_int(raw_p, default=0)
        except Exception:
            pass

        comuna = "Santiago"
        try:
            c_tag = card.select_one(".listing-card__city, [itemprop='addressLocality']")
            if c_tag:
                comuna = c_tag.get_text(strip=True).replace("|", "").strip()
        except Exception:
            pass

        descripcion = ""
        try:
            meta_desc = card.select_one("meta[itemprop='description']")
            if meta_desc and meta_desc.get("content"):
                descripcion = meta_desc["content"].strip()
        except Exception:
            pass

        titulo = f"{marca.title()} {modelo} {version} {anio}".strip()
        combustible, transmision = self._infer_fuel_and_trans(
            f"{titulo} {descripcion}", marca=marca, modelo=modelo
        )

        return {
            "id_aviso": id_aviso,
            "url": url,
            "marca": marca or "DESCONOCIDA",
            "modelo": modelo or "GENERICO",
            "anio": anio,
            "kilometraje": kilometraje,
            "combustible": combustible,
            "transmision": transmision,
            "comuna": comuna or "Santiago",
            "titulo": titulo,
            "descripcion": f"[Autocosmos Chile] {descripcion} {version}".strip(),
            "precio": precio,
        }

    def fetch_autocosmos(
        self,
        marca: Optional[str] = None,
        modelo: Optional[str] = None,
        max_pages: int = 2,
        anio_min: Optional[int] = None,
        anio_max: Optional[int] = None,
        precio_min: Optional[int] = None,
        precio_max: Optional[int] = None,
        km_max: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Extrae anuncios reales desde Autocosmos Chile (general o segmentado por marca/modelo y filtros)."""
        results: List[Dict[str, Any]] = []
        base_url = config.AUTOCOSMOS_BASE_URL
        if marca:
            m_slug = marca.strip().lower().replace(" ", "-")
            base_url = f"{base_url}/{m_slug}"
            if modelo:
                mod_slug = modelo.strip().lower().replace(" ", "-")
                base_url = f"{base_url}/{mod_slug}"

        query_params: Dict[str, Any] = {}
        if anio_min:
            query_params["fye"] = anio_min
        if anio_max:
            query_params["tye"] = anio_max
        if precio_min:
            query_params["fpr"] = precio_min
        if precio_max:
            query_params["tpr"] = precio_max

        for page in range(1, max_pages + 1):
            page_params = dict(query_params)
            if page > 1:
                page_params["pidx"] = page
            try:
                resp = self.session.get(
                    base_url,
                    params=page_params if page_params else None,
                    headers=self._build_headers(),
                    timeout=config.SCRAPER_TIMEOUT_SECONDS,
                )
                if resp.status_code != 200:
                    break
                soup = BeautifulSoup(resp.text, "html.parser")
                cards = soup.select("article.card, article.listing-card, article[itemtype*='Car']")
                if not cards:
                    break
                for card in cards:
                    parsed = self._parse_autocosmos_card(card, forced_marca=marca, forced_modelo=modelo)
                    if not parsed or parsed["precio"] < config.MIN_VALID_PRICE_CLP:
                        continue
                    if anio_min and parsed["anio"] < anio_min:
                        continue
                    if anio_max and parsed["anio"] > anio_max:
                        continue
                    if precio_min and parsed["precio"] < precio_min:
                        continue
                    if precio_max and parsed["precio"] > precio_max:
                        continue
                    if km_max and parsed["kilometraje"] > km_max:
                        continue
                    results.append(parsed)
                if page < max_pages:
                    time.sleep(random.uniform(config.SCRAPER_MIN_DELAY_SEC, config.SCRAPER_MAX_DELAY_SEC))
            except Exception as exc:
                logger.debug("Aviso al consultar Autocosmos (%s): %s", base_url, exc)
                break

        return results

    # ==========================================================================
    # PORTAL 2: MERCADOLIBRE AUTOS CHILE (autos.mercadolibre.cl)
    # ==========================================================================
    def _parse_mercadolibre_card(
        self,
        card: Tag,
        forced_marca: Optional[str] = None,
        forced_modelo: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Parsea una tarjeta 'div.poly-card' o 'li.ui-search-layout__item' de MercadoLibre Autos Chile."""
        try:
            link_tag = card.find("a", href=True)
            if not link_tag:
                return None
            raw_href = link_tag["href"].split("#")[0].strip()
            href_lower = raw_href.lower()
            # Descartar anuncios patrocinados / banners publicitarios de MercadoLibre
            if (
                "click1.mercadolibre" in href_lower
                or "/brand_ads/" in href_lower
                or "/mclics/" in href_lower
                or "/publicidad/" in href_lower
            ):
                return None
            raw_href = raw_href.split("?")[0].strip()
            id_match = re.search(r"(MLC-?\d+)", raw_href, flags=re.IGNORECASE)
            if id_match:
                clean_id = id_match.group(1).upper().replace("MLC", "MLC-").replace("--", "-")
            else:
                clean_id = "MLC-" + hashlib.md5(raw_href.encode("utf-8")).hexdigest()[:10].upper()
        except Exception:
            return None

        text_parts = [t.strip() for t in card.stripped_strings if t.strip()]
        full_blob = " | ".join(text_parts)

        # Título
        titulo = ""
        try:
            title_el = card.select_one("h2, h3, .poly-component__title, .ui-search-item__title")
            if title_el:
                titulo = title_el.get_text(" ", strip=True)
            elif text_parts:
                titulo = text_parts[0]
        except Exception:
            titulo = text_parts[0] if text_parts else "Auto MercadoLibre"

        if self._title_contradicts_forced_brand(titulo, forced_marca):
            return None

        marca, modelo = self._infer_brand_model_from_title(
            titulo, forced_marca=forced_marca, forced_modelo=forced_modelo
        )

        # Precio en CLP (si viene en UF, convertir a CLP aproximado ~38.000 CLP/UF)
        precio = 0
        try:
            price_el = card.select_one(".andes-money-amount__fraction")
            curr_el = card.select_one(".andes-money-amount__currency-symbol")
            if price_el:
                val = self._clean_int(price_el.get_text(), default=0)
                curr = curr_el.get_text(strip=True).upper() if curr_el else "$"
                if "UF" in curr or val < 10_000:
                    precio = val * 38_000
                else:
                    precio = val
        except Exception:
            pass

        # Año y Kilometraje
        anio = 2018
        kilometraje: Optional[int] = None
        try:
            for part in text_parts:
                if re.match(r"^(19[9]\d|20[0-2]\d)$", part):
                    anio = int(part)
                elif "km" in part.lower() and "/l" not in part.lower() and "/h" not in part.lower():
                    km_val = self._clean_int(part, default=-1)
                    if 0 <= km_val <= 600_000:
                        kilometraje = km_val
        except Exception:
            pass
        if kilometraje is None:
            kilometraje = self._estimate_fallback_km(anio)

        # Comuna / Ubicación (suele ser el último elemento con '-')
        comuna = "Santiago"
        try:
            loc_el = card.select_one(".poly-component__location, .ui-search-item__location")
            if loc_el:
                raw_loc = loc_el.get_text(strip=True)
                comuna = raw_loc.split("-")[0].strip()
            elif text_parts:
                last_part = text_parts[-1]
                if "-" in last_part or len(last_part) < 35:
                    comuna = last_part.split("-")[0].strip()
        except Exception:
            pass

        combustible, transmision = self._infer_fuel_and_trans(
            full_blob, marca=marca, modelo=modelo
        )

        return {
            "id_aviso": clean_id,
            "url": raw_href,
            "marca": marca,
            "modelo": modelo,
            "anio": anio,
            "kilometraje": kilometraje,
            "combustible": combustible,
            "transmision": transmision,
            "comuna": comuna or "Santiago",
            "titulo": titulo,
            "descripcion": f"[MercadoLibre Autos Chile] {full_blob}",
            "precio": precio,
        }

    def fetch_mercadolibre(
        self,
        marca: Optional[str] = None,
        modelo: Optional[str] = None,
        max_pages: int = 2,
        anio_min: Optional[int] = None,
        anio_max: Optional[int] = None,
        precio_min: Optional[int] = None,
        precio_max: Optional[int] = None,
        km_max: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Extrae anuncios reales desde MercadoLibre Autos Chile (con ruta jerárquica y filtros aplicados)."""
        results: List[Dict[str, Any]] = []
        norm_marca = self._normalize_brand_name(marca) if marca else None
        candidate_bases: List[str] = []

        year_part = ""
        if anio_min and anio_max:
            year_part = f"{anio_min}-{anio_max}/"
        elif anio_min:
            year_part = f"desde-{anio_min}/"
        elif anio_max:
            year_part = f"hasta-{anio_max}/"

        price_part = ""
        if precio_min and precio_max:
            price_part = f"_PriceRange_{precio_min}CLP-{precio_max}CLP"
        elif precio_min:
            price_part = f"_PriceRange_{precio_min}CLP-0CLP"
        elif precio_max:
            price_part = f"_PriceRange_0CLP-{precio_max}CLP"

        if norm_marca and modelo:
            m_slug = norm_marca.strip().lower().replace(" ", "-")
            mod_slug = modelo.strip().lower().replace(" ", "-")
            candidate_bases = [
                f"{config.MERCADOLIBRE_BASE_URL}/{m_slug}-{mod_slug}/{year_part}",
                f"{config.MERCADOLIBRE_BASE_URL}/{m_slug}/{mod_slug}/{year_part}",
            ]
        elif norm_marca:
            m_slug = norm_marca.strip().lower().replace(" ", "-")
            candidate_bases = [f"{config.MERCADOLIBRE_BASE_URL}/{m_slug}/{year_part}"]
        else:
            candidate_bases = [f"{config.MERCADOLIBRE_BASE_URL}/autos-camionetas/{year_part}"]

        active_base_url = candidate_bases[0]

        for page in range(1, max_pages + 1):
            offset = (page - 1) * 48 + 1
            if page == 1:
                url = f"{active_base_url.rstrip('/')}/{price_part}".rstrip("/") if price_part else active_base_url
            else:
                url = f"{active_base_url.rstrip('/')}/_Desde_{offset}{price_part}_NoIndex_True"
            try:
                resp = self.session.get(url, headers=self._build_headers(), timeout=config.SCRAPER_TIMEOUT_SECONDS)
                cards = []
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    cards = soup.select("div.poly-card, div.ui-search-result")

                # Si la primera URL candidata en página 1 dio pocos resultados y existe segunda candidata, probarla
                if page == 1 and len(cards) < 8 and len(candidate_bases) > 1:
                    alt_url_base = candidate_bases[1]
                    alt_url = f"{alt_url_base.rstrip('/')}/{price_part}".rstrip("/") if price_part else alt_url_base
                    resp_alt = self.session.get(alt_url, headers=self._build_headers(), timeout=config.SCRAPER_TIMEOUT_SECONDS)
                    if resp_alt.status_code == 200:
                        soup_alt = BeautifulSoup(resp_alt.text, "html.parser")
                        cards_alt = soup_alt.select("div.poly-card, div.ui-search-result")
                        if len(cards_alt) > len(cards):
                            cards = cards_alt
                            active_base_url = alt_url_base

                if not cards:
                    break
                for card in cards:
                    parsed = self._parse_mercadolibre_card(card, forced_marca=norm_marca, forced_modelo=modelo)
                    if not parsed or parsed["precio"] < config.MIN_VALID_PRICE_CLP:
                        continue
                    if anio_min and parsed["anio"] < anio_min:
                        continue
                    if anio_max and parsed["anio"] > anio_max:
                        continue
                    if precio_min and parsed["precio"] < precio_min:
                        continue
                    if precio_max and parsed["precio"] > precio_max:
                        continue
                    if km_max and parsed["kilometraje"] > km_max:
                        continue
                    results.append(parsed)
                if page < max_pages:
                    time.sleep(random.uniform(config.SCRAPER_MIN_DELAY_SEC, config.SCRAPER_MAX_DELAY_SEC))
            except Exception as exc:
                logger.debug("Aviso al consultar MercadoLibre (%s): %s", url, exc)
                break

        return results

    # ==========================================================================
    # PORTAL 3: YAPO AUTOS CHILE (yapo.cl/autos-usados)
    # ==========================================================================
    def _parse_yapo_card(
        self,
        card: Tag,
        forced_marca: Optional[str] = None,
        forced_modelo: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Parsea un enlace de tarjeta 'a.item-card-link' de Yapo.cl con descripción completa."""
        try:
            href = card.get("href", "").strip()
            if not href or "/autos-usados/" not in href:
                return None
            url = urljoin("https://www.yapo.cl", href)
            id_match = re.search(r"/(\d{6,10})$", href)
            id_aviso = f"YAPO-{id_match.group(1)}" if id_match else "YAPO-" + hashlib.md5(url.encode()).hexdigest()[:10].upper()
        except Exception:
            return None

        text_parts = [t.strip() for t in card.stripped_strings if t.strip()]
        full_blob = " | ".join(text_parts)

        # Precio
        precio = 0
        try:
            for part in text_parts:
                if "$" in part and any(ch.isdigit() for ch in part):
                    val = self._clean_int(part, default=0)
                    if val >= config.MIN_VALID_PRICE_CLP:
                        precio = val
                        break
        except Exception:
            pass

        # Año, Kilometraje y Comuna
        anio = 2018
        kilometraje = 60_000
        comuna = "Santiago"
        titulo = ""
        descripcion = ""

        try:
            for idx, part in enumerate(text_parts):
                if re.match(r"^(19[9]\d|20[0-2]\d)$", part):
                    anio = int(part)
                    if idx > 0 and "," in text_parts[idx - 1]:
                        comuna = text_parts[idx - 1].split(",")[-1].strip()
                elif re.search(r"\d+\s*km", part, flags=re.IGNORECASE):
                    km_val = self._clean_int(part, default=60_000)
                    if 0 <= km_val <= 600_000:
                        kilometraje = km_val
                elif len(part) > 65 and not descripcion:
                    descripcion = part
                elif 15 <= len(part) <= 65 and not titulo and not any(
                    skip in part.lower() for skip in ("previous", "next", "slide", "whatsapp", "contactar")
                ):
                    if any(b.lower() in part.lower() for b in self.KNOWN_BRANDS) or forced_marca:
                        titulo = part
        except Exception:
            pass

        if not titulo:
            slug_title = href.split("/autos-usados/")[-1].split("/")[0].replace("-", " ").title()
            titulo = slug_title

        marca, modelo = self._infer_brand_model_from_title(
            titulo, forced_marca=forced_marca, forced_modelo=forced_modelo
        )
        combustible, transmision = self._infer_fuel_and_trans(
            f"{titulo} {descripcion} {full_blob}", marca=marca, modelo=modelo
        )

        return {
            "id_aviso": id_aviso,
            "url": url,
            "marca": marca,
            "modelo": modelo,
            "anio": anio,
            "kilometraje": kilometraje,
            "combustible": combustible,
            "transmision": transmision,
            "comuna": comuna or "Santiago",
            "titulo": titulo,
            "descripcion": f"[Yapo Chile] {descripcion or titulo}",
            "precio": precio,
        }

    def fetch_yapo(
        self,
        marca: Optional[str] = None,
        modelo: Optional[str] = None,
        max_pages: int = 2,
        anio_min: Optional[int] = None,
        anio_max: Optional[int] = None,
        precio_min: Optional[int] = None,
        precio_max: Optional[int] = None,
        km_max: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Extrae anuncios reales desde Yapo Autos Chile."""
        results: List[Dict[str, Any]] = []
        query = " ".join(x for x in (marca, modelo) if x).strip()

        for page in range(1, max_pages + 1):
            if query:
                q_encoded = quote_plus(query)
                url = f"{config.YAPO_BASE_URL}?q={q_encoded}" if page == 1 else f"{config.YAPO_BASE_URL}.{page}?q={q_encoded}"
            else:
                url = config.YAPO_BASE_URL if page == 1 else f"{config.YAPO_BASE_URL}.{page}"

            try:
                resp = self.session.get(url, headers=self._build_headers(), timeout=config.SCRAPER_TIMEOUT_SECONDS)
                if resp.status_code != 200:
                    break
                soup = BeautifulSoup(resp.text, "html.parser")
                cards = soup.select("a.item-card-link")
                if not cards:
                    break
                for card in cards:
                    parsed = self._parse_yapo_card(card, forced_marca=marca, forced_modelo=modelo)
                    if not parsed or parsed["precio"] < config.MIN_VALID_PRICE_CLP:
                        continue
                    if marca and marca.upper() not in (parsed["titulo"] + " " + parsed["descripcion"]).upper():
                        continue
                    if anio_min and parsed["anio"] < anio_min:
                        continue
                    if anio_max and parsed["anio"] > anio_max:
                        continue
                    if precio_min and parsed["precio"] < precio_min:
                        continue
                    if precio_max and parsed["precio"] > precio_max:
                        continue
                    if km_max and parsed["kilometraje"] > km_max:
                        continue
                    results.append(parsed)
                if page < max_pages:
                    time.sleep(random.uniform(config.SCRAPER_MIN_DELAY_SEC, config.SCRAPER_MAX_DELAY_SEC))
            except Exception as exc:
                logger.debug("Aviso al consultar Yapo (%s): %s", url, exc)
                break

        return results

    # ==========================================================================
    # PORTAL 4: CHILEAUTOS PARSER (Compatible con HTML directo y tests)
    # ==========================================================================
    def _parse_single_card(self, card: Tag) -> Optional[Dict[str, Any]]:
        """Parser de tarjetas HTML de Chileautos con bloques Try/Except aislados."""
        id_aviso = None
        try:
            id_aviso = card.get("data-id") or card.get("data-webm-network-id") or card.get("id")
            if not id_aviso:
                link_el = card.select_one("a[href*='CL-AD-'], a[href*='/detalles/']")
                if link_el and link_el.get("href"):
                    match = re.search(r"(CL-AD-\d+|SSE-AD-\d+|\d{7,10})", link_el["href"])
                    if match:
                        id_aviso = match.group(1)
        except Exception:
            pass

        url = ""
        try:
            link_tag = card.select_one("a[href*='/detalles/'], h3 a, a.listing-item__title")
            if link_tag and link_tag.get("href"):
                url = urljoin(config.CHILEAUTOS_BASE_URL, link_tag["href"].strip())
            if not id_aviso and url:
                id_aviso = "CL-HASH-" + hashlib.md5(url.encode("utf-8")).hexdigest()[:10].upper()
        except Exception:
            pass

        if not id_aviso:
            return None

        titulo = ""
        try:
            title_tag = card.select_one("h3, .listing-item__title, [data-webm-clickvalue='sv-title']")
            if title_tag:
                titulo = " ".join(title_tag.get_text(" ", strip=True).split())
        except Exception:
            pass

        marca = (card.get("data-webm-make") or card.get("data-make") or "").strip().upper()
        modelo = (card.get("data-webm-model") or card.get("data-model") or "").strip().upper()
        if not marca or not modelo:
            marca, modelo = self._infer_brand_model_from_title(titulo, marca or None, modelo or None)

        anio = 2018
        try:
            year_match = re.search(r"\b(19[89]\d|20[0-2]\d)\b", titulo)
            if year_match:
                anio = int(year_match.group(1))
        except Exception:
            pass

        precio = 0
        try:
            raw_price = card.get("data-webm-price") or card.get("data-price")
            if raw_price:
                precio = self._clean_int(raw_price, default=0)
            else:
                price_tag = card.select_one(".price, .listing-item__price")
                if price_tag:
                    precio = self._clean_int(price_tag.get_text(), default=0)
        except Exception:
            pass

        details_blob = card.get_text(" ", strip=True)
        kilometraje = 0
        km_match = re.search(r"([\d\.]+)\s*km", details_blob, flags=re.IGNORECASE)
        if km_match:
            kilometraje = self._clean_int(km_match.group(1), default=0)

        combustible, transmision = self._infer_fuel_and_trans(details_blob)

        comuna = "Santiago"
        try:
            loc_tag = card.select_one(".seller-location, .listing-item__location")
            if loc_tag:
                parts = [p.strip() for p in re.split(r"[-|,]", loc_tag.get_text(" ", strip=True)) if p.strip()]
                if parts:
                    comuna = parts[-1]
        except Exception:
            pass

        descripcion = titulo
        try:
            desc_tag = card.select_one(".listing-item__description, p.description")
            if desc_tag:
                descripcion = " ".join(desc_tag.get_text(" ", strip=True).split())
        except Exception:
            pass

        return {
            "id_aviso": str(id_aviso).strip(),
            "url": url or config.CHILEAUTOS_BASE_URL,
            "marca": marca,
            "modelo": modelo,
            "anio": anio,
            "kilometraje": kilometraje,
            "combustible": combustible,
            "transmision": transmision,
            "comuna": comuna,
            "titulo": titulo or f"{marca} {modelo} {anio}",
            "descripcion": descripcion,
            "precio": precio,
        }

    def parse_html_page(self, html_content: str) -> List[Dict[str, Any]]:
        """Extrae tarjetas HTML de Chileautos."""
        soup = BeautifulSoup(html_content, "html.parser")
        extracted: List[Dict[str, Any]] = []
        for selector in ("div.listing-item.card", "div.listing-item", "article.listing-item", "div[data-webm-network-id]"):
            cards = soup.select(selector)
            if cards:
                for c in cards:
                    parsed = self._parse_single_card(c)
                    if parsed:
                        extracted.append(parsed)
                break
        return extracted

    def import_local_chileautos_html(
        self,
        marca_filter: Optional[str] = None,
        modelo_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Lee páginas HTML de Chileautos guardadas manualmente por el usuario desde su
        navegador (Ctrl+S) en la carpeta 'data/import_chileautos/'.
        """
        imported: List[Dict[str, Any]] = []
        import_dir = config.CHILEAUTOS_IMPORT_DIR
        if not import_dir.exists():
            return imported

        html_files = list(import_dir.glob("*.html")) + list(import_dir.glob("*.htm"))
        for html_file in html_files:
            try:
                content = html_file.read_text(encoding="utf-8", errors="ignore")
                items = self.parse_html_page(content)
                for item in items:
                    if marca_filter and item["marca"].upper() != self._normalize_brand_name(marca_filter):
                        continue
                    if modelo_filter and item["modelo"].upper() != modelo_filter.strip().upper():
                        continue
                    if item["precio"] >= config.MIN_VALID_PRICE_CLP:
                        imported.append(item)
                logger.info("Importados %d avisos desde archivo local Chileautos: %s", len(items), html_file.name)
            except Exception as exc:
                logger.warning("Error leyendo archivo HTML local %s: %s", html_file, exc)

        return imported

    def fetch_chileautos_via_google_cse(
        self,
        marca: Optional[str] = None,
        modelo: Optional[str] = None,
        anio_min: Optional[int] = None,
        anio_max: Optional[int] = None,
        precio_min: Optional[int] = None,
        precio_max: Optional[int] = None,
        km_max: Optional[int] = None,
        deep: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Consulta fichas de Chileautos indexadas en Google mediante Serper.dev API
        (SERPER_API_KEY) o Google Custom Search JSON API (GOOGLE_CSE_API_KEY).
        Aplica particionamiento por año (Year-Slicing) para superar el límite de 10-20
        resultados por consulta de Google y extraer decenas de publicaciones reales.
        """
        serper_key = (getattr(config, "SERPER_API_KEY", "") or "").strip()
        api_key = (getattr(config, "GOOGLE_CSE_API_KEY", "") or "").strip()
        cx = (getattr(config, "GOOGLE_CSE_CX", "") or "").strip()

        base_term = "site:chileautos.cl/vehiculos/detalles/"
        target_brand = (marca or "").strip()
        target_model = (modelo or "").strip()
        target_prefix = f"{base_term} {target_brand} {target_model}".strip()

        queries_to_run: List[str] = []
        if anio_min and anio_max:
            for yr in range(int(anio_min), int(anio_max) + 1):
                queries_to_run.append(f"{target_prefix} {yr}".strip())
        elif deep:
            for yr in [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016]:
                queries_to_run.append(f"{target_prefix} {yr}".strip())
        elif target_model:
            queries_to_run.append(target_prefix)
            for yr in [2022, 2021, 2020, 2019]:
                queries_to_run.append(f"{target_prefix} {yr}".strip())
        else:
            queries_to_run.append(target_prefix)

        raw_entries: List[Dict[str, str]] = []
        seen_links: Set[str] = set()

        # 1. Intentar primero vía Serper.dev (Google Search API sin bloqueo)
        if serper_key:
            for q in queries_to_run:
                pages_limit = 1 if len(queries_to_run) > 3 else min(2, config.SERPER_PAGES_PER_QUERY)
                for page in range(1, pages_limit + 1):
                    try:
                        resp_s = self.session.post(
                            "https://google.serper.dev/search",
                            headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                            json={"q": q, "gl": "cl", "hl": "es", "num": 10, "page": page},
                            timeout=config.SCRAPER_TIMEOUT_SECONDS,
                        )
                        if resp_s.status_code != 200:
                            logger.warning("Serper respondió HTTP %s para '%s': %s", resp_s.status_code, q, resp_s.text[:120])
                            break
                        organic = resp_s.json().get("organic", [])
                        new_items = 0
                        for item in organic:
                            link = item.get("link", "")
                            if link and link not in seen_links:
                                seen_links.add(link)
                                new_items += 1
                                raw_entries.append(
                                    {
                                        "link": link,
                                        "title": item.get("title", ""),
                                        "snippet": item.get("snippet", ""),
                                    }
                                )
                        if new_items == 0:
                            break
                    except Exception as exc:
                        logger.debug("Aviso en consulta Serper para Chileautos ('%s'): %s", q, exc)
                        break
                time.sleep(0.15)

        # 2. Intentar vía Google Custom Search API si no hubo entradas de Serper
        if not raw_entries and api_key and cx:
            for q in queries_to_run[:2]:
                try:
                    resp = self.session.get(
                        "https://www.googleapis.com/customsearch/v1",
                        params={"key": api_key, "cx": cx, "q": q, "num": 10},
                        timeout=config.SCRAPER_TIMEOUT_SECONDS,
                    )
                    if resp.status_code == 200:
                        for item in resp.json().get("items", []):
                            raw_entries.append(
                                {
                                    "link": item.get("link", ""),
                                    "title": item.get("title", ""),
                                    "snippet": item.get("snippet", ""),
                                }
                            )
                except Exception as exc:
                    logger.debug("Aviso en consulta Google CSE para Chileautos: %s", exc)

        results: List[Dict[str, Any]] = []
        for entry in raw_entries:
            link = entry.get("link", "")
            title = entry.get("title", "")
            snippet = entry.get("snippet", "")
            if "chileautos.cl" not in link:
                continue

            id_match = re.search(r"(CL-AD-\d+|SSE-AD-\d+)", link)
            id_aviso = (
                id_match.group(1)
                if id_match
                else "CL-IDX-" + hashlib.md5(link.encode()).hexdigest()[:10].upper()
            )

            blob = f"{title} {snippet}"
            # Formato típico del snippet: "Precio. $17.900.000 CLP ; Kilometraje. 7.000km"
            price_match = re.search(r"Precio\.?\s*\$\s*([\d\.]{6,12})", blob) or re.search(
                r"\$\s*([\d\.]{7,12})", blob
            )
            precio = self._clean_int(price_match.group(1), default=0) if price_match else 0
            if precio < config.MIN_VALID_PRICE_CLP:
                continue
            if precio_min and precio < precio_min:
                continue
            if precio_max and precio > precio_max:
                continue

            # El año va al inicio del título en Chileautos ("2026 Mazda 3 ...")
            year_match = re.match(r"^\s*(19[9]\d|20[0-2]\d)\b", title) or re.search(
                r"\b(19[9]\d|20[0-2]\d)\b", blob
            )
            anio = int(year_match.group(1)) if year_match else 2019
            if anio_min and anio < anio_min:
                continue
            if anio_max and anio > anio_max:
                continue

            km_match = re.search(
                r"Kilometraje\.?\s*([\d\.]+)\s*km(?!\s*/)", blob, flags=re.IGNORECASE
            ) or re.search(
                r"([\d\.]+)\s*(?:km|kil[oó]metros)(?!\s*/)", blob, flags=re.IGNORECASE
            )
            fallback_km = self._estimate_fallback_km(anio)
            kilometraje = (
                self._clean_int(km_match.group(1), default=fallback_km)
                if km_match
                else fallback_km
            )
            if km_max and kilometraje > km_max:
                continue

            m_norm, mod_norm = self._infer_brand_model_from_title(title, marca, modelo)
            combustible, transmision = self._infer_fuel_and_trans(
                blob, marca=m_norm, modelo=mod_norm
            )
            # Códigos de caja típicos de Chileautos: "6MT", "6AT", "AT", "CVT"
            if re.search(r"\d(AT)\b|\bCVT\b", title, flags=re.IGNORECASE):
                transmision = "Automática"
            elif re.search(r"\d(MT)\b", title, flags=re.IGNORECASE):
                transmision = "Manual"

            comuna_detected = "Santiago"
            for c_name in config.COMUNA_WEALTH_TIER.keys():
                if re.search(r"\b" + re.escape(c_name.title()) + r"\b", blob, flags=re.IGNORECASE):
                    comuna_detected = c_name.title()
                    break

            results.append(
                {
                    "id_aviso": id_aviso,
                    "url": link,
                    "marca": m_norm,
                    "modelo": mod_norm,
                    "anio": anio,
                    "kilometraje": kilometraje,
                    "combustible": combustible,
                    "transmision": transmision,
                    "comuna": comuna_detected,
                    "titulo": title,
                    "descripcion": f"[Chileautos via Google Index] {snippet}",
                    "precio": precio,
                }
            )

        return results

    @staticmethod
    def merge_and_deduplicate_cross_platform(listings: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """
        Detecta únicamente cuando el MISMO vehículo físico exacto está publicado más de una vez,
        evitando estrictamente eliminar autos distintos que solo coincidan en año o km redondo.
        Reglas obligatorias simultáneas para considerar que dos avisos son el mismo auto físico:
        1. Misma Marca, mismo Modelo y mismo Año exacto.
        2. Misma Transmisión, mismo Combustible y misma Comuna exacta.
        3. Kilometraje de odómetro NO redondo (km % 100 != 0, ej. 68.134 km o 21.931 km)
           y 100% idéntico al dígito (km_1 == km_2) sin redondear.
        4. Diferencia de precio entre ambas publicaciones <= 12%.
        Cuando se cumplen las 4 reglas, conserva la plataforma más barata y suma las descripciones.
        """
        grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        untouched_listings: List[Dict[str, Any]] = []

        for item in listings:
            marca = str(item.get("marca", "")).strip().upper()
            modelo = str(item.get("modelo", "")).strip().upper()
            anio = int(item.get("anio") or 2018)
            km = int(item.get("kilometraje") or 0)
            trans = str(item.get("transmision", "Manual")).strip().title()
            comb = str(item.get("combustible", "Bencina")).strip().title()
            comuna = str(item.get("comuna", "Santiago")).strip().upper()

            # Si el kilometraje es un número redondo (múltiplo de 100 o 1.000, ej. 60.000 km),
            # NUNCA lo fusionamos con otro aviso porque dos autos distintos pueden tener 60.000 km.
            if km <= 0 or (km % 100 == 0):
                untouched_listings.append(item)
                continue

            # Huella ultra-estricta: km exacto al dígito + marca + modelo + año + trans + comb + comuna
            strict_fingerprint = (marca, modelo, anio, km, trans, comb, comuna)
            grouped.setdefault(strict_fingerprint, []).append(item)

        merged_listings: List[Dict[str, Any]] = list(untouched_listings)
        cross_portal_matches = 0

        for _, group in grouped.items():
            unique_by_id = {x["id_aviso"]: x for x in group}
            candidates = list(unique_by_id.values())

            if len(candidates) == 1:
                merged_listings.append(candidates[0])
                continue

            valid_candidates = [c for c in candidates if (c.get("precio") or 0) > 0]
            if len(valid_candidates) <= 1:
                merged_listings.extend(candidates)
                continue

            valid_candidates.sort(key=lambda x: x["precio"])
            min_p = valid_candidates[0]["precio"]
            max_p = valid_candidates[-1]["precio"]

            # Si la diferencia de precio supera el 12%, podrían ser unidades/versiones distintas -> conservar ambos
            if (max_p - min_p) / max(1, min_p) > 0.12:
                merged_listings.extend(valid_candidates)
                continue

            cheapest = dict(valid_candidates[0])
            other_offers = valid_candidates[1:]
            cross_portal_matches += len(other_offers)

            other_summary = []
            combined_descriptions = [cheapest.get("descripcion", "")]
            for other in other_offers:
                diff = other["precio"] - cheapest["precio"]
                other_price_str = f"${other['precio']:,}".replace(",", ".")
                diff_str = f"+${diff:,}".replace(",", ".") if diff > 0 else "mismo precio"
                other_summary.append(f"{other['id_aviso']} a {other_price_str} ({diff_str})")
                if other.get("descripcion") and other["descripcion"] not in combined_descriptions:
                    combined_descriptions.append(other["descripcion"])

            multi_badge = (
                f"[Mismo auto en {len(valid_candidates)} avisos | "
                f"Más barato aquí vs {', '.join(other_summary)}] "
            )
            cheapest["descripcion"] = multi_badge + " // ".join(combined_descriptions)
            merged_listings.append(cheapest)

        return merged_listings, cross_portal_matches

    # ==========================================================================
    # ORQUESTADOR DE EXTRACCIÓN MULTI-PORTAL Y SEGMENTADA
    # ==========================================================================
    def fetch_segment_across_portals(
        self,
        marca: Optional[str] = None,
        modelo: Optional[str] = None,
        pages_per_portal: int = 1,
        anio_min: Optional[int] = None,
        anio_max: Optional[int] = None,
        precio_min: Optional[int] = None,
        precio_max: Optional[int] = None,
        km_max: Optional[int] = None,
        deep: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Consulta en paralelo controlado los portales públicos habilitados (MercadoLibre,
        Autocosmos, Yapo y Chileautos via Google/Serper) para una Marca/Modelo específico o catálogo general.
        Aplica filtros de año, precio y kilometraje a nivel de portal y extracción.
        """
        portal_results: Dict[str, List[Dict[str, Any]]] = {
            "MercadoLibre": [],
            "Autocosmos": [],
            "Yapo": [],
            "Chileautos": [],
        }

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    self.fetch_mercadolibre,
                    marca,
                    modelo,
                    pages_per_portal,
                    anio_min=anio_min,
                    anio_max=anio_max,
                    precio_min=precio_min,
                    precio_max=precio_max,
                    km_max=km_max,
                ): "MercadoLibre",
                executor.submit(
                    self.fetch_autocosmos,
                    marca,
                    modelo,
                    pages_per_portal,
                    anio_min=anio_min,
                    anio_max=anio_max,
                    precio_min=precio_min,
                    precio_max=precio_max,
                    km_max=km_max,
                ): "Autocosmos",
                executor.submit(
                    self.fetch_yapo,
                    marca,
                    modelo,
                    pages_per_portal,
                    anio_min=anio_min,
                    anio_max=anio_max,
                    precio_min=precio_min,
                    precio_max=precio_max,
                    km_max=km_max,
                ): "Yapo",
                executor.submit(
                    self.fetch_chileautos_via_google_cse,
                    marca,
                    modelo,
                    anio_min=anio_min,
                    anio_max=anio_max,
                    precio_min=precio_min,
                    precio_max=precio_max,
                    km_max=km_max,
                    deep=deep,
                ): "Chileautos",
            }
            for future in as_completed(futures):
                portal_name = futures[future]
                try:
                    items = future.result()
                    portal_results[portal_name] = items
                except Exception as exc:
                    logger.warning("Fallo controlado en portal %s: %s", portal_name, exc)

        return portal_results

    def run_daily_acquisition(
        self,
        max_pages: int = config.SCRAPER_MAX_PAGES,
        force_fallback: bool = False,
        simulate_lifecycle_Verify: bool = False,
        marca_filter: Optional[str] = None,
        modelo_filter: Optional[str] = None,
        segmented_sweep: bool = True,
        purge_simulated: bool = True,
        anio_min: Optional[int] = None,
        anio_max: Optional[int] = None,
        precio_min: Optional[int] = None,
        precio_max: Optional[int] = None,
        km_max: Optional[int] = None,
        deep: bool = False,
    ) -> Dict[str, Any]:
        """
        Ejecuta la adquisición de datos en tres modos posibles:
        1. Búsqueda Focalizada (--marca y/o --modelo): Escanea en vivo todos los portales
           específicamente para ese vehículo con filtros de año, precio y kilometraje.
        2. Barrido Multi-Segmento Automático (por defecto): Recorre las marcas y modelos
           principales de Chile (config.TARGET_SEGMENTS) en MercadoLibre, Autocosmos y Yapo
           recopilando un gran volumen de datos reales clasificados por modelo.
        3. Fallback Offline (--fallback): Usa el generador local si no hay conexión.
        """
        logger.info("=" * 72)
        logger.info("INICIANDO MÓDULO 1: MOTOR MULTI-PORTAL Y BUSCADOR SEGMENTADO (CHILE)")
        logger.info("=" * 72)

        dedup_map: Dict[str, Dict[str, Any]] = {}
        portal_counts = {"MercadoLibre": 0, "Autocosmos": 0, "Yapo": 0, "Chileautos": 0}
        used_fallback = False

        if not force_fallback:
            # 0. Importar cualquier HTML de Chileautos guardado localmente en data/import_chileautos/
            local_ca_items = self.import_local_chileautos_html(
                marca_filter=marca_filter, modelo_filter=modelo_filter
            )
            if local_ca_items:
                portal_counts["Chileautos"] += len(local_ca_items)
                for item in local_ca_items:
                    dedup_map[item["id_aviso"]] = item
            if marca_filter:
                # MODO A: BÚSQUEDA FOCALIZADA BAJO DEMANDA (Marca / Modelo específico)
                filter_desc = []
                if anio_min or anio_max:
                    filter_desc.append(f"Años: {anio_min or 'min'}..{anio_max or 'max'}")
                if precio_min or precio_max:
                    filter_desc.append(f"Precio: ${precio_min or 0:,}..${precio_max or 'max':,}".replace(",", "."))
                if km_max:
                    filter_desc.append(f"Km Max: {km_max:,}".replace(",", "."))
                if deep:
                    filter_desc.append("MODO DEEP/AÑO-POR-AÑO ACTIVO")

                logger.info(
                    "Ejecutando BÚSQUEDA FOCALIZADA en múltiples portales para: Marca='%s' | Modelo='%s' (Páginas=%d%s)",
                    marca_filter,
                    modelo_filter or "TODOS",
                    max_pages,
                    f" | {', '.join(filter_desc)}" if filter_desc else "",
                )
                res = self.fetch_segment_across_portals(
                    marca=marca_filter,
                    modelo=modelo_filter,
                    pages_per_portal=max_pages,
                    anio_min=anio_min,
                    anio_max=anio_max,
                    precio_min=precio_min,
                    precio_max=precio_max,
                    km_max=km_max,
                    deep=deep,
                )
                for p_name, items in res.items():
                    portal_counts[p_name] += len(items)
                    for item in items:
                        dedup_map[item["id_aviso"]] = item

            elif segmented_sweep:
                # MODO B: BARRIDO MULTI-SEGMENTO AUTOMÁTICO (Construye curvas de cada modelo)
                logger.info(
                    "Ejecutando BARRIDO MULTI-SEGMENTO sobre %d modelos líderes en Chile (MercadoLibre + Autocosmos + Yapo)...",
                    len(config.TARGET_SEGMENTS),
                )
                for idx, (m_seg, mod_seg) in enumerate(config.TARGET_SEGMENTS, start=1):
                    res = self.fetch_segment_across_portals(
                        marca=m_seg, modelo=mod_seg, pages_per_portal=1
                    )
                    seg_total = 0
                    for p_name, items in res.items():
                        portal_counts[p_name] += len(items)
                        seg_total += len(items)
                        for item in items:
                            dedup_map[item["id_aviso"]] = item
                    logger.info(
                        "[%02d/%02d] Segmento %s %s -> +%d avisos reales capturados (Acumulado único: %d)",
                        idx,
                        len(config.TARGET_SEGMENTS),
                        m_seg,
                        mod_seg,
                        seg_total,
                        len(dedup_map),
                    )
                    time.sleep(0.4)
            else:
                # MODO C: Búsqueda general por páginas
                res = self.fetch_segment_across_portals(
                    marca=None, modelo=None, pages_per_portal=max_pages
                )
                for p_name, items in res.items():
                    portal_counts[p_name] += len(items)
                    for item in items:
                        dedup_map[item["id_aviso"]] = item

        listings = list(dedup_map.values())
        cross_portal_merged = 0
        if listings:
            listings, cross_portal_merged = self.merge_and_deduplicate_cross_platform(listings)
            if cross_portal_merged > 0:
                logger.info(
                    "Fusión Multi-Plataforma: Se detectaron %d publicaciones duplicadas entre portales y se conservó siempre la plataforma más barata.",
                    cross_portal_merged,
                )

        if force_fallback or len(listings) == 0:
            logger.info("Activando Generador de Respaldo Realista del Mercado Chileno...")
            used_fallback = True
            listings = RealisticChileanMarketFallback.generate_listings(total_records=600)
        elif purge_simulated:
            # Si obtuvimos autos 100% reales en vivo, limpiamos los 260 registros simulados antiguos
            purged = self.db.purge_simulated_records()
            if purged > 0:
                logger.info(
                    "Se eliminaron %d registros sintéticos de prueba anteriores para dejar la base 100%% real.",
                    purged,
                )

        if simulate_lifecycle_Verify and used_fallback:
            past_date = (datetime.now() - timedelta(days=5)).isoformat(timespec="seconds")
            self.db.upsert_batch(listings, seen_timestamp=past_date)
            today_listings = []
            for idx, item in enumerate(listings):
                if idx in range(20, 32):
                    continue
                item_copy = dict(item)
                if idx in range(5, 13) and item_copy["precio"] > 1_000_000:
                    item_copy["precio"] = int(round((item_copy["precio"] * 0.90) / 50_000) * 50_000)
                today_listings.append(item_copy)
            listings = today_listings

        now_iso = datetime.now().isoformat(timespec="seconds")
        upsert_stats = self.db.upsert_batch(listings, seen_timestamp=now_iso)

        active_seen_ids: Set[str] = {str(item["id_aviso"]).strip() for item in listings if item.get("id_aviso")}
        # Si fue una búsqueda focalizada de una marca/modelo, solo evaluamos bajas dentro de esa marca/modelo
        sold_count = 0
        if marca_filter or not segmented_sweep:
            sold_count = self.db.mark_missing_as_sold(
                active_seen_ids=active_seen_ids,
                reference_timestamp=now_iso,
                marca_filter=marca_filter,
                modelo_filter=modelo_filter,
            )

        db_summary = self.db.get_summary_stats()
        result = {
            "fuente": "Fallback Realista Chileno" if used_fallback else "Multi-Portal En Vivo (MercadoLibre + Autocosmos + Yapo)",
            "desglose_portales": portal_counts if not used_fallback else {"Fallback": len(listings)},
            "filtro_focalizado": {"marca": marca_filter, "modelo": modelo_filter},
            "procesados_en_corrida": len(listings),
            "nuevos_insertados": upsert_stats.get("inserted", 0),
            "precios_actualizados": upsert_stats.get("updated_price", 0),
            "vistos_sin_cambio": upsert_stats.get("updated_seen", 0),
            "marcados_como_vendidos_hoy": sold_count,
            "resumen_db": db_summary,
        }

        logger.info("Resultado de adquisición: %s", json.dumps(result, ensure_ascii=False))
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Módulo 1: Scraper Multi-Portal y Buscador Focalizado por Marca/Modelo (Chile)"
    )
    parser.add_argument("--marca", type=str, default=None, help="Filtrar búsqueda focalizada por Marca (ej. MAZDA)")
    parser.add_argument("--modelo", type=str, default=None, help="Filtrar búsqueda focalizada por Modelo (ej. 3 o CX-5)")
    parser.add_argument("--anio-min", type=int, default=None, help="Año mínimo (ej. 2018)")
    parser.add_argument("--anio-max", type=int, default=None, help="Año máximo (ej. 2022)")
    parser.add_argument("--precio-min", type=int, default=None, help="Precio mínimo en CLP")
    parser.add_argument("--precio-max", type=int, default=None, help="Precio máximo en CLP")
    parser.add_argument("--km-max", type=int, default=None, help="Kilometraje máximo (ej. 80000)")
    parser.add_argument("--deep", action="store_true", help="Búsqueda profunda año por año en Google / Chileautos")
    parser.add_argument("--pages", type=int, default=2, help="Páginas por portal a consultar")
    parser.add_argument("--fallback", action="store_true", help="Usar generador de respaldo offline")
    parser.add_argument("--verify-lifecycle", action="store_true", help="Verificar ciclo de vida en SQLite")
    args = parser.parse_args()

    scraper = ChileautosScraper()
    scraper.run_daily_acquisition(
        max_pages=args.pages,
        force_fallback=args.fallback,
        simulate_lifecycle_Verify=args.verify_lifecycle,
        marca_filter=args.marca,
        modelo_filter=args.modelo,
        anio_min=args.anio_min,
        anio_max=args.anio_max,
        precio_min=args.precio_min,
        precio_max=args.precio_max,
        km_max=args.km_max,
        deep=args.deep,
    )


if __name__ == "__main__":
    main()
