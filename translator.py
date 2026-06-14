"""
translator.py — Traduction offline avec Helsinki-NLP (MarianMT)
Traduit les segments FR → EN / ES / DE en préservant la prosodie
"""

import torch
from transformers import MarianMTModel, MarianTokenizer


# Modèles Helsinki-NLP par langue cible
MODELES = {
    "en": "Helsinki-NLP/opus-mt-fr-en",
    "es": "Helsinki-NLP/opus-mt-fr-es",
    "de": "Helsinki-NLP/opus-mt-fr-de",
}

# Cache en mémoire pour éviter les rechargements
_cache_tokenizer = {}
_cache_modele    = {}


def charger_modele_traduction(langue: str):
    """
    Charge et met en cache le modèle Helsinki pour une langue.

    Args:
        langue : en / es / de

    Returns:
        (tokenizer, model)
    """
    if langue not in MODELES:
        raise ValueError(
            f"Langue '{langue}' non supportée. "
            f"Disponibles : {list(MODELES.keys())}"
        )

    if langue not in _cache_modele:
        nom_modele = MODELES[langue]
        print(f"Chargement modèle FR→{langue.upper()} ({nom_modele})...")

        try:
            tokenizer = MarianTokenizer.from_pretrained(nom_modele)
            modele    = MarianMTModel.from_pretrained(nom_modele)
            modele.eval()

            _cache_tokenizer[langue] = tokenizer
            _cache_modele[langue]    = modele

            print(f"✅ FR→{langue.upper()} prêt !")

        except Exception as e:
            raise RuntimeError(
                f"Erreur chargement modèle {nom_modele} : {str(e)}"
            )

    return _cache_tokenizer[langue], _cache_modele[langue]


def initialiser_traducteurs():
    """
    Précharge tous les modèles de traduction en mémoire.
    À appeler une seule fois au démarrage de l'app.
    """
    print("Initialisation des traducteurs Helsinki-NLP...")
    for langue in MODELES:
        charger_modele_traduction(langue)
    print("✅ Tous les traducteurs sont prêts !")


def traduire_texte(texte: str, langue_cible: str) -> str:
    """
    Traduit un texte du français vers la langue cible.

    Args:
        texte        : texte source en français
        langue_cible : en / es / de

    Returns:
        Texte traduit
    """
    if not texte or not texte.strip():
        return texte

    try:
        tokenizer, modele = charger_modele_traduction(langue_cible)

        # Tokenisation
        inputs = tokenizer(
            texte,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Génération
        with torch.no_grad():
            traduit = modele.generate(
                **inputs,
                num_beams=4,
                early_stopping=True
            )

        # Décodage
        resultat = tokenizer.decode(
            traduit[0],
            skip_special_tokens=True
        )

        return resultat.strip()

    except Exception as e:
        # En cas d'erreur on garde le texte original
        print(f"⚠️ Erreur traduction '{texte[:30]}...' : {str(e)}")
        return texte


def traduire_segments(
    segments: list,
    langue_cible: str
) -> list:
    """
    Traduit tous les segments en préservant les métadonnées
    prosodiques (timestamps, silences, pitch, énergie...).

    Args:
        segments     : liste de segments depuis transcriber.py
        langue_cible : en / es / de

    Returns:
        Nouvelle liste avec champ 'texte_traduit' ajouté
    """
    if not segments:
        return []

    if langue_cible not in MODELES:
        raise ValueError(
            f"Langue '{langue_cible}' non supportée."
        )

    print(f"Traduction FR→{langue_cible.upper()} "
          f"({len(segments)} segments)...")

    resultats = []

    for i, seg in enumerate(segments):
        nouveau_seg = dict(seg)  # copie complète
        texte_original = seg.get("text", "")

        try:
            nouveau_seg["texte_traduit"] = traduire_texte(
                texte_original,
                langue_cible
            )
        except Exception:
            nouveau_seg["texte_traduit"] = texte_original

        resultats.append(nouveau_seg)

        # Progression
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(segments)} segments traduits...")

    print(f"✅ Traduction FR→{langue_cible.upper()} terminée !")
    return resultats


# ─── TEST RAPIDE ───────────────────────────────────────────
if __name__ == "__main__":

    # Test de traduction simple
    phrases = [
        "Bonjour, bienvenue dans cette vidéo.",
        "Aujourd'hui nous allons parler de patience.",
        "Le monde se trouve au bord du précipice.",
        "Il faut savoir attendre le bon moment.",
    ]

    # Simulation de segments comme transcriber.py les produit
    segments_test = [
        {
            "text"         : p,
            "start"        : i * 3.0,
            "end"          : i * 3.0 + 2.5,
            "silence_avant": 0.0,
            "silence_apres": 0.5,
            "pitch_mean"   : 180.0,
            "energy_mean"  : 0.05,
            "speaking_rate": 3.2,
            "word_count"   : len(p.split()),
            "mots"         : []
        }
        for i, p in enumerate(phrases)
    ]

    print("=" * 50)
    print("TEST TRADUCTEUR HELSINKI-NLP")
    print("=" * 50)

    for langue in ["en", "es", "de"]:
        print(f"\n🌍 FR → {langue.upper()}")
        print("-" * 30)
        segs_traduits = traduire_segments(segments_test, langue)
        for s in segs_traduits:
            print(f"  FR : {s['text']}")
            print(f"  {langue.upper()} : {s['texte_traduit']}")
            print()