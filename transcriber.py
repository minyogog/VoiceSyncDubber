"""
transcriber.py — Transcription STT avec Faster-Whisper
Transcrit l'audio vocal avec timestamps et analyse prosodique
Normalise le français camerounais vers le français de France
"""

import os
import numpy as np
import librosa
from pathlib import Path
from faster_whisper import WhisperModel


def charger_whisper(
    taille: str = "medium",
    device: str = "cpu"
) -> WhisperModel:
    try:
        compute_type = "int8" if device == "cpu" else "float16"
        print(f"Chargement Whisper {taille} ({device})...")
        model = WhisperModel(
            taille,
            device=device,
            compute_type=compute_type,
            cpu_threads=os.cpu_count() or 4
        )
        print(f"✅ Whisper {taille} chargé !")
        return model
    except Exception as e:
        raise RuntimeError(
            f"Erreur chargement Whisper : {str(e)}"
        )


def normaliser_français(texte: str) -> str:
    """
    Normalise le français camerounais vers
    le français de France standard.
    Remplace les expressions et tournures locales.
    """
    if not texte:
        return texte

    # Dictionnaire expressions camerounaises → français France
    corrections = {
        # Verbes et expressions courantes
        "je suis en train de"  : "je suis en train de",
        "on va faire comment"  : "qu'est-ce qu'on va faire",
        "c'est comment"        : "comment ça va",
        "ça va aller"          : "ça va bien se passer",
        "je vais descendre"    : "je vais partir",
        "il a gâté"            : "il est cassé",
        "ça a gâté"            : "ça s'est cassé",
        "mettre ça"            : "poser ça",
        "porter"               : "apporter",
        "je sors"              : "je pars",
        "je rentre"            : "je reviens",
        "il a fini"            : "il a terminé",
        "je fais comment"      : "comment je fais",
        "on peut faire quoi"   : "que peut-on faire",
        "c'est quoi ça"        : "qu'est-ce que c'est",
        "c'est qui"            : "qui est-ce",
        "tu as vu"             : "tu as remarqué",
        "ça fait mal au cœur"  : "c'est douloureux",
        "il est parti où"      : "où est-il allé",
        "depuis quand"         : "depuis quand",
        "jusqu'à quand"        : "jusqu'à quand",
        "on va où"             : "où allons-nous",
        "je viens d'arriver"   : "je viens d'arriver",
        "je viens juste"       : "je viens juste de",
        "il a dit que"         : "il a dit que",
        "tu as entendu"        : "as-tu entendu",
        "vous avez vu"         : "avez-vous vu",
        "on a fait quoi"       : "qu'avons-nous fait",
        "c'est pas normal"     : "ce n'est pas normal",
        "c'est pas bon"        : "ce n'est pas bien",
        "c'est pas possible"   : "ce n'est pas possible",
        "dieu merci"           : "heureusement",
        "mon frère"            : "mon ami",
        "ma sœur"              : "mon amie",
        "on est ensemble"      : "nous sommes ensemble",
        "tu pars où"           : "où vas-tu",
        "je reviens"           : "je reviens tout de suite",
        "maintenant maintenant": "immédiatement",
        "vite vite"            : "rapidement",
        "doucement doucement"  : "lentement",
        "c'est ça même"        : "c'est exactement ça",
        "il est là"            : "il est là",
        "on est là"            : "nous sommes là",
        "laisse"               : "laisse tomber",
        "j'arrive"             : "j'arrive bientôt",
    }

    texte_normalise = texte
    for expression_cm, expression_fr in corrections.items():
        texte_normalise = texte_normalise.replace(
            expression_cm, expression_fr
        )

    return texte_normalise


def analyser_prosodie(
    audio: np.ndarray,
    sr: int,
    start: float,
    end: float
) -> dict:
    try:
        s = max(0, min(int(start * sr), len(audio)))
        e = max(0, min(int(end   * sr), len(audio)))
        segment = audio[s:e]

        pitch_mean  = 0.0
        energy_mean = 0.0

        if len(segment) > 1024:
            f0, voiced, _ = librosa.pyin(
                segment,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            valides = f0[~np.isnan(f0)]
            if len(valides) > 0:
                pitch_mean = float(np.mean(valides))

        if len(segment) > 0:
            energy_mean = float(
                np.sqrt(np.mean(segment ** 2))
            )

        return {
            "pitch_mean" : round(pitch_mean, 2),
            "energy_mean": round(energy_mean, 6)
        }
    except Exception:
        return {"pitch_mean": 0.0, "energy_mean": 0.0}


def transcrire(
    model: WhisperModel,
    vocals_path: str,
    langue: str = "fr",
    normaliser: bool = True
) -> list:
    """
    Transcrit l'audio avec timestamps mot par mot.

    Args:
        model       : instance WhisperModel
        vocals_path : chemin du fichier vocals.wav
        langue      : code langue (fr)
        normaliser  : normaliser vers français de France

    Returns:
        Liste de segments prosodiques
    """
    try:
        print(f"Transcription en cours ({langue})...")

        audio, sr = librosa.load(
            vocals_path,
            sr=16000,
            mono=True
        )
        duree_totale = len(audio) / sr

        # Prompt initial pour guider Whisper
        # vers le français standard
        initial_prompt = (
            "Transcription en français standard de France. "
            "Vocabulaire français neutre et académique. "
            "Prononciation française européenne."
        ) if normaliser else None

        segments_iter, info = model.transcribe(
            vocals_path,
            language=langue,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=True,
            initial_prompt=initial_prompt
        )

        print(
            f"Langue détectée : {info.language} "
            f"({info.language_probability:.0%})"
        )

        segments  = list(segments_iter)
        resultats = []
        prev_end  = 0.0

        for i, seg in enumerate(segments):
            start = float(seg.start)
            end   = float(seg.end)
            texte = seg.text.strip()

            # Normalisation français camerounais → France
            if normaliser and langue == "fr":
                texte = normaliser_français(texte)

            silence_avant = max(0.0, start - prev_end)
            silence_apres = 0.0
            if i < len(segments) - 1:
                silence_apres = max(
                    0.0,
                    float(segments[i + 1].start) - end
                )
            else:
                silence_apres = max(
                    0.0, duree_totale - end
                )

            prosodie = analyser_prosodie(
                audio, sr, start, end
            )

            mots = []
            if seg.words:
                for w in seg.words:
                    mots.append({
                        "mot"  : w.word.strip(),
                        "start": float(w.start),
                        "end"  : float(w.end)
                    })

            nb_mots = len(mots) if mots else len(texte.split())
            duree   = max(end - start, 0.001)
            debit   = round(nb_mots / duree, 2)

            resultats.append({
                "text"         : texte,
                "start"        : start,
                "end"          : end,
                "silence_avant": round(silence_avant, 3),
                "silence_apres": round(silence_apres, 3),
                "pitch_mean"   : prosodie["pitch_mean"],
                "energy_mean"  : prosodie["energy_mean"],
                "speaking_rate": debit,
                "word_count"   : nb_mots,
                "mots"         : mots
            })

            prev_end = end

        print(
            f"✅ {len(resultats)} segments transcrits "
            f"et normalisés"
        )
        return resultats

    except Exception as e:
        raise RuntimeError(
            f"Erreur transcription : {str(e)}"
        )