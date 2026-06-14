"""
synthesizer.py — TTS + Clonage vocal avec Chatterbox
Synthétise la voix traduite dans le timbre du locuteur original
SANS time-stretching — audio naturel prioritaire
"""

import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path


def charger_chatterbox(device: str = "cpu"):
    try:
        from chatterbox.tts import ChatterboxTTS
        print(f"Chargement Chatterbox TTS ({device})...")
        model = ChatterboxTTS.from_pretrained(device=device)
        print("✅ Chatterbox TTS chargé !")
        return model
    except ImportError:
        raise RuntimeError(
            "chatterbox-tts non installé. "
            "pip install chatterbox-tts"
        )
    except Exception as e:
        raise RuntimeError(
            f"Erreur chargement Chatterbox : {str(e)}"
        )


def extraire_profil_vocal(
    vocals_path: str,
    output_path: str,
    duree: float = 10.0,
    sr_cible: int = 24000
) -> str:
    try:
        print(f"Extraction profil vocal ({duree}s)...")

        wav, sr = librosa.load(
            vocals_path,
            sr=sr_cible,
            mono=True
        )

        nb_samples = int(duree * sr_cible)
        profil     = wav[:nb_samples]

        max_val = np.max(np.abs(profil))
        if max_val > 0:
            profil = profil / max_val * 0.95

        Path(output_path).parent.mkdir(
            parents=True, exist_ok=True
        )
        sf.write(output_path, profil, sr_cible)

        print(f"✅ Profil vocal : {output_path}")
        return output_path

    except Exception as e:
        raise RuntimeError(
            f"Erreur profil vocal : {str(e)}"
        )


def synthetiser_doublage(
    model,
    segments: list,
    profil_vocal: str,
    langue_cible: str,
    exaggeration: float = 0.5,
    output_path: str = "dubbed.wav"
) -> str:
    """
    Synthétise l'audio doublé SANS time-stretching.

    L'audio TTS garde sa durée naturelle.
    Les silences originaux sont préservés entre segments.
    Pas de déformation de la voix.
    """
    try:
        sr            = getattr(model, 'sr', 24000)
        tous_segments = []

        print(
            f"Synthèse FR→{langue_cible.upper()} "
            f"({len(segments)} segments)..."
        )

        for i, seg in enumerate(segments):
            texte = seg.get(
                "texte_traduit",
                seg.get("text", "")
            )
            silence_apres = seg.get("silence_apres", 0.0)

            # Segment vide → silence court
            if not texte or not texte.strip():
                duree_seg = max(
                    seg.get("end", 0) -
                    seg.get("start", 0),
                    0.1
                )
                silence = np.zeros(
                    int(duree_seg * sr),
                    dtype=np.float32
                )
                tous_segments.append(silence)
                continue

            # Génération TTS — durée NATURELLE
            # pas de time-stretch appliqué
            try:
                audio_tts = model.generate(
                    text=texte,
                    audio_prompt_path=profil_vocal,
                    exaggeration=exaggeration
                )

                # Conversion robuste numpy 1D
                if hasattr(audio_tts, 'numpy'):
                    audio_tts = audio_tts.numpy()
                elif hasattr(audio_tts, 'cpu'):
                    audio_tts = audio_tts.cpu().numpy()
                elif isinstance(audio_tts, (list, tuple)):
                    audio_tts = np.array(audio_tts[0])

                audio_tts = np.asarray(
                    audio_tts
                ).squeeze().astype(np.float32)

            except Exception as gen_err:
                print(
                    f"  ⚠️ Segment {i+1} : {gen_err}"
                )
                # Silence de remplacement
                duree_seg = max(
                    seg.get("end", 0) -
                    seg.get("start", 0),
                    0.1
                )
                audio_tts = np.zeros(
                    int(duree_seg * sr),
                    dtype=np.float32
                )

            # Ajout silence après segment
            # pour préserver le rythme naturel
            if silence_apres > 0.01:
                silence = np.zeros(
                    int(silence_apres * sr),
                    dtype=np.float32
                )
                audio_tts = np.concatenate(
                    [audio_tts, silence]
                )

            tous_segments.append(audio_tts)

            print(
                f"  [{i+1}/{len(segments)}] "
                f"{texte[:45]}..."
            )

        # Concaténation finale
        if not tous_segments:
            audio_final = np.zeros(sr, dtype=np.float32)
        else:
            audio_final = np.concatenate(tous_segments)

        # Normalisation -20 dBFS
        rms = np.sqrt(np.mean(audio_final ** 2))
        if rms > 0:
            gain        = (10 ** (-20 / 20)) / rms
            audio_final = np.clip(
                audio_final * gain, -1.0, 1.0
            )

        Path(output_path).parent.mkdir(
            parents=True, exist_ok=True
        )
        sf.write(
            output_path,
            audio_final,
            sr,
            subtype='PCM_16'
        )

        duree_finale = len(audio_final) / sr
        print(
            f"✅ Audio doublé : {output_path} "
            f"({duree_finale:.1f}s)"
        )
        return output_path

    except Exception as e:
        raise RuntimeError(
            f"Erreur synthèse : {str(e)}"
        )