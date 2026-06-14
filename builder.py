"""
builder.py — Montage vidéo final avec ffmpeg
Mixe la voix doublée avec la musique originale
et remplace la piste audio de la vidéo source
"""

import os
import shutil
import ffmpeg
from pathlib import Path


def mixer_audio(
    voix_path: str,
    musique_path: str,
    output_path: str,
    volume_voix: float = 1.0,
    volume_musique: float = 0.8
) -> str:
    """
    Mixe la voix doublée avec la musique/bruits de fond.

    Args:
        voix_path      : chemin WAV de la voix doublée
        musique_path   : chemin WAV de l'accompagnement
        output_path    : chemin WAV du mix final
        volume_voix    : volume de la voix (défaut 1.0)
        volume_musique : volume de la musique (défaut 0.8)

    Returns:
        Chemin du fichier mixé
    """
    try:
        print(f"Mixage audio...")

        voix    = ffmpeg.input(voix_path).audio
        musique = ffmpeg.input(musique_path).audio

        # Application des volumes
        voix    = voix.filter('volume', volume=volume_voix)
        musique = musique.filter('volume', volume=volume_musique)

        # Mixage avec amix
        mix = ffmpeg.filter(
            [voix, musique],
            'amix',
            inputs=2,
            duration='longest'
        )

        # Export WAV 48kHz pour compatibilité vidéo
        (
            ffmpeg
            .output(
                mix,
                output_path,
                acodec='pcm_s16le',
                ar=48000
            )
            .overwrite_output()
            .run(quiet=True)
        )

        print(f"✅ Mix sauvegardé : {output_path}")
        return output_path

    except ffmpeg.Error as e:
        erreur = e.stderr.decode('utf-8') if e.stderr else str(e)
        raise RuntimeError(f"Erreur ffmpeg mixage : {erreur}")
    except Exception as e:
        raise RuntimeError(f"Erreur mixage audio : {str(e)}")


def remplacer_audio_video(
    video_originale: str,
    audio_path: str,
    output_video: str
) -> str:
    """
    Remplace la piste audio de la vidéo originale
    sans ré-encoder la vidéo (copy codec = rapide).

    Args:
        video_originale : chemin de la vidéo source
        audio_path      : chemin du nouvel audio WAV
        output_video    : chemin de la vidéo finale MP4

    Returns:
        Chemin de la vidéo finale
    """
    try:
        print(f"Remplacement piste audio...")

        video  = ffmpeg.input(video_originale).video
        audio  = ffmpeg.input(audio_path).audio

        (
            ffmpeg
            .output(
                video,
                audio,
                output_video,
                vcodec='copy',
                acodec='aac',
                audio_bitrate='192k',
                strict='experimental'
            )
            .overwrite_output()
            .run(quiet=True)
        )

        print(f"✅ Vidéo finale : {output_video}")
        return output_video

    except ffmpeg.Error as e:
        erreur = e.stderr.decode('utf-8') if e.stderr else str(e)
        raise RuntimeError(
            f"Erreur ffmpeg remplacement audio : {erreur}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Erreur remplacement audio : {str(e)}"
        )


def construire_videos(
    video_originale: str,
    audios_doubles: dict,
    accompaniment_path: str,
    output_dir: str
) -> dict:
    """
    Construit toutes les vidéos doublées.

    Pipeline pour chaque langue :
    1. Mixer voix doublée + accompagnement original
    2. Remplacer la piste audio de la vidéo
    3. Exporter video_EN.mp4 / video_ES.mp4 / video_DE.mp4

    La vidéo française originale est copiée telle quelle.

    Args:
        video_originale    : chemin de la vidéo source
        audios_doubles     : dict {langue: chemin_wav}
                             ex: {"en": "dubbed_en.wav"}
        accompaniment_path : chemin accompaniment.wav
        output_dir         : dossier de sortie

    Returns:
        dict {langue: chemin_video_finale}
        ex: {"fr": "video_FR.mp4", "en": "video_EN.mp4"}
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        videos_finales = {}

        # Vidéo FR = copie de l'originale
        video_fr = os.path.join(output_dir, "video_FR.mp4")
        shutil.copy2(video_originale, video_fr)
        videos_finales["fr"] = video_fr
        print(f"✅ video_FR.mp4 copiée")

        # Vidéos doublées pour chaque langue
        for langue, audio_double in audios_doubles.items():

            if not os.path.exists(audio_double):
                print(f"⚠️ Audio {langue} introuvable, ignoré.")
                continue

            print(f"\n── Construction video_{langue.upper()}.mp4 ──")

            # Étape 1 : mixage voix + musique
            mix_path = os.path.join(
                output_dir,
                f"mix_{langue}.wav"
            )
            mixer_audio(
                voix_path=audio_double,
                musique_path=accompaniment_path,
                output_path=mix_path
            )

            # Étape 2 : remplacement audio dans la vidéo
            video_finale = os.path.join(
                output_dir,
                f"video_{langue.upper()}.mp4"
            )
            remplacer_audio_video(
                video_originale=video_originale,
                audio_path=mix_path,
                output_video=video_finale
            )

            videos_finales[langue] = video_finale

            # Nettoyage du mix temporaire
            try:
                os.remove(mix_path)
            except Exception:
                pass

        print(f"\n✅ {len(videos_finales)} vidéos construites !")
        return videos_finales

    except Exception as e:
        raise RuntimeError(
            f"Erreur construction vidéos : {str(e)}"
        )


# ─── TEST RAPIDE ───────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage : python builder.py "
            "video.mp4 vocals.wav accompaniment.wav"
        )
        sys.exit(1)

    video        = sys.argv[1]
    vocals       = sys.argv[2]
    accompaniment = sys.argv[3]

    # Simulation d'audios doublés
    # (en production ils viennent de synthesizer.py)
    audios_test = {
        "en": vocals,  # on utilise vocals comme test
    }

    videos = construire_videos(
        video_originale=video,
        audios_doubles=audios_test,
        accompaniment_path=accompaniment,
        output_dir="test_output"
    )

    print("\n── VIDÉOS GÉNÉRÉES ──")
    for langue, chemin in videos.items():
        print(f"  {langue.upper()} : {chemin}")