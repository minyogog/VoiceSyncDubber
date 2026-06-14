"""
separator.py — Séparation voix/musique avec Demucs
Prend une vidéo, extrait l'audio, sépare vocals et accompaniment
"""

import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path


def extraire_audio(video_path: str, output_wav: str) -> str:
    """
    Extrait la piste audio d'une vidéo en WAV stéréo 44100Hz via ffmpeg.
    """
    try:
        import ffmpeg
        (
            ffmpeg
            .input(video_path)
            .output(
                output_wav,
                ar=44100,
                ac=2,
                vn=None,
                acodec='pcm_s16le'
            )
            .overwrite_output()
            .run(quiet=True)
        )
        return output_wav
    except Exception as e:
        raise RuntimeError(f"Erreur extraction audio : {str(e)}")


def separer_audio(
    video_path: str,
    output_dir: str,
    device: str = "cpu"
) -> tuple:
    """
    Sépare la voix et la musique d'une vidéo.

    Args:
        video_path : chemin de la vidéo source
        output_dir : dossier de sortie
        device     : cpu ou cuda

    Returns:
        (vocals_path, accompaniment_path)
    """
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        # Création du dossier de sortie
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Étape 1 : extraction audio brute
        raw_wav = os.path.join(output_dir, "raw_audio.wav")
        extraire_audio(video_path, raw_wav)

        if not os.path.exists(raw_wav):
            raise FileNotFoundError("Extraction audio échouée.")

        # Étape 2 : chargement modèle Demucs
        print("Chargement modèle Demucs htdemucs...")
        model = get_model("htdemucs")
        model.to(device)
        model.eval()

        # Étape 3 : chargement audio
        audio_data, sr = sf.read(raw_wav, always_2d=True)

        # Conversion en tensor [1, channels, samples]
        audio_tensor = torch.from_numpy(
            audio_data.T
        ).float().unsqueeze(0).to(device)

        # Rééchantillonnage si nécessaire
        if sr != model.samplerate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=model.samplerate
            )
            audio_tensor = resampler(audio_tensor)

        # Étape 4 : séparation
        print("Séparation en cours...")
        with torch.no_grad():
            sources = apply_model(
                model,
                audio_tensor,
                device=device,
                progress=True
            )

        # sources [batch, stems, channels, samples]
        # htdemucs : drums=0, bass=1, other=2, vocals=3
        sources = sources[0].cpu().numpy()

        vocals        = sources[3]  # [channels, samples]
        accompaniment = sources[0] + sources[1] + sources[2]

        # Étape 5 : sauvegarde
        vocals_path = os.path.join(output_dir, "vocals.wav")
        accompaniment_path = os.path.join(
            output_dir, "accompaniment.wav"
        )

        sf.write(vocals_path, vocals.T, model.samplerate)
        sf.write(accompaniment_path, accompaniment.T, model.samplerate)

        print(f"✅ vocals.wav → {vocals_path}")
        print(f"✅ accompaniment.wav → {accompaniment_path}")

        return vocals_path, accompaniment_path

    except Exception as e:
        raise RuntimeError(f"Erreur séparation audio : {str(e)}")


# ─── TEST RAPIDE ───────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python separator.py chemin/video.mp4")
        sys.exit(1)

    video = sys.argv[1]
    out   = "test_separation"

    print(f"Vidéo : {video}")
    print(f"Sortie : {out}")

    vocals, bgm = separer_audio(video, out)
    print(f"\n✅ Terminé !")
    print(f"   Vocals        : {vocals}")
    print(f"   Accompagnement: {bgm}")