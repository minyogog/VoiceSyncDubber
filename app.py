"""
app.py — VoiceSync Dubber
Interface Streamlit complète pour le doublage automatique
multilingue avec clonage vocal et préservation de la prosodie.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import torch

# Ajout du dossier racine au path
sys.path.insert(0, str(Path(__file__).parent))

from separator   import separer_audio
from transcriber import charger_whisper, transcrire
from translator  import initialiser_traducteurs, traduire_segments
from synthesizer import (
    charger_chatterbox,
    extraire_profil_vocal,
    synthetiser_doublage
)
from builder import construire_videos


# ═══════════════════════════════════════════════════
# CONFIGURATION PAGE
# ═══════════════════════════════════════════════════
st.set_page_config(
    page_title="VoiceSync Dubber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════
# CACHE DES MODÈLES — chargés UNE SEULE FOIS
# Évite les redémarrages intempestifs de Streamlit
# ═══════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_whisper(taille: str, device: str):
    """Whisper chargé une seule fois en mémoire."""
    return charger_whisper(taille, device)


@st.cache_resource(show_spinner=False)
def get_chatterbox(device: str):
    """Chatterbox chargé une seule fois en mémoire."""
    return charger_chatterbox(device)


@st.cache_resource(show_spinner=False)
def get_traducteurs():
    """Helsinki-NLP chargé une seule fois en mémoire."""
    initialiser_traducteurs()
    return True


@st.cache_resource(show_spinner=False)
def precharger_tout(device: str):
    """
    Précharge TOUS les modèles au démarrage.
    Évite les rechargements qui causent les redémarrages.
    """
    print("Préchargement de tous les modèles...")
    get_traducteurs()
    get_whisper("medium", device)
    get_chatterbox(device)
    print("✅ Tous les modèles sont en mémoire !")
    return True


# ═══════════════════════════════════════════════════
# PRÉCHARGEMENT AU DÉMARRAGE
# ═══════════════════════════════════════════════════
device = "cuda" if torch.cuda.is_available() else "cpu"

with st.spinner("⏳ Initialisation des modèles IA..."):
    precharger_tout(device)


# ═══════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎙️ VoiceSync Dubber")
    st.caption("Doublage automatique multilingue")
    st.divider()

    # Modèle Whisper
    st.subheader("⚙️ Whisper")
    whisper_taille = st.selectbox(
        "Taille du modèle",
        options=["tiny", "base", "small", "medium", "large"],
        index=2,
        help=(
            "tiny = rapide, moins précis\n"
            "medium = recommandé\n"
            "large = meilleur, très lent"
        )
    )

    # Langues cibles
    st.subheader("🌍 Langues cibles")
    lang_en = st.checkbox("🇬🇧 Anglais (EN)", value=True)
    lang_es = st.checkbox("🇪🇸 Espagnol (ES)", value=True)
    lang_de = st.checkbox("🇩🇪 Allemand (DE)", value=True)

    langues_cibles = []
    if lang_en: langues_cibles.append("en")
    if lang_es: langues_cibles.append("es")
    if lang_de: langues_cibles.append("de")

    # Paramètres TTS
    st.subheader("🎭 Voix")
    exaggeration = st.slider(
        "Expressivité",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0.1 = neutre  |  1.0 = très expressif"
    )

    volume_musique = st.slider(
        "Volume musique de fond",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05
    )

    # Option normalisation accent
    st.subheader("🗣️ Accent")
    normaliser_accent = st.checkbox(
        "Normaliser vers français de France",
        value=True,
        help=(
            "Convertit le français camerounais "
            "vers le français standard de France "
            "avant le clonage vocal."
        )
    )

    st.divider()

    st.caption(
        f"**Device** : `{device.upper()}`  \n"
        f"**Python** : {sys.version_info.major}."
        f"{sys.version_info.minor}  \n"
        f"**PyTorch** : {torch.__version__}"
    )

    if st.button(
        "🗑️ Vider le cache",
        use_container_width=True
    ):
        st.cache_resource.clear()
        st.success("Cache vidé !")
        st.rerun()


# ═══════════════════════════════════════════════════
# PAGE PRINCIPALE
# ═══════════════════════════════════════════════════
st.title("🎬 VoiceSync Dubber")
st.markdown(
    "Upload une vidéo en **français** pour obtenir "
    "automatiquement les versions doublées en "
    "**Anglais**, **Espagnol** et **Allemand** "
    "avec **clonage vocal** et **préservation "
    "de la prosodie**."
)

# Zone upload
fichier = st.file_uploader(
    "📁 Charger une vidéo",
    type=["mp4", "mkv", "avi", "mov"],
    help="Formats supportés : MP4, MKV, AVI, MOV"
)

if fichier is not None:

    st.video(fichier)

    if not langues_cibles:
        st.error(
            "❌ Sélectionne au moins une langue "
            "dans la sidebar !"
        )
        st.stop()

    lancer = st.button(
        "🚀 Lancer le doublage",
        type="primary",
        use_container_width=True
    )

    if lancer:

        progress = st.progress(0, text="Initialisation...")

        def maj_progress(etape: int, total: int, msg: str):
            pct = int((etape / total) * 100)
            progress.progress(
                pct, text=f"[{etape}/{total}] {msg}"
            )
            st.toast(msg)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            video_path = tmp / fichier.name
            with open(video_path, "wb") as f:
                f.write(fichier.getbuffer())

            try:
                TOTAL_ETAPES = 6

                # ───────────────────────────────────
                # ÉTAPE 1 : Séparation voix/musique
                # ───────────────────────────────────
                maj_progress(
                    1, TOTAL_ETAPES,
                    "Séparation voix/musique (Demucs)..."
                )

                sep_dir = str(tmp / "separation")
                vocals, accompaniment = separer_audio(
                    str(video_path),
                    sep_dir,
                    device
                )

                if not os.path.exists(vocals):
                    raise FileNotFoundError(
                        "vocals.wav non généré. "
                        "Vérifie que la vidéo a une piste audio."
                    )

                # ───────────────────────────────────
                # ÉTAPE 2 : Clonage vocal
                # ───────────────────────────────────
                maj_progress(
                    2, TOTAL_ETAPES,
                    "Extraction profil vocal (Chatterbox)..."
                )

                profil_path = str(tmp / "profil_vocal.wav")
                extraire_profil_vocal(vocals, profil_path)

                # Récupération depuis le cache
                chatterbox = get_chatterbox(device)

                # ───────────────────────────────────
                # ÉTAPE 3 : Transcription + Prosodie
                # ───────────────────────────────────
                maj_progress(
                    3, TOTAL_ETAPES,
                    "Transcription + normalisation accent..."
                )

                whisper = get_whisper(whisper_taille, device)

                segments = transcrire(
                    whisper,
                    vocals,
                    "fr",
                    normaliser=normaliser_accent
                )

                if not segments:
                    raise ValueError(
                        "Aucun segment vocal détecté. "
                        "La vidéo est peut-être muette."
                    )

                st.info(
                    f"✅ {len(segments)} segments transcrits"
                    + (" et normalisés vers FR France"
                       if normaliser_accent else "")
                )

                # ───────────────────────────────────
                # ÉTAPE 4 : Traduction
                # ───────────────────────────────────
                maj_progress(
                    4, TOTAL_ETAPES,
                    "Traduction multilingue (Helsinki-NLP)..."
                )

                get_traducteurs()

                segments_par_langue = {}
                for lang in langues_cibles:
                    segments_par_langue[lang] = \
                        traduire_segments(segments, lang)

                # ───────────────────────────────────
                # ÉTAPE 5 : Synthèse vocale
                # ───────────────────────────────────
                maj_progress(
                    5, TOTAL_ETAPES,
                    "Synthèse vocale naturelle (Chatterbox)..."
                )

                audios_doubles = {}
                for lang in langues_cibles:
                    output_wav = str(
                        tmp / f"dubbed_{lang}.wav"
                    )
                    with st.spinner(
                        f"Synthèse {lang.upper()} "
                        f"(audio naturel sans déformation)..."
                    ):
                        synthetiser_doublage(
                            model=chatterbox,
                            segments=segments_par_langue[lang],
                            profil_vocal=profil_path,
                            langue_cible=lang,
                            exaggeration=exaggeration,
                            output_path=output_wav
                        )
                    audios_doubles[lang] = output_wav

                # ───────────────────────────────────
                # ÉTAPE 6 : Montage vidéo
                # ───────────────────────────────────
                maj_progress(
                    6, TOTAL_ETAPES,
                    "Montage vidéo final (ffmpeg)..."
                )

                output_dir = str(tmp / "output")
                videos = construire_videos(
                    video_originale=str(video_path),
                    audios_doubles=audios_doubles,
                    accompaniment_path=accompaniment,
                    output_dir=output_dir
                )

                # ───────────────────────────────────
                # RÉSULTATS
                # ───────────────────────────────────
                progress.empty()
                st.balloons()
                st.success("🎉 Doublage terminé avec succès !")
                st.divider()

                st.header("📼 Vidéos finales")

                drapeaux = {
                    "fr": "🇫🇷 Français",
                    "en": "🇬🇧 Anglais",
                    "es": "🇪🇸 Espagnol",
                    "de": "🇩🇪 Allemand"
                }

                toutes_langues  = ["fr"] + langues_cibles
                noms_onglets    = [
                    drapeaux.get(l, l.upper())
                    for l in toutes_langues
                ]
                onglets = st.tabs(noms_onglets)

                for onglet, langue in zip(
                    onglets, toutes_langues
                ):
                    with onglet:
                        video_finale = videos.get(langue)

                        if video_finale and \
                           os.path.exists(str(video_finale)):

                            st.video(str(video_finale))

                            with open(
                                str(video_finale), "rb"
                            ) as vf:
                                st.download_button(
                                    label=(
                                        f"⬇️ Télécharger "
                                        f"video_{langue.upper()}"
                                        f".mp4"
                                    ),
                                    data=vf,
                                    file_name=(
                                        f"video_"
                                        f"{langue.upper()}.mp4"
                                    ),
                                    mime="video/mp4",
                                    use_container_width=True,
                                    type="primary"
                                )

                            if langue == "fr":
                                with st.expander(
                                    "📝 Transcription FR"
                                    + (" (normalisée)"
                                       if normaliser_accent
                                       else ""),
                                    expanded=False
                                ):
                                    for s in segments:
                                        st.markdown(
                                            f"**{s['start']:.1f}s"
                                            f" → {s['end']:.1f}s**"
                                            f"  \n{s['text']}"
                                        )
                            else:
                                with st.expander(
                                    f"📝 Script "
                                    f"{langue.upper()}",
                                    expanded=False
                                ):
                                    segs_tr = \
                                        segments_par_langue\
                                        .get(langue, [])
                                    for s in segs_tr:
                                        st.markdown(
                                            f"**{s['start']:.1f}s"
                                            f" → {s['end']:.1f}s**"
                                            f"  \n🇫🇷 {s['text']}"
                                            f"  \n🌍 "
                                            f"{s['texte_traduit']}"
                                        )
                        else:
                            st.error(
                                f"❌ Vidéo {langue.upper()} "
                                f"non générée."
                            )

                with st.expander(
                    "🔧 Résumé technique",
                    expanded=False
                ):
                    st.json({
                        "device"            : device,
                        "whisper"           : whisper_taille,
                        "segments"          : len(segments),
                        "langues"           : langues_cibles,
                        "exaggeration"      : exaggeration,
                        "volume_musique"    : volume_musique,
                        "normaliser_accent" : normaliser_accent,
                        "time_stretch"      : False
                    })

            except Exception as e:
                progress.empty()
                st.error(f"❌ Erreur : {str(e)}")
                st.exception(e)

else:
    st.info(
        "👆 Upload une vidéo en français "
        "pour démarrer le doublage automatique."
    )
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎵 Séparation")
        st.markdown(
            "**Demucs** sépare la voix "
            "de la musique automatiquement."
        )

    with col2:
        st.markdown("### 🎙️ Clonage")
        st.markdown(
            "**Chatterbox** clone le timbre vocal. "
            "Normalisation accent camerounais → France."
        )

    with col3:
        st.markdown("### 🌍 Traduction")
        st.markdown(
            "**Helsinki-NLP** traduit offline "
            "en EN, ES et DE."
        )

    st.divider()

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("### 📝 Transcription")
        st.markdown(
            "**Faster-Whisper** transcrit avec "
            "timestamps mot par mot."
        )

    with col5:
        st.markdown("### 🔊 Synthèse")
        st.markdown(
            "Audio naturel sans déformation. "
            "Silences originaux préservés."
        )

    with col6:
        st.markdown("### 🎬 Montage")
        st.markdown(
            "**FFmpeg** remixe la nouvelle voix "
            "avec la musique originale."
        )