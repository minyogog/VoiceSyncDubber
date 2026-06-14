"""
deploy_github.py — Déploiement automatique sur GitHub
Lance : python deploy_github.py
"""

import subprocess
import sys
import os
from pathlib import Path

# ═══════════════════════════════════════
# CONFIGURATION — modifie ici
# ═══════════════════════════════════════
GITHUB_USERNAME = "Minyogog"
REPO_NAME       = "VoiceSyncDubber"
DOSSIER         = Path(__file__).parent
MESSAGE_COMMIT  = "Update automatique VoiceSync Dubber"
# ═══════════════════════════════════════


def run(cmd: str, cwd: str = None) -> tuple:
    """Exécute une commande et retourne (succès, output)."""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd or str(DOSSIER),
        capture_output=True,
        text=True
    )
    return result.returncode == 0, result.stdout + result.stderr


def etape(num: int, msg: str):
    print(f"\n{'─'*50}")
    print(f"  ÉTAPE {num} — {msg}")
    print(f"{'─'*50}")


def verifier_git():
    """Vérifie que Git est installé."""
    ok, out = run("git --version")
    if ok:
        print(f"✅ {out.strip()}")
        return True
    print("❌ Git non trouvé — winget install git")
    return False


def verifier_authentification():
    """Vérifie que GitHub CLI est authentifié."""
    ok, out = run("gh auth status")
    if ok:
        print("✅ GitHub CLI authentifié")
        return True
    print("❌ Non authentifié — lance : gh auth login")
    return False


def creer_gitignore():
    """Crée le .gitignore pour exclure les fichiers lourds."""
    contenu = """# Environnement Python
venv/
__pycache__/
*.pyc
*.pyo
.env

# Modèles IA — trop lourds pour GitHub
models/
.cache/
*.bin
*.safetensors
*.pt
*.pth

# Fichiers temporaires
temp/
outputs/
tmp/
*.wav
*.mp4
*.mp3
*.avi
*.mkv
*.mov

# Fichiers de test
test_*.py
test_*/
poids*.py
poids*.txt

# Logs
*.log

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
"""
    gitignore = DOSSIER / ".gitignore"
    with open(gitignore, "w") as f:
        f.write(contenu)
    print("✅ .gitignore créé")


def creer_requirements():
    """Vérifie que requirements.txt existe."""
    req = DOSSIER / "requirements.txt"
    if req.exists():
        print("✅ requirements.txt existe")
    else:
        contenu = """streamlit>=1.35.0
faster-whisper>=1.0.0
chatterbox-tts>=0.1.7
demucs>=4.0.0
librosa>=0.10.0
soundfile>=0.12.0
ffmpeg-python>=0.2.0
numpy==1.26.4
scipy>=1.12.0
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.40.0
sentencepiece>=0.1.99
sacremoses>=0.1.1
"""
        with open(req, "w") as f:
            f.write(contenu)
        print("✅ requirements.txt créé")


def initialiser_git():
    """Initialise le repo git local si nécessaire."""
    git_dir = DOSSIER / ".git"
    if git_dir.exists():
        print("✅ Repo Git déjà initialisé")
        return True

    print("Initialisation Git...")
    ok, _ = run("git init")
    if not ok:
        print("❌ Erreur git init")
        return False

    ok, _ = run('git config user.email "minyogog@chariow.com"')
    ok, _ = run('git config user.name "Minyogog J."')
    print("✅ Git initialisé")
    return True


def creer_repo_github():
    """Crée le repo sur GitHub si inexistant."""
    url = f"https://github.com/{GITHUB_USERNAME}/{REPO_NAME}"

    # Vérifier si le repo existe
    ok, _ = run(f"gh repo view {GITHUB_USERNAME}/{REPO_NAME}")
    if ok:
        print(f"✅ Repo existe déjà : {url}")
        return url

    # Créer le repo
    print(f"Création du repo GitHub : {REPO_NAME}...")
    ok, out = run(
        f"gh repo create {REPO_NAME} "
        f"--public "
        f"--description 'VoiceSync Dubber — Doublage automatique multilingue' "
        f"--confirm"
    )

    if not ok:
        # Essai sans --confirm (nouvelle version CLI)
        ok, out = run(
            f"gh repo create {REPO_NAME} "
            f"--public "
            f"--description 'VoiceSync Dubber — Doublage automatique multilingue'"
        )

    if ok:
        print(f"✅ Repo créé : {url}")
        return url
    else:
        print(f"⚠️ Erreur création : {out[:200]}")
        print(f"   Crée manuellement sur github.com")
        return url


def connecter_remote(url: str):
    """Connecte le repo local au remote GitHub."""
    remote_url = f"https://github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"

    # Vérifier si remote existe
    ok, out = run("git remote -v")
    if "origin" in out:
        # Mettre à jour l'URL
        run(f"git remote set-url origin {remote_url}")
        print(f"✅ Remote mis à jour : {remote_url}")
    else:
        # Ajouter remote
        ok, _ = run(f"git remote add origin {remote_url}")
        print(f"✅ Remote ajouté : {remote_url}")


def committer_et_pusher():
    """Ajoute, commit et push tous les fichiers."""

    # Ajouter tous les fichiers
    print("Ajout des fichiers...")
    run("git add .")

    # Vérifier s'il y a des changements
    ok, out = run("git status --porcelain")
    if not out.strip():
        print("ℹ️ Aucun changement à committer")
        return True

    # Commit
    print("Commit...")
    ok, out = run(f'git commit -m "{MESSAGE_COMMIT}"')
    if not ok and "nothing to commit" not in out:
        print(f"⚠️ Commit : {out[:100]}")

    # Définir branche main
    run("git branch -M main")

    # Push
    print("Push vers GitHub...")
    ok, out = run("git push -u origin main")

    if not ok:
        print("Tentative force push...")
        ok, out = run("git push -u origin main --force")

    if ok:
        print(
            f"✅ Code pushé sur GitHub !\n"
            f"   → https://github.com/{GITHUB_USERNAME}/{REPO_NAME}"
        )
        return True
    else:
        print(f"❌ Erreur push : {out[:300]}")
        return False


def afficher_bilan(succes: bool):
    """Affiche le bilan final."""
    print(f"\n{'='*50}")
    if succes:
        print("  ✅ DÉPLOIEMENT GITHUB RÉUSSI !")
        print(f"{'='*50}")
        print(f"""
  🔗 Ton repo :
  https://github.com/{GITHUB_USERNAME}/{REPO_NAME}

  📋 Prochaines étapes :
  1. Vérifie le repo sur github.com
  2. Lance : python deploy_huggingface.py
     pour déployer sur Hugging Face
        """)
    else:
        print("  ❌ DÉPLOIEMENT ÉCHOUÉ")
        print(f"{'='*50}")
        print("""
  💡 Solutions :
  1. Vérifie : gh auth login
  2. Vérifie ta connexion internet
  3. Crée le repo manuellement sur github.com
        """)


# ═══════════════════════════════════════
# PROGRAMME PRINCIPAL
# ═══════════════════════════════════════
def main():
    print("=" * 50)
    print("  🚀 DÉPLOIEMENT GITHUB — VoiceSync Dubber")
    print("=" * 50)
    print(f"  Dossier : {DOSSIER}")
    print(f"  Repo    : {GITHUB_USERNAME}/{REPO_NAME}")

    # Étape 1 — Vérifications
    etape(1, "Vérification des outils")
    if not verifier_git():
        sys.exit(1)
    if not verifier_authentification():
        print("\n💡 Lance d'abord : gh auth login")
        sys.exit(1)

    # Étape 2 — Préparation fichiers
    etape(2, "Préparation des fichiers")
    creer_gitignore()
    creer_requirements()

    # Étape 3 — Git local
    etape(3, "Initialisation Git local")
    if not initialiser_git():
        sys.exit(1)

    # Étape 4 — Repo GitHub
    etape(4, "Création repo GitHub")
    url = creer_repo_github()

    # Étape 5 — Connexion remote
    etape(5, "Connexion remote GitHub")
    connecter_remote(url)

    # Étape 6 — Push
    etape(6, "Commit et Push")
    succes = committer_et_pusher()

    # Bilan
    afficher_bilan(succes)


if __name__ == "__main__":
    main()