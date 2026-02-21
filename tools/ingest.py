#!/usr/bin/env python3
"""
Ingest pipeline: Source-Images -> web-ready asset + per-work Markdown + README index.
- Detect new images in source_dir
- Convert/resize to web format
- OCR via tesseract (optional)
- Prompt user for metadata
- Write works/<slug>.md
- Copy images/<slug>.<ext>
- Update README.md index
- Commit & push

State is stored in repo_dir/.ingest/state.json and manifest.json
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image, ImageOps

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def sh(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, text=True, capture_output=True)


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "untitled"


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs(repo: Path) -> Dict[str, Path]:
    d = {
        "images": repo / "images",
        "works": repo / "works",
        "ingest": repo / ".ingest",
    }
    for k, p in d.items():
        p.mkdir(parents=True, exist_ok=True)
    return d


def load_json(p: Path, default):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return default


def save_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class WorkMeta:
    title: str
    year: int
    creator: str
    license: str
    language: str
    keywords: List[str]
    description: str
    transcription: str
    source_filename: str
    image_path: str  # repo-relative
    work_md_path: str  # repo-relative


def find_new_images(source_dir: Path, seen_hashes: set) -> List[Path]:
    imgs = []
    for p in sorted(source_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        h = file_sha256(p)
        if h not in seen_hashes:
            imgs.append(p)
    return imgs


def web_convert(
    src: Path,
    dst: Path,
    max_width: int,
    fmt: str,
    webp_quality: int,
    jpeg_quality: int,
    png_optimize: bool,
) -> Tuple[int, int]:
    img = Image.open(src)
    img = ImageOps.exif_transpose(img)  # respect orientation
    w, h = img.size
    if w > max_width:
        new_h = int(h * (max_width / w))
        img = img.resize((max_width, new_h), Image.LANCZOS)
        w, h = img.size

    fmt = fmt.lower()
    if fmt == "webp":
        img.save(dst, format="WEBP", quality=webp_quality, method=6)
    elif fmt == "jpg" or fmt == "jpeg":
        if img.mode in ("RGBA", "LA"):
            # flatten alpha for JPEG
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
        img.save(dst, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)
    elif fmt == "png":
        # keep alpha if exists
        img.save(dst, format="PNG", optimize=png_optimize)
    else:
        raise ValueError(f"Unsupported output_format: {fmt}")
    return w, h


def run_tesseract_ocr(image_path: Path, lang: str) -> str:
    # Use tesseract to stdout
    # Note: some tesseract builds support 'stdout' target; if not, we use temp file.
    try:
        proc = sh(["tesseract", str(image_path), "stdout", "-l", lang], check=False)
        if proc.returncode == 0:
            return (proc.stdout or "").strip()
    except FileNotFoundError:
        return ""
    # fallback with temp file
    tmpbase = image_path.parent / (image_path.stem + "_ocr_tmp")
    try:
        sh(["tesseract", str(image_path), str(tmpbase), "-l", lang], check=False)
        txt = (tmpbase.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")).strip()
        return txt
    except Exception:
        return ""
    finally:
        for ext in (".txt", ".log"):
            try:
                tmpbase.with_suffix(ext).unlink(missing_ok=True)  # py3.8+: missing_ok
            except Exception:
                pass


def prompt_list(prompt: str) -> List[str]:
    raw = input(prompt).strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def prompt_multiline(prompt: str) -> str:
    print(prompt)
    print("(Ende mit einer einzelnen Zeile: .)")
    lines = []
    while True:
        line = input()
        if line.strip() == ".":
            break
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def prompt_meta(
    default_title: str,
    default_year: int,
    creator: str,
    default_license: str,
    language: str,
    ocr_text: str,
    source_filename: str,
) -> WorkMeta:
    print("\n" + "=" * 72)
    print(f"Neues Bild: {source_filename}")
    print("=" * 72)

    title = input(f"Titel [{default_title}]: ").strip() or default_title
    year_raw = input(f"Jahr [{default_year}]: ").strip()
    year = int(year_raw) if year_raw else default_year

    lic = input(f"Lizenz [{default_license}]: ").strip() or default_license
    kws = prompt_list("Keywords (Komma-getrennt) [optional]: ")

    print("\nOCR-Transkription (Vorschlag):")
    print("-" * 72)
    print(ocr_text if ocr_text else "(leer)")
    print("-" * 72)
    transcription = prompt_multiline("Transkription final eingeben/korrektieren (oder '.' für leer):")
    if transcription == "":
        # if user entered '.' immediately, keep OCR text? -> no, explicit empty means empty
        transcription = ocr_text

    description = prompt_multiline("Kurzbeschreibung / Kontext (2–6 Zeilen empfohlen):")

    # Placeholders for paths filled later
    return WorkMeta(
        title=title,
        year=year,
        creator=creator,
        license=lic,
        language=language,
        keywords=kws,
        description=description,
        transcription=transcription,
        source_filename=source_filename,
        image_path="",
        work_md_path="",
    )


def render_work_md(meta: WorkMeta) -> str:
    # Minimal but strong: frontmatter + readable sections
    kw_line = ", ".join(meta.keywords) if meta.keywords else ""
    front = {
        "title": meta.title,
        "creator": meta.creator,
        "year": meta.year,
        "license": meta.license,
        "language": meta.language,
        "keywords": meta.keywords,
        "image": meta.image_path,
        "source_filename": meta.source_filename,
        "generated": str(date.today()),
    }
    fm = "---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip() + "\n---\n"

    body = []
    body.append(f"# {meta.title}\n")
    body.append(f"![{meta.title}](/" + meta.image_path.replace("\\", "/") + ")\n")
    body.append(f"**Künstler:** {meta.creator}  \n**Jahr:** {meta.year}  \n**Lizenz:** {meta.license}\n")
    if kw_line:
        body.append(f"**Begriffe:** {kw_line}\n")

    body.append("## Text im Bild (Transkription)\n")
    body.append(meta.transcription.strip() + "\n" if meta.transcription.strip() else "_(keine Transkription)_\n")

    body.append("## Kurzbeschreibung\n")
    body.append(meta.description.strip() + "\n" if meta.description.strip() else "_(keine Beschreibung)_\n")

    return fm + "\n".join(body).rstrip() + "\n"


def update_readme(readme_path: Path, works: List[Tuple[str, str]]) -> None:
    """
    works: list of (title, md_relpath)
    README keeps an index section between markers.
    """
    header = "# Infologische Bildtafeln\n\nOffenes Roharchiv (Bild + Transkription + Kontext) für maschinenlesbare Auffindbarkeit.\n\n"
    marker_a = "<!-- INDEX:BEGIN -->"
    marker_b = "<!-- INDEX:END -->"

    index_lines = [marker_a, "", "## Werke", ""]
    for title, md in sorted(works, key=lambda x: x[0].lower()):
        index_lines.append(f"- [{title}]({md})")
    index_lines += ["", marker_b, ""]

    if readme_path.exists():
        txt = readme_path.read_text(encoding="utf-8")
    else:
        txt = header + "\n".join(index_lines)

    if marker_a in txt and marker_b in txt:
        pre = txt.split(marker_a)[0].rstrip() + "\n"
        post = txt.split(marker_b)[1].lstrip()
        txt = pre + "\n".join(index_lines) + post
    else:
        txt = header + "\n".join(index_lines)

    readme_path.write_text(txt, encoding="utf-8")


def git_commit_push(repo: Path, message: str, push: bool) -> None:
    sh(["git", "add", "-A"], cwd=repo)
    # only commit if there are changes
    st = sh(["git", "status", "--porcelain"], cwd=repo)
    if not st.stdout.strip():
        print("git: keine Änderungen – kein Commit.")
        return
    sh(["git", "commit", "-m", message], cwd=repo)
    if push:
        sh(["git", "push"], cwd=repo)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: ingest.py tools/ingest_config.yaml")
        return 2

    cfg_path = Path(sys.argv[1]).expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    source_dir = Path(cfg["source_dir"]).expanduser().resolve()
    repo_dir = Path(cfg["repo_dir"]).expanduser().resolve()

    if not source_dir.exists():
        print(f"ERROR: source_dir not found: {source_dir}")
        return 2
    if not (repo_dir / ".git").exists():
        print(f"ERROR: repo_dir is not a git repo: {repo_dir}")
        return 2

    dirs = ensure_dirs(repo_dir)
    state_path = dirs["ingest"] / "state.json"
    manifest_path = dirs["ingest"] / "manifest.json"

    state = load_json(state_path, default={"seen_hashes": [], "works": []})
    seen_hashes = set(state.get("seen_hashes", []))
    works_state = state.get("works", [])

    new_imgs = find_new_images(source_dir, seen_hashes)
    if not new_imgs:
        print("Keine neuen Bilder im Source-Verzeichnis.")
        return 0

    output_format = str(cfg.get("output_format", "webp")).lower()
    max_width = int(cfg.get("max_width", 2000))
    webp_quality = int(cfg.get("webp_quality", 82))
    jpeg_quality = int(cfg.get("jpeg_quality", 85))
    png_optimize = bool(cfg.get("png_optimize", True))

    run_ocr = bool(cfg.get("run_ocr", True))
    tess_lang = str(cfg.get("tesseract_lang", "deu"))
    creator = str(cfg.get("creator", ""))
    language = str(cfg.get("language", "de"))
    default_license = str(cfg.get("default_license", "CC BY-SA 4.0"))
    dry_run = bool(cfg.get("dry_run", False))
    auto_commit = bool(cfg.get("auto_commit", True))
    auto_push = bool(cfg.get("auto_push", True))

    manifest = load_json(manifest_path, default={"works": []})

    for src in new_imgs:
        h = file_sha256(src)
        base_default_title = src.stem.replace("_", " ").replace("-", " ").strip()
        default_title = base_default_title[:60] if base_default_title else "Untitled"
        default_year = date.today().year

        # Stage to a temp web asset to OCR against (better than huge originals)
        tmp_out = dirs["ingest"] / f"_tmp_{src.stem}.{output_format}"
        if tmp_out.exists():
            tmp_out.unlink()

        if not dry_run:
            web_convert(
                src=src,
                dst=tmp_out,
                max_width=max_width,
                fmt=output_format,
                webp_quality=webp_quality,
                jpeg_quality=jpeg_quality,
                png_optimize=png_optimize,
            )

        ocr_text = ""
        if run_ocr and not dry_run:
            ocr_text = run_tesseract_ocr(tmp_out, lang=tess_lang)

        meta = prompt_meta(
            default_title=default_title,
            default_year=default_year,
            creator=creator,
            default_license=default_license,
            language=language,
            ocr_text=ocr_text,
            source_filename=src.name,
        )

        slug = slugify(f"{meta.title}-{meta.year}")
        # Avoid collisions
        img_rel = f"images/{slug}.{output_format}"
        md_rel = f"works/{slug}.md"

        img_dst = repo_dir / img_rel
        md_dst = repo_dir / md_rel

        # If exists, ask for suffix
        n = 2
        while img_dst.exists() or md_dst.exists():
            slug2 = f"{slug}-{n}"
            img_rel = f"images/{slug2}.{output_format}"
            md_rel = f"works/{slug2}.md"
            img_dst = repo_dir / img_rel
            md_dst = repo_dir / md_rel
            n += 1

        meta.image_path = img_rel
        meta.work_md_path = md_rel

        # Write outputs
        if dry_run:
            print(f"[dry-run] would write {img_rel} and {md_rel}")
        else:
            shutil.move(str(tmp_out), str(img_dst))
            md_dst.write_text(render_work_md(meta), encoding="utf-8")

        # Update state / manifest
        works_state.append(
            {
                "title": meta.title,
                "year": meta.year,
                "slug": Path(meta.work_md_path).stem,
                "md": meta.work_md_path.replace("\\", "/"),
                "image": meta.image_path.replace("\\", "/"),
                "sha256": h,
            }
        )
        seen_hashes.add(h)
        manifest["works"] = works_state

        # Move original source into an archive subfolder to prevent re-ingest
        if not dry_run:
            archive_dir = source_dir / "_ingested"
            archive_dir.mkdir(exist_ok=True)
            shutil.move(str(src), str(archive_dir / src.name))

        # Cleanup temp if still exists
        try:
            tmp_out.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass

    # Update README index from works_state
    works_list = [(w["title"], w["md"]) for w in works_state]
    if not dry_run:
        update_readme(repo_dir / "README.md", works_list)

    # Persist state/manifest
    state["seen_hashes"] = sorted(seen_hashes)
    state["works"] = works_state
    if not dry_run:
        save_json(state_path, state)
        save_json(manifest_path, manifest)


# --- sitemap.xml generieren ---
 #   from pathlib import Path
    import datetime

    BASE_URL = "https://nocard-code.github.io/NOT4BFLU55"
    urls = []
    for md in (repo_dir / "works").glob("*.md"):
        urls.append(f"{BASE_URL}/{md.as_posix()}")

    today = datetime.date.today().isoformat()
    xml = ["<?xml version='1.0' encoding='UTF-8'?>",
           "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"]
    for u in urls:
        xml += ["  <url>", f"    <loc>{u}</loc>", f"    <lastmod>{today}</lastmod>", "  </url>"]
    xml.append("</urlset>")

    (repo_dir / "sitemap.xml").write_text("\n".join(xml), encoding="utf-8")

    # git commit/push
    if auto_commit and not dry_run:
        msg = f"ingest: {len(new_imgs)} work(s) ({date.today().isoformat()})"
        git_commit_push(repo_dir, msg, push=auto_push)

    print("Fertig.")

# optionaler Ping (sanfter Stupser)
    try:
        sh(["curl", f"https://www.google.com/ping?sitemap={BASE_URL}/sitemap.xml"], check=False)
        sh(["curl", f"https://www.bing.com/ping?sitemap={BASE_URL}/sitemap.xml"], check=False)
    except Exception:
        pass

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
