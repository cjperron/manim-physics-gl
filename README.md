# 🎬 Manim Physics — @lomejorphysics

Educational physics animations built with [Manim](https://www.manim.community/) for [@lomejorphysics](https://instagram.com/lomejorphysics) — an Instagram channel dedicated to showing the mathematical foundations behind physical phenomena.

## 🎥 Animations

| Topic | Description | Key Concepts |
|-------|-------------|--------------|
| [**Gauss's Law**](./ley_de_gauss/) | Full derivation with 3D Gaussian surface, vector field visualization, and step-by-step math | ∇·E = ρ/ε₀, flux integral, spherical symmetry |
| [**Electric Field**](./campo_electrico/) | Vector field visualization with ArrowVectorField + point charge interactions | Coulomb's law, superposition, field lines |
| [**Electric Potential**](./potencial_electrico/) | Scalar potential visualization with equipotential surfaces | V = kQ/r, gradient → E, work-energy |
| [**Electric Dipole**](./dipolo_electrico/) | Dipole field and potential with 3D visualization | p = qd, far-field approximation, torque |
| [**Electrostatic Energy**](./energia_electrostatica/) | Energy stored in charge distributions and capacitors | U = ½ε₀∫E²dV, capacitance |

## ⭐ Featured

The **Gauss's Law** animation reached **400+ likes** on Instagram in 2 days — the channel's most successful video.

## 🛠️ Tech Stack

- **Manim** v0.18.1 — Mathematical animation engine (3Blue1Brown style)
- **Python** — All animations are pure Python
- **LaTeX** — Equations rendered with MathTex
- **3D Scenes** — ThreeDScene for Gaussian surfaces and spatial visualizations

## 🚀 How to Run

### Requisitos del sistema

Además de Python y `uv`, Manim necesita LaTeX y FFmpeg. En Arch Linux,
CachyOS y derivados:

```bash
sudo pacman -S --needed ffmpeg texlive-basic texlive-latexextra texlive-latexrecommended
latex --version
```

Después, desde la raíz del repositorio:

```bash
# Install dependencies and create the virtual environment
uv sync

# Preview the electric field (low quality, fast)
uv run manim -ql campo_electrico/campo_electrico.py CampoElectrico

# Render the electric field in desktop quality (1920x1080, 16:9, 60 fps)
uv run manim -qh campo_electrico/campo_electrico.py CampoElectrico
```

`uv sync` uses Python 3.11 or 3.12 and installs the exact dependency versions
recorded in `uv.lock`. For example, to render Gauss's Law:

```bash
uv run manim -ql ley_de_gauss/ley_de_gauss.py LeyDeGauss
```

Para renderizar todos los videos en 1920×1080 (16:9):

```bash
uv run manim -qh campo_electrico/campo_electrico.py CampoElectrico
uv run manim -qh campo_electrico/campo_electrico_big.py CampoElectrico
uv run manim -qh ley_de_gauss/ley_de_gauss.py LeyDeGauss
uv run manim -qh dipolo_electrico/dipolo_electrico.py DipoloElectrico
uv run manim -qh potencial_electrico/potencial_electrico.py PotencialElectrico
uv run manim -qh energia_electrostatica/energia_electrostatica.py EnergiaElectrostatica
```

### ⚡ Versión ManimGL (renders rápidos)

Las escenas portadas a [ManimGL](https://github.com/3b1b/manim) viven junto a las
originales con sufijo `_gl.py`. Usan `manimgl_compat/`, una capa que reproduce el
comportamiento de ManimCE 0.18.1 (tamaños, colores, timing, geometría) sobre el
motor OpenGL: el código de cada escena es el mismo que el de CE. Correr siempre
desde la raíz del repo (ahí está `custom_config.yml`):

```bash
uv run manimgl ley_de_gauss/ley_de_gauss_gl.py LeyDeGauss -w      # 1920x1080 60fps → media/manimgl/videos/
uv run manimgl ley_de_gauss/ley_de_gauss_gl.py LeyDeGauss -w -l   # preview 480p
uv run manimgl ley_de_gauss/ley_de_gauss_gl.py LeyDeGauss         # ventana interactiva
```

| Escena | ManimGL |
|--------|---------|
| Ley de Gauss | ✅ `ley_de_gauss/ley_de_gauss_gl.py` |
| Campo Eléctrico, Potencial, Dipolo, Energía | ⏳ pendientes |

## 🎨 Style

- **Dark background** with warm color palette (gold, orange, teal)
- **Step-by-step mathematical derivations** — no shortcuts
- **Vector field visualizations** using `ArrowVectorField`
- **3D scenes** with camera rotation for spatial concepts
- **Branding** — @lomejorphysics at start and end of each video

---

<div align="center">

*Made with ❤️ and physics — [@lomejorphysics](https://instagram.com/lomejorphysics)*

</div>
