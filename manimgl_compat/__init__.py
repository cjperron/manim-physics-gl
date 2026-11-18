"""
manimgl_compat — capa de compatibilidad Manim Community (CE 0.18.1) → ManimGL (1.7.2)
=====================================================================================

Permite escribir escenas para ManimGL con la MISMA API que usaban las escenas de
ManimCE de este repo, reproduciendo el comportamiento de CE (geometría, tamaños,
colores, timing y curvas de animación) para que el video resultante sea
prácticamente idéntico. Lo único que cambia es el motor de render (OpenGL en
lugar de Cairo): antialiasing e iluminación 3D.

Uso (desde la raíz del repo, donde está `custom_config.yml`):

    from manimgl_compat import *

Qué replica respecto de CE (cada punto fue verificado contra el código fuente
de manim 0.18.1 y medido en ambos motores):

  • Text / MathTex: misma fuente (CE usa la fuente por defecto de Pango, "Serif"),
    mismo tamaño (factores de escala calibrados), MathTex separado en partes
    igual que CE (`{{ }}`, argumentos múltiples) para que `eq[i]` indexe lo mismo.
  • Colores: la paleta de CE (algunos difieren de GL, p.ej. BLUE_E).
  • `smooth`: la curva sigmoide de CE (GL usa un polinomio distinto).
  • Write (lag_ratio y color del contorno), FadeIn(scale=…) (GL lo invierte),
    Create, TransformMatchingTex (algoritmo de CE).
  • Arrow / Vector / ArrowVectorField / Cross / SurroundingRectangle /
    RoundedRectangle con la geometría de CE (puntas, grosor según largo,
    orden de trazado para Create).
  • ThreeDAxes, Surface, Sphere, Arrow3D con la estructura de CE.
  • ThreeDScene: cámara (phi/theta, focal_distance=20), move_camera,
    rotación ambiental y add_fixed_in_frame_mobjects con la convención de CE.
"""

from __future__ import annotations

import inspect
import itertools as it
import re

import numpy as np

import manimlib
from manimlib import *  # noqa: F401,F403  (re-exporta toda la API de GL)
import manimlib.utils.rate_functions as _gl_rate_functions
from manimlib import Arrow as _GLArrow  # noqa: F401
from manimlib import FadeIn as _GLFadeIn
from manimlib import FadeOut as _GLFadeOut
from manimlib import Scene as _GLScene
from manimlib import Surface as _GLSurface
from manimlib import Text as _GLText
from manimlib import Write as _GLWrite
from manimlib.animation.creation import DrawBorderThenFill as _GLDrawBorderThenFill
from manimlib.mobject.svg.old_tex_mobject import SingleStringTex as _GLSingleStringTex
from manimlib.utils.tex_file_writing import latex_to_svg

__all__ = [name for name in dir(manimlib) if not name.startswith("_")]


def _export(obj):
    """Registra `obj` en __all__ (sobrescribe el nombre de GL si existía)."""
    if obj.__name__ not in __all__:
        __all__.append(obj.__name__)
    return obj


# ═════════════════════════════════════════════════════════════════════════════
# Paleta de colores de ManimCE 0.18.1 (manim.utils.color.manim_colors)
# ═════════════════════════════════════════════════════════════════════════════
_CE_COLORS = dict(
    BLACK="#000000", BLUE="#58C4DD", BLUE_A="#C7E9F1", BLUE_B="#9CDCEB",
    BLUE_C="#58C4DD", BLUE_D="#29ABCA", BLUE_E="#236B8E", DARKER_GRAY="#222222",
    DARKER_GREY="#222222", DARK_BLUE="#236B8E", DARK_BROWN="#8B4513",
    DARK_GRAY="#444444", DARK_GREY="#444444", GOLD="#F0AC5F", GOLD_A="#F7C797",
    GOLD_B="#F9B775", GOLD_C="#F0AC5F", GOLD_D="#E1A158", GOLD_E="#C78D46",
    GRAY="#888888", GRAY_A="#DDDDDD", GRAY_B="#BBBBBB", GRAY_BROWN="#736357",
    GRAY_C="#888888", GRAY_D="#444444", GRAY_E="#222222", GREEN="#83C167",
    GREEN_A="#C9E2AE", GREEN_B="#A6CF8C", GREEN_C="#83C167", GREEN_D="#77B05D",
    GREEN_E="#699C52", GREY="#888888", GREY_A="#DDDDDD", GREY_B="#BBBBBB",
    GREY_BROWN="#736357", GREY_C="#888888", GREY_D="#444444", GREY_E="#222222",
    LIGHTER_GRAY="#DDDDDD", LIGHTER_GREY="#DDDDDD", LIGHT_BROWN="#CD853F",
    LIGHT_GRAY="#BBBBBB", LIGHT_GREY="#BBBBBB", LIGHT_PINK="#DC75CD",
    LOGO_BLACK="#343434", LOGO_BLUE="#525893", LOGO_GREEN="#87C2A5",
    LOGO_RED="#E07A5F", LOGO_WHITE="#ECE7E2", MAROON="#C55F73",
    MAROON_A="#ECABC1", MAROON_B="#EC92AB", MAROON_C="#C55F73",
    MAROON_D="#A24D61", MAROON_E="#94424F", ORANGE="#FF862F", PINK="#D147BD",
    PURE_BLUE="#0000FF", PURE_GREEN="#00FF00", PURE_RED="#FF0000",
    PURPLE="#9A72AC", PURPLE_A="#CAA3E8", PURPLE_B="#B189C6", PURPLE_C="#9A72AC",
    PURPLE_D="#715582", PURPLE_E="#644172", RED="#FC6255", RED_A="#F7A1A3",
    RED_B="#FF8080", RED_C="#FC6255", RED_D="#E65A4C", RED_E="#CF5044",
    TEAL="#5CD0B3", TEAL_A="#ACEAD7", TEAL_B="#76DDC0", TEAL_C="#5CD0B3",
    TEAL_D="#55C1A7", TEAL_E="#49A88F", WHITE="#FFFFFF", YELLOW="#FFFF00",
    YELLOW_A="#FFF1B6", YELLOW_B="#FFEA94", YELLOW_C="#FFFF00",
    YELLOW_D="#F4D345", YELLOW_E="#E8C11C",
)
globals().update(_CE_COLORS)
for _name in _CE_COLORS:
    if _name not in __all__:
        __all__.append(_name)

# Constantes de CE que no existen en GL
DEFAULT_ARROW_TIP_LENGTH = 0.35
DEFAULT_SCALAR_FIELD_COLORS = [BLUE_E, GREEN, YELLOW, RED]  # noqa: F821
__all__ += ["DEFAULT_ARROW_TIP_LENGTH", "DEFAULT_SCALAR_FIELD_COLORS"]


# ═════════════════════════════════════════════════════════════════════════════
# Curvas de animación: `smooth` de CE (sigmoide). La de GL es un polinomio
# distinto, así que se reemplaza en todos los defaults de las animaciones.
# ═════════════════════════════════════════════════════════════════════════════
def _sigmoid(x: float) -> float:
    return 1.0 / (1 + np.exp(-x))


def smooth(t: float, inflection: float = 10.0) -> float:
    error = _sigmoid(-inflection / 2)
    return min(
        max((_sigmoid(inflection * (t - 0.5)) - error) / (1 - 2 * error), 0),
        1,
    )


def _install_ce_smooth() -> None:
    gl_smooth = _gl_rate_functions.smooth
    # Funciones compuestas (double_smooth, there_and_back, …) llaman a
    # `smooth` del módulo: con esto usan la versión de CE, igual que en CE.
    _gl_rate_functions.smooth = smooth
    # Los defaults `rate_func=smooth` quedaron capturados al definir cada clase.
    import manimlib.animation as anim_pkg
    import pkgutil
    import importlib

    for info in pkgutil.iter_modules(anim_pkg.__path__):
        module = importlib.import_module(f"manimlib.animation.{info.name}")
        if getattr(module, "smooth", None) is gl_smooth:
            module.smooth = smooth
        for obj in vars(module).values():
            if not inspect.isclass(obj):
                continue
            for attr in ("__init__",):
                fn = obj.__dict__.get(attr)
                if fn is None:
                    continue
                if fn.__defaults__:
                    fn.__defaults__ = tuple(
                        smooth if d is gl_smooth else d for d in fn.__defaults__
                    )
                if fn.__kwdefaults__:
                    for k, d in fn.__kwdefaults__.items():
                        if d is gl_smooth:
                            fn.__kwdefaults__[k] = smooth
    import manimlib.mobject.mobject as mob_module  # `.animate` builder
    if getattr(mob_module, "smooth", None) is gl_smooth:
        mob_module.smooth = smooth


_install_ce_smooth()
__all__.append("smooth")


def _ce_interpolate_mobject(self, alpha: float) -> None:
    """Escalonado (lag_ratio) como CE: solo cuentan las piezas con puntos.

    GL reparte el lag entre toda la familia, incluidos contenedores vacíos
    (el Text, cada parte de un MathTex, los VGroup), lo que corre el timing
    de Write/Create/FadeIn con lag respecto de CE.
    """
    families = self.families
    with_points = [mobs for mobs in families if any(m.has_points() for m in mobs)]
    n = len(with_points)
    t = self.time_spanned_alpha(alpha)
    for k, mobs in enumerate(with_points):
        self.interpolate_submobject(*mobs, self.get_sub_alpha(t, k, n))


manimlib.Animation.interpolate_mobject = _ce_interpolate_mobject


def _ce_svg_handle_commands(self) -> None:
    """Paths de SVG (glifos de Text/MathTex) segmentados como CE (Cairo).

    CE convierte cada segmento del SVG (línea, cuadrática, cúbica, cierre) en
    UNA cúbica; acá cada una se vuelve exactamente 2 cuadráticas de GL. Así
    Write/Create, que avanzan por número de curva, trazan los glifos al mismo
    ritmo que en CE.
    """
    import svgelements as se
    from manimlib.mobject.svg.svg_mobject import _convert_point_to_3d

    last_move = None
    last_true_move = None

    def cubic(a, h1, h2, b):
        nonlocal last_move
        if not self.has_points() or self.has_new_path_started() is False and not np.allclose(self.get_last_point(), a):
            self.start_new_path(a)
        _add_cubic_as_quads(self, (a, h1, h2, b))
        last_move = b

    for segment in self.path_obj:
        cls = segment.__class__
        if cls is se.Move:
            last_move = last_true_move = _convert_point_to_3d(*segment.end)
            self.start_new_path(last_move)
        elif cls is se.Line:
            a, b = last_move, _convert_point_to_3d(*segment.end)
            cubic(a, (2 * a + b) / 3, (a + 2 * b) / 3, b)
        elif cls is se.QuadraticBezier:
            a = last_move
            c = _convert_point_to_3d(*segment.control)
            b = _convert_point_to_3d(*segment.end)
            cubic(a, (a + 2 * c) / 3, (2 * c + b) / 3, b)
        elif cls is se.CubicBezier:
            cubic(
                last_move,
                _convert_point_to_3d(*segment.control1),
                _convert_point_to_3d(*segment.control2),
                _convert_point_to_3d(*segment.end),
            )
        elif cls is se.Close:
            if np.linalg.norm(last_move - last_true_move) > 0.0001:
                a, b = last_move, last_true_move
                cubic(a, (2 * a + b) / 3, (a + 2 * b) / 3, b)
    # Igual que GL: descartar el "Z M" final que deja un path nuevo vacío
    if self.has_new_path_started():
        self.resize_points(self.get_num_points() - 2)


from manimlib.mobject.svg.svg_mobject import VMobjectFromSVGPath as _GLVMobjectFromSVGPath  # noqa: E402

_GLVMobjectFromSVGPath.handle_commands = _ce_svg_handle_commands


# ═════════════════════════════════════════════════════════════════════════════
# Helpers internos
# ═════════════════════════════════════════════════════════════════════════════
def _norm(v) -> float:
    return float(np.linalg.norm(v))


def _ce_cartesian_to_spherical(vec):
    """manim.utils.space_ops.cartesian_to_spherical (CE)."""
    r = np.linalg.norm(vec)
    if r == 0:
        return np.zeros(3)
    phi = np.arccos(vec[2] / r)
    theta = np.arctan2(vec[1], vec[0])
    return np.array([r, theta, phi])


def _ce_angle_of_vector(vec) -> float:
    return float(np.angle(complex(vec[0], vec[1])))


def _split_cubic(p0, p1, p2, p3):
    """De Casteljau en t=0.5."""
    p01, p12, p23 = (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p3) / 2
    p012, p123 = (p01 + p12) / 2, (p12 + p23) / 2
    mid = (p012 + p123) / 2
    return (p0, p01, p012, mid), (mid, p123, p23, p3)


def _add_cubic_as_quads(vmob: VMobject, cubic) -> None:  # noqa: F821
    """Agrega un tramo cúbico (estilo CE) como 2 cuadráticas de GL.

    Cada tramo cúbico de CE ↔ exactamente 2 cuadráticas, así Create
    (que avanza por cantidad de curvas) recorre el trazo al mismo ritmo.
    """
    for half in _split_cubic(*cubic):
        a0, h0, h1, a1 = half
        control = (3 * (h0 + h1) - (a0 + a1)) / 4
        vmob.add_quadratic_bezier_curve_to(control, a1)


def _ce_arc_cubic(start, end, angle):
    """Arco entre dos puntos como UNA cúbica (Arc de CE con num_components=2)."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if angle == 0:
        return start, start + (end - start) / 3, start + 2 * (end - start) / 3, end
    # Centro y radio del arco (misma convención que ArcBetweenPoints de CE)
    chord = end - start
    half = _norm(chord) / 2
    radius = half / np.sin(angle / 2)
    mid = (start + end) / 2
    perp = np.array([-chord[1], chord[0], 0.0]) / (2 * half)
    center = mid + perp * radius * np.cos(angle / 2)
    a0 = np.arctan2(*(start - center)[[1, 0]])
    angles = np.array([a0, a0 + angle])
    r = _norm(start - center)
    anchors = np.array([center + r * np.array([np.cos(a), np.sin(a), 0]) for a in angles])
    tangents = np.array([[-np.sin(a), np.cos(a), 0] for a in angles])
    # Arc de CE: handles = anchor ± (dθ/3)·tangente  (con num_components=2, dθ = angle)
    handle_len = (angle / 3) * r
    h1 = anchors[0] + tangents[0] * handle_len
    h2 = anchors[1] - tangents[1] * handle_len
    return anchors[0], h1, h2, anchors[1]


# ═════════════════════════════════════════════════════════════════════════════
# Gradientes de color como CE (interpolación RGB lineal sobre las piezas)
# ═════════════════════════════════════════════════════════════════════════════
def _ce_color_gradient(reference_colors, length_of_output):
    if length_of_output == 0:
        return []
    rgbs = [np.array(color_to_rgb(c)) for c in reference_colors]  # noqa: F821
    alphas = np.linspace(0, len(rgbs) - 1, length_of_output)
    floors = alphas.astype("int")
    alphas_mod1 = alphas % 1
    alphas_mod1[-1] = 1
    floors[-1] = len(rgbs) - 2
    return [rgb_to_hex(rgbs[i] + (rgbs[i + 1] - rgbs[i]) * a) for i, a in zip(floors, alphas_mod1)]  # noqa: F821


def _ce_set_submobject_colors_by_gradient(self, *colors):
    if len(colors) == 0:
        raise Exception("Need at least one color")
    if len(colors) == 1:
        return self.set_color(*colors)
    mobs = self.family_members_with_points()
    for mob, color in zip(mobs, _ce_color_gradient(colors, len(mobs))):
        mob.set_color(color, recurse=False)
    return self


manimlib.Mobject.set_submobject_colors_by_gradient = _ce_set_submobject_colors_by_gradient
manimlib.Mobject.set_color_by_gradient = _ce_set_submobject_colors_by_gradient
color_gradient = _ce_color_gradient
__all__.append("color_gradient")


# ═════════════════════════════════════════════════════════════════════════════
# Propiedades .width / .height / .depth de CE
# ═════════════════════════════════════════════════════════════════════════════
class _CESize:
    """`mob.width`, `mob.height`, `mob.depth` como en CE (GL usa get_width())."""

    @property
    def width(self):
        return self.get_width()

    @width.setter
    def width(self, value):
        if self.has_points() or self.submobjects:
            self.set_width(value)

    @property
    def height(self):
        return self.get_height()

    @height.setter
    def height(self, value):
        if self.has_points() or self.submobjects:
            self.set_height(value)

    @property
    def depth(self):
        return self.get_depth()

    @depth.setter
    def depth(self, value):
        if self.has_points() or self.submobjects:
            self.set_depth(value)


# ═════════════════════════════════════════════════════════════════════════════
# VGroup que, como en CE, puede contener objetos 3D (Arrow3D)
# ═════════════════════════════════════════════════════════════════════════════
_GLVGroup = manimlib.VGroup
_GLVMobject = manimlib.VMobject
_GLMobject = manimlib.Mobject


@_export
class VGroup(_CESize, _GLVGroup):
    """VGroup de GL que además acepta superficies (en CE Arrow3D es un VMobject).

    Si contiene superficies, los métodos de estilo se reparten hijo por hijo
    según su tipo. GL renderiza sin problema familias mixtas.
    """

    def __init__(self, *vmobjects, **kwargs):
        _GLVMobject.__init__(self, **kwargs)
        self._ingest_args(*vmobjects)
        if self.submobjects and isinstance(self.submobjects[0], _GLVMobject):
            self.uniforms.update(self.submobjects[0].uniforms)

    def add(self, *mobjects):
        if all(isinstance(m, _GLVMobject) for m in mobjects):
            return super().add(*mobjects)
        return _GLMobject.add(self, *mobjects)

    def _is_pure(self) -> bool:
        return all(isinstance(m, _GLVMobject) for m in self.get_family())

    def set_fill(self, color=None, opacity=None, border_width=None, recurse=True):
        if self._is_pure() or not recurse:
            return super().set_fill(color, opacity, border_width, recurse)
        super().set_fill(color, opacity, border_width, recurse=False)
        for sm in self.submobjects:
            if isinstance(sm, _GLVMobject):
                sm.set_fill(color, opacity, border_width)
            else:
                sm.set_color(color, opacity)
        return self

    def set_stroke(self, color=None, width=None, opacity=None, behind=None, flat=None, recurse=True):
        if self._is_pure() or not recurse:
            return super().set_stroke(color, width, opacity, behind, flat, recurse)
        super().set_stroke(color, width, opacity, behind, flat, recurse=False)
        for sm in self.submobjects:
            if isinstance(sm, _GLVMobject):
                sm.set_stroke(color, width, opacity, behind, flat)
        return self

    def set_color(self, color, opacity=None, recurse=True):
        if self._is_pure() or not recurse:
            return super().set_color(color, opacity, recurse)
        super().set_color(color, opacity, recurse=False)
        for sm in self.submobjects:
            sm.set_color(color, opacity)
        return self

    def set_opacity(self, opacity, recurse=True):
        if self._is_pure() or not recurse:
            return super().set_opacity(opacity, recurse)
        super().set_opacity(opacity, recurse=False)
        for sm in self.submobjects:
            sm.set_opacity(opacity)
        return self


# ═════════════════════════════════════════════════════════════════════════════
# Texto
# ═════════════════════════════════════════════════════════════════════════════
# Calibración medida con manim 0.18.1 vs manimgl 1.7.2 (misma fuente/LaTeX):
#   Text(font_size=f) de CE  = Text(font_size=f) de GL × 1.3706
#   MathTex(font_size=f) CE  = Tex(font_size=f) de GL  × 1.0417
TEXT_SCALE_CE = 1.3706
TEX_SCALE_CE = 1.0417
# CE crea los Text sin familia de fuente → Pango usa su fuente por defecto, "Serif".
CE_DEFAULT_FONT = "Serif"


def _ce_line_spacing_to_gl_lsh(line_spacing: float) -> float:
    """Convierte `line_spacing` de Text (CE) al `lsh` de Text (GL).

    Distancia entre renglones medida (font_size=48, en unidades de CE):
      CE: 0.49297 + 0.5·line_spacing        (line_spacing=-1 equivale a 0.3)
      GL: 0.53829 + 0.54479·lsh             (ya escalado por TEXT_SCALE_CE)
    """
    if line_spacing == -1:
        line_spacing = 0.3
    lsh = (0.49297 + 0.5 * line_spacing - 0.53829) / 0.54479
    return max(lsh, 1e-3)  # GL trata lsh=0 como "default"


@_export
class Text(_CESize, _GLText):
    """`Text` con métricas de CE (fuente, tamaño e interlineado)."""

    def __init__(
        self,
        text: str,
        font_size: float = 48,
        color=WHITE,  # noqa: F821
        line_spacing: float = -1,
        font: str = "",
        **kwargs,
    ):
        lsh = None
        if "\n" in text:
            lsh = _ce_line_spacing_to_gl_lsh(line_spacing)
        kwargs.setdefault("disable_ligatures", False)  # CE usa ligaduras (fi, fl, ffi)
        super().__init__(
            text,
            font_size=font_size,
            font=font or CE_DEFAULT_FONT,
            lsh=lsh,
            **kwargs,
        )
        self.scale(TEXT_SCALE_CE)
        self.set_color(color)


class _TexSVG(_CESize, _GLSingleStringTex):
    """Una cadena LaTeX en entorno align* (SingleStringMathTex de CE).

    `SingleStringTex` de manimgl 1.7.2 no le pasa el SVG a SVGMobject (está
    roto), así que acá se compila el LaTeX y se construye el SVGMobject a mano.
    """

    def __init__(self, tex_string: str, font_size: float = 48, fill_color=WHITE, **kwargs):  # noqa: F821
        self.tex_string = tex_string
        self.svg_default = dict(
            color=None, opacity=None, fill_color=WHITE, fill_opacity=None,  # noqa: F821
            stroke_width=None, stroke_color=None, stroke_opacity=None,
        )
        self.path_string_config = dict()
        self.font_size = font_size
        self.alignment = R"\centering"
        self.math_mode = True
        self.organize_left_to_right = False
        self.template = ""
        self.additional_preamble = ""
        svg_string = latex_to_svg(  # noqa: F821
            self.get_tex_file_body(tex_string), self.template, self.additional_preamble,
            short_tex=tex_string,
        )
        SVGMobject.__init__(  # noqa: F821
            self,
            svg_string=svg_string,
            fill_color=fill_color,
            fill_opacity=1.0,
            stroke_width=0,
            svg_default=self.svg_default,
            path_string_config=self.path_string_config,
            **kwargs,
        )
        self.scale(0.001 * self.font_size)


class _SingleStringMathTex(_TexSVG):
    """Una parte de un MathTex (equivale a SingleStringMathTex de CE)."""


@_export
class MathTex(_TexSVG):
    """`MathTex` de CE: entorno align*, partes por argumento y `{{ }}`.

    `eq[i]` devuelve la i-ésima parte, exactamente como en CE.
    """

    def __init__(
        self,
        *tex_strings: str,
        arg_separator: str = " ",
        substrings_to_isolate=None,
        tex_to_color_map=None,
        font_size: float = 48,
        color=WHITE,  # noqa: F821
        **kwargs,
    ):
        self.arg_separator = arg_separator
        self.substrings_to_isolate = list(substrings_to_isolate or [])
        self.tex_to_color_map = dict(tex_to_color_map or {})
        self.tex_strings = self._break_up_tex_strings(tex_strings)
        super().__init__(
            arg_separator.join(self.tex_strings),
            font_size=font_size * TEX_SCALE_CE,
            fill_color=color,
            **kwargs,
        )
        self._break_up_by_substrings()
        self.set_color(color)
        for tex, tex_color in self.tex_to_color_map.items():
            for part in self.submobjects:
                if part.tex_string.strip() == tex.strip():
                    part.set_color(tex_color)

    def _break_up_tex_strings(self, tex_strings):
        tex_strings = [re.split("{{(.*?)}}", str(t)) for t in tex_strings]
        tex_strings = sum(tex_strings, [])
        patterns = [
            f"({re.escape(ss)})"
            for ss in it.chain(self.substrings_to_isolate, self.tex_to_color_map.keys())
        ]
        pattern = "|".join(patterns)
        if pattern:
            pieces = []
            for s in tex_strings:
                pieces.extend(re.split(pattern, s))
        else:
            pieces = tex_strings
        return [p for p in pieces if p]

    def _break_up_by_substrings(self):
        glyphs = list(self.submobjects)
        new_submobjects = []
        curr_index = 0
        sep_len = len("".join(self.arg_separator.split()))
        for tex_string in self.tex_strings:
            sub = _SingleStringMathTex(tex_string)
            num_submobs = len(sub.submobjects)
            new_index = curr_index + num_submobs + sep_len
            if num_submobs == 0:
                last = min(curr_index, len(glyphs) - 1)
                sub.set_submobjects([])
                if glyphs and sub.has_points():
                    sub.move_to(glyphs[last], RIGHT)  # noqa: F821
            else:
                sub.set_submobjects(glyphs[curr_index:new_index])
            sub.tex_string = tex_string
            new_submobjects.append(sub)
            curr_index = new_index
        self.set_submobjects(new_submobjects)
        return self


# ═════════════════════════════════════════════════════════════════════════════
# Geometría con el comportamiento de CE
# ═════════════════════════════════════════════════════════════════════════════
@_export
class Circle(manimlib.Circle):
    """Circle de CE: `color=` pinta trazo y relleno (en GL lo pisa stroke_color=RED)."""

    def __init__(self, radius: float = 1.0, color=RED, **kwargs):  # noqa: F821
        kwargs.setdefault("stroke_color", color)
        kwargs.setdefault("fill_color", color)
        super().__init__(radius=radius, **kwargs)


@_export
class Dot(manimlib.Dot):
    """Dot de CE: `color=` pinta el relleno (en GL lo pisa fill_color=WHITE)."""

    def __init__(self, point=ORIGIN, radius: float = 0.08, color=WHITE, **kwargs):  # noqa: F821
        kwargs.setdefault("fill_color", color)
        kwargs.setdefault("stroke_color", color)
        kwargs.setdefault("stroke_width", 0)
        super().__init__(point, radius=radius, **kwargs)


def _ce_triangle_tip(length: float, width: float, color) -> VMobject:  # noqa: F821
    """ArrowTriangleFilledTip de CE: triángulo con la punta en points[0]."""
    angles = PI + np.arange(3) * TAU / 3  # noqa: F821  (start_angle = PI)
    verts = np.array([[np.cos(a), np.sin(a), 0.0] for a in angles])
    # stretch_to_fit_width(length) / stretch_to_fit_height(width) sobre el centro
    center = (verts.min(0) + verts.max(0)) / 2
    size = verts.max(0) - verts.min(0)
    verts = center + (verts - center) * np.array([length / size[0], width / size[1], 1])
    tip = Polygon(*verts, color=color, fill_opacity=1, stroke_width=0)  # noqa: F821
    return tip


def _tip_point(tip):
    return tip.get_points()[0].copy()


def _tip_base(tip):
    v = tip.get_vertices()
    return (v[1] + v[2]) / 2


@_export
class Arrow(_CESize, VMobject):  # noqa: F821
    """`Arrow` de CE: línea + punta triangular rellena.

    El largo de la punta y el grosor del trazo dependen del largo de la flecha
    (max_tip_length_to_length_ratio / max_stroke_width_to_length_ratio), y
    `scale()` mantiene el tamaño de la punta igual que en CE (esto define cómo
    se ve GrowArrow).
    """

    def __init__(
        self,
        start=LEFT,  # noqa: F821
        end=RIGHT,  # noqa: F821
        buff: float = 0.25,
        stroke_width: float = 6,
        max_tip_length_to_length_ratio: float = 0.25,
        max_stroke_width_to_length_ratio: float = 5,
        tip_length: float = DEFAULT_ARROW_TIP_LENGTH,
        color=WHITE,  # noqa: F821
        **kwargs,
    ):
        super().__init__(color=color, stroke_width=stroke_width, **kwargs)
        self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio
        self.max_stroke_width_to_length_ratio = max_stroke_width_to_length_ratio
        self.tip_length = tip_length
        self.buff = buff
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        # Line._account_for_buff
        length = _norm(end - start)
        if buff != 0 and length >= 2 * buff:
            unit = (end - start) / length
            start, end = start + unit * buff, end - unit * buff
        self._set_line(start, end)
        self.initial_stroke_width = stroke_width
        self.tip = None
        tl = min(self.tip_length, self.max_tip_length_to_length_ratio * self._line_length())
        tip = _ce_triangle_tip(tl, tl, self.get_stroke_color())
        self._add_tip(tip)
        self._set_stroke_width_from_length()

    # ── línea ──
    def _set_line(self, start, end):
        self.set_points_as_corners([start, end])

    def _line_start(self):
        return self.get_points()[0].copy()

    def _line_end(self):
        return self.get_points()[-1].copy()

    def _line_length(self):
        return _norm(self._line_end() - self._line_start())

    # ── API pública estilo CE ──
    def get_start(self):
        return self._line_start()

    def get_end(self):
        if self.tip is not None and self.tip in self.submobjects:
            return _tip_point(self.tip)
        return self._line_end()

    def get_length(self):
        return _norm(self.get_end() - self.get_start())

    def get_vector(self):
        return self.get_end() - self.get_start()

    # ── punta (TipableVMobject.position_tip / add_tip de CE) ──
    def _position_tip(self, tip):
        pts = self.get_points()
        handle, anchor = pts[-2], pts[-1]
        angles = _ce_cartesian_to_spherical(handle - anchor)
        tip_angle = _ce_angle_of_vector(_tip_point(tip) - _tip_base(tip))
        tip.rotate(angles[1] - PI - tip_angle)  # noqa: F821
        if not hasattr(self, "_init_positioning_axis"):
            axis = [np.sin(angles[1]), -np.cos(angles[1]), 0]
            tip.rotate(-angles[2] + PI / 2, axis=axis)  # noqa: F821
            self._init_positioning_axis = axis
        tip.shift(anchor - _tip_point(tip))
        return tip

    def _add_tip(self, tip):
        self._position_tip(tip)
        if self._line_length() != 0:
            self._set_line(self._line_start(), _tip_base(tip))
        self.tip = tip
        self.add(tip)

    def _pop_tip(self):
        start, end = self.get_start(), self.get_end()
        tip = self.tip
        self.remove(tip)
        self._set_line(start, end)
        return tip

    def _set_stroke_width_from_length(self):
        width = min(
            self.initial_stroke_width,
            self.max_stroke_width_to_length_ratio * self.get_length(),
        )
        self.set_stroke(width=width, recurse=False)
        return self

    def scale(self, scale_factor, scale_tips: bool = False, about_point=None, about_edge=ORIGIN, **kwargs):  # noqa: F821
        if self.tip is None or self.tip not in self.submobjects:
            return super().scale(scale_factor, about_point=about_point, about_edge=about_edge, **kwargs)
        if self.get_length() == 0:
            return self
        if scale_tips:
            super().scale(scale_factor, about_point=about_point, about_edge=about_edge, **kwargs)
            self._set_stroke_width_from_length()
            return self
        tip = self._pop_tip()
        if about_point is None:
            about_point = self.get_bounding_box_point(about_edge)
        if scale_factor == 0:
            # GL limita el factor a 1e-8; CE colapsa la línea a un punto.
            self.set_points(np.repeat([about_point], len(self.get_points()), axis=0))
        else:
            super().scale(scale_factor, about_point=about_point, **kwargs)
        self._set_stroke_width_from_length()
        self._add_tip(tip)
        return self


@_export
class Vector(Arrow):
    def __init__(self, direction=RIGHT, buff: float = 0, **kwargs):  # noqa: F821
        direction = np.array(direction, dtype=float)
        if len(direction) == 2:
            direction = np.hstack([direction, 0])
        super().__init__(ORIGIN, direction, buff=buff, **kwargs)  # noqa: F821


@_export
class DoubleArrow(Arrow):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DoubleArrow todavía no está portado a manimgl_compat")


@_export
class ArrowVectorField(VGroup):  # noqa: F821
    """`ArrowVectorField` de CE (grilla, largo y color idénticos)."""

    def __init__(
        self,
        func,
        color=None,
        color_scheme=None,
        min_color_scheme_value: float = 0,
        max_color_scheme_value: float = 2,
        colors=DEFAULT_SCALAR_FIELD_COLORS,
        x_range=None,
        y_range=None,
        z_range=None,
        three_dimensions: bool = False,
        length_func=lambda norm: 0.45 * _sigmoid(norm),
        opacity: float = 1.0,
        vector_config: dict | None = None,
        **kwargs,
    ):
        frame_w, frame_h = FRAME_WIDTH, FRAME_HEIGHT  # noqa: F821
        x_range = list(x_range) if x_range else [np.floor(-frame_w / 2), np.ceil(frame_w / 2)]
        y_range = list(y_range) if y_range else [np.floor(-frame_h / 2), np.ceil(frame_h / 2)]
        ranges = [x_range, y_range]
        if three_dimensions or z_range:
            ranges.append(list(z_range) if z_range else list(y_range))
        else:
            ranges.append([0, 0])
        for r in ranges:
            if len(r) == 2:
                r.append(0.5)
            r[1] += r[2]
        super().__init__(**kwargs)
        self.func = func
        self.length_func = length_func
        self.vector_config = dict(vector_config or {})
        if color is None:
            self.single_color = False
            if color_scheme is None:
                color_scheme = np.linalg.norm
            rgbs = np.array([color_to_rgb(c) for c in colors])  # noqa: F821

            def pos_to_rgb(pos):
                vec = self.func(pos)
                value = np.clip(color_scheme(vec), min_color_scheme_value, max_color_scheme_value)
                alpha = (value - min_color_scheme_value) / (max_color_scheme_value - min_color_scheme_value)
                alpha *= len(rgbs) - 1
                c1 = rgbs[int(alpha)]
                c2 = rgbs[min(int(alpha + 1), len(rgbs) - 1)]
                alpha %= 1
                return c1 + (c2 - c1) * alpha

            self.pos_to_color = lambda pos: rgb_to_color(pos_to_rgb(pos))  # noqa: F821
        else:
            self.single_color = True
            self.single_color_value = color
        xs, ys, zs = (np.arange(*r) for r in ranges)
        self.add(*[
            self.get_vector(x * RIGHT + y * UP + z * OUT)  # noqa: F821
            for x, y, z in it.product(xs, ys, zs)
        ])
        self.set_opacity(opacity)

    def get_vector(self, point):
        output = np.array(self.func(point), dtype=float)
        norm = np.linalg.norm(output)
        if norm != 0:
            output *= self.length_func(norm) / norm
        vect = Vector(output, **self.vector_config)
        vect.shift(point)
        vect.set_color(self.single_color_value if self.single_color else self.pos_to_color(point))
        return vect


def _ce_rounded_rectangle_path(width, height, corner_radius) -> VMobject:  # noqa: F821
    """Trazo de RoundedRectangle de CE (mismo punto de inicio y sentido)."""
    vm = VMobject()  # noqa: F821
    w, h = width / 2, height / 2
    verts = [np.array(v, dtype=float) for v in ([w, h, 0], [-w, h, 0], [-w, -h, 0], [w, -h, 0])]
    if corner_radius == 0:
        vm.start_new_path(verts[0])
        for v in verts[1:] + verts[:1]:
            p0 = vm.get_points()[-1]
            _add_cubic_as_quads(vm, (p0, p0 + (v - p0) / 3, p0 + 2 * (v - p0) / 3, v))
        return vm
    arcs = []
    n = len(verts)
    for k in range(n):
        v1, v2, v3 = verts[k], verts[(k + 1) % n], verts[(k + 2) % n]
        vect1, vect2 = v2 - v1, v3 - v2
        u1, u2 = vect1 / _norm(vect1), vect2 / _norm(vect2)
        angle = np.arccos(np.clip(np.dot(u1, u2), -1, 1))
        cut = corner_radius * np.tan(angle / 2)
        sign = np.sign(np.cross(vect1, vect2)[2])
        arcs.append(_ce_arc_cubic(v2 - u1 * cut, v2 + u2 * cut, sign * angle))
    arcs = [arcs[-1], *arcs[:-1]]
    vm.start_new_path(arcs[0][0])
    for k in range(n):
        arc, nxt = arcs[k], arcs[(k + 1) % n]
        _add_cubic_as_quads(vm, tuple(np.asarray(p) for p in arc))
        a, b = arc[3], nxt[0]
        _add_cubic_as_quads(vm, (a, a + (b - a) / 3, a + 2 * (b - a) / 3, b))
    return vm


@_export
class RoundedRectangle(_CESize, VMobject):  # noqa: F821
    def __init__(self, corner_radius: float = 0.5, width: float = 4.0, height: float = 2.0,
                 color=WHITE, **kwargs):  # noqa: F821
        super().__init__(color=color, **kwargs)
        self.set_points(_ce_rounded_rectangle_path(width, height, corner_radius).get_points())
        self.corner_radius = corner_radius


@_export
class SurroundingRectangle(RoundedRectangle):
    def __init__(self, *mobjects, color=YELLOW, buff: float = SMALL_BUFF,  # noqa: F821
                 corner_radius: float = 0.0, **kwargs):
        group = mobjects[0] if len(mobjects) == 1 else Group(*mobjects)  # noqa: F821
        super().__init__(
            corner_radius=corner_radius,
            width=group.get_width() + 2 * buff,
            height=group.get_height() + 2 * buff,
            color=color,
            **kwargs,
        )
        self.buff = buff
        self.move_to(group)


@_export
class BackgroundRectangle(SurroundingRectangle):
    def __init__(self, *mobjects, color=None, stroke_width: float = 0, stroke_opacity: float = 0,
                 fill_opacity: float = 0.75, buff: float = 0, **kwargs):
        if color is None:
            color = BLACK  # noqa: F821
        super().__init__(
            *mobjects, color=color, stroke_width=stroke_width, stroke_opacity=stroke_opacity,
            fill_opacity=fill_opacity, buff=buff, **kwargs,
        )
        self.original_fill_opacity = fill_opacity

    def pointwise_become_partial(self, mobject, a, b):
        self.set_fill(opacity=b * self.original_fill_opacity)
        return self


@_export
class Cross(VGroup):  # noqa: F821
    """`Cross` de CE (dos líneas). Nota: CE ignora `color=` y usa stroke_color."""

    def __init__(self, mobject=None, stroke_color=RED, stroke_width: float = 6.0,  # noqa: F821
                 scale_factor: float = 1.0, **kwargs):
        kwargs.pop("color", None)
        super().__init__(
            Line(UP + LEFT, DOWN + RIGHT),  # noqa: F821
            Line(UP + RIGHT, DOWN + LEFT),  # noqa: F821
            **kwargs,
        )
        if mobject is not None:
            self.replace(mobject, stretch=True)
        self.scale(scale_factor)
        self.set_stroke(color=stroke_color, width=stroke_width)


# ═════════════════════════════════════════════════════════════════════════════
# 3D con la estructura de CE
# ═════════════════════════════════════════════════════════════════════════════
_Z_INDEX_3D = -1  # CE dibuja lo 3D sombreado antes que los objetos 2D
# Iluminación de superficies: CE sólo aclara las caras que miran a la luz
# (rgb + 0.5·(n·l)³); el componente "shadow" de GL oscurece las siluetas.
_SURFACE_SHADING = (0.3, 0.2, 0.0)


class _VMobjectFamilyStubs:
    """Métodos que los VMobject de GL llaman sobre toda su familia.

    Permite que superficies (Arrow3D, relleno de caras) vivan dentro de un
    VGroup, como en CE.
    """

    stroke_behind = False
    needs_new_joint_angles = False

    def get_joint_angles(self, refresh: bool = False):
        return None

    def refresh_joint_angles(self):
        return self

    def get_unit_normal(self, refresh: bool = False):
        return OUT  # noqa: F821

    def refresh_unit_normal(self):
        return self

    def set_fill(self, color=None, opacity=None, border_width=None, recurse=True):
        if color is not None or opacity is not None:
            _GLMobject.set_color(self, color, opacity, recurse)
        return self

    def set_stroke(self, *args, **kwargs):
        return self


class _UVSurface(_VMobjectFamilyStubs, _GLSurface):
    def __init__(self, uv_func, **kwargs):
        self._uv = uv_func
        super().__init__(**kwargs)

    def uv_func(self, u, v):
        return self._uv(u, v)


class _SurfaceFace(VMobject):  # noqa: F821
    """Una cara de Surface de CE.

    El contorno es un trazo (VMobject) y el relleno un parche de Surface de GL.
    El relleno de VMobjects de GL usa triángulos con signo para calcular el
    "winding": en una superficie translúcida las caras de atrás cancelarían a
    las de adelante (CE dibuja cada cara por separado y se suman).
    """

    def __init__(self, patch: _UVSurface, **kwargs):
        super().__init__(fill_opacity=0, **kwargs)
        self.patch = patch
        _GLMobject.add(self, patch)  # VMobject.add sólo acepta VMobjects

    # El relleno vive en el parche (durante __init__ todavía no existe)
    def set_fill(self, color=None, opacity=None, border_width=None, recurse=True):
        if "patch" not in self.__dict__:
            return VMobject.set_fill(self, color, opacity, border_width, recurse=False)  # noqa: F821
        if color is not None or opacity is not None:
            self.patch.set_color(color, opacity)
        return self

    def get_fill_color(self):
        return self.patch.get_color() if "patch" in self.__dict__ else VMobject.get_fill_color(self)  # noqa: F821

    def get_fill_opacity(self):
        return self.patch.get_opacity() if "patch" in self.__dict__ else VMobject.get_fill_opacity(self)  # noqa: F821

    def set_stroke(self, color=None, width=None, opacity=None, behind=None, flat=None, recurse=True):
        return VMobject.set_stroke(self, color, width, opacity, behind, flat, recurse=False)  # noqa: F821

    def set_color(self, color, opacity=None, recurse=True):
        VMobject.set_stroke(self, color=color, opacity=opacity, recurse=False)  # noqa: F821
        if "patch" in self.__dict__:
            self.patch.set_color(color, opacity)
        return self

    def set_opacity(self, opacity, recurse=True):
        VMobject.set_stroke(self, opacity=opacity, recurse=False)  # noqa: F821
        if "patch" in self.__dict__:
            self.patch.set_opacity(opacity)
        return self


def _ce_partial_fill_fraction(s: float) -> float:
    """Fracción del área de una cara cuadrilátera rellena cuando su contorno
    está dibujado hasta la proporción s (Create de CE rellena el trazo parcial)."""
    if s <= 0.25:
        return 0.0
    if s <= 0.5:
        return 0.5 * (s - 0.25) / 0.25
    if s <= 0.75:
        return 0.5 + 0.5 * (s - 0.5) / 0.25
    return 1.0


@_export
class Surface(VGroup):  # noqa: F821
    """`Surface` de CE: una grilla de caras con damero opcional.

    Mismas caras, mismo orden, mismos bordes (cúbicas como en CE) y mismos
    colores. Si la superficie es opaca se activa depth test (CE ordena las
    caras por profundidad); si es translúcida las caras se suman como en CE.
    """

    def __init__(
        self,
        func,
        u_range=(0, 1),
        v_range=(0, 1),
        resolution=32,
        fill_color=BLUE_D,  # noqa: F821
        fill_opacity: float = 1.0,
        checkerboard_colors=(BLUE_D, BLUE_E),  # noqa: F821
        stroke_color=LIGHT_GREY,  # noqa: F821
        stroke_width: float = 0.5,
        stroke_opacity: float = 1.0,
        patch_resolution: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.u_range, self.v_range = u_range, v_range
        res = resolution if isinstance(resolution, (tuple, list)) else (resolution, resolution)
        u_values = np.linspace(*u_range, res[0] + 1)
        v_values = np.linspace(*v_range, res[1] + 1)

        def f(uv):
            return np.array(func(uv[0], uv[1]), dtype=float)

        faces = []
        for i in range(len(u_values) - 1):
            for j in range(len(v_values) - 1):
                u1, u2 = u_values[i:i + 2]
                v1, v2 = v_values[j:j + 2]
                patch = _UVSurface(
                    lambda a, b, u1=u1, u2=u2, v1=v1, v2=v2: f((u1 + a * (u2 - u1), v1 + b * (v2 - v1))),
                    u_range=(0, 1), v_range=(0, 1),
                    resolution=(patch_resolution, patch_resolution),
                    depth_test=False,
                    shading=_SURFACE_SHADING,
                )
                face = _SurfaceFace(patch)
                # CE: cada arista es una cúbica recta en (u, v) con handles en 1/3 y 2/3.
                # VMobject.apply_function de CE acerca los handles a su anchor (×1e-5),
                # aplica func y los vuelve a alejar: los handles quedan sobre la tangente.
                corners = [np.array(c, dtype=float) for c in ([u1, v1], [u2, v1], [u2, v2], [u1, v2], [u1, v1])]
                face.start_new_path(f(corners[0]))
                eps = 1e-5
                for a, b in zip(corners[:-1], corners[1:]):
                    fa, fb = f(a), f(b)
                    h0 = fa + (f(a + eps * (b - a) / 3) - fa) / eps
                    h1 = fb + (f(b + eps * (a - b) / 3) - fb) / eps
                    _add_cubic_as_quads(face, (fa, h0, h1, fb))
                face.u_index, face.v_index = i, j
                faces.append(face)
        self.add(*faces)
        self.set_fill(color=fill_color, opacity=fill_opacity)
        self.set_stroke(color=stroke_color, width=stroke_width, opacity=stroke_opacity)
        if checkerboard_colors:
            self.set_fill_by_checkerboard(*checkerboard_colors)
        self._update_depth_test()
        self.set_z_index(_Z_INDEX_3D)

    def faces(self):
        return [f for f in self.submobjects if isinstance(f, _SurfaceFace)]

    def _update_depth_test(self):
        opaque = all(f.patch.get_opacity() >= 0.999 for f in self.faces())
        for face in self.faces():
            for mob in (face, face.patch):
                if opaque:
                    mob.apply_depth_test(recurse=False)
                else:
                    mob.deactivate_depth_test(recurse=False)
        return self

    def set_fill_by_checkerboard(self, *colors, opacity=None):
        for face in self.faces():
            face.set_fill(colors[(face.u_index + face.v_index) % len(colors)], opacity=opacity)
        return self

    def set_fill(self, *args, **kwargs):
        super().set_fill(*args, **kwargs)
        return self._update_depth_test() if hasattr(self, "u_range") else self

    def set_opacity(self, opacity, recurse=True):
        super().set_opacity(opacity, recurse)
        return self._update_depth_test()


class _CreateSurface(Animation):  # noqa: F821
    """Create de CE sobre Surface: cara por cara (lag_ratio=1), cada una con su
    contorno parcial y el relleno del trazo parcial."""

    def interpolate_mobject(self, alpha: float) -> None:
        faces = self.mobject.faces()
        starts = self.starting_mobject.faces()
        n = len(faces)
        for k, (face, start) in enumerate(zip(faces, starts)):
            s = self.rate_func(float(np.clip(alpha * n - k, 0, 1)))
            face.pointwise_become_partial(start, 0, s)
            face.patch.set_opacity(start.patch.get_opacity() * _ce_partial_fill_fraction(s))


@_export
class Sphere(Surface):
    """`Sphere` de CE (Cairo): 24×12 caras con damero BLUE_D/BLUE_E.

    Ojo: igual que en CE, `color=` NO pinta las caras (CE lo ignora).
    """

    def __init__(self, center=ORIGIN, radius: float = 1, resolution=None,  # noqa: F821
                 u_range=(0, TAU), v_range=(0, PI), **kwargs):  # noqa: F821
        self.radius = radius
        kwargs.setdefault("patch_resolution", 3)
        super().__init__(
            lambda u, v: radius * np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), -np.cos(v)]),
            u_range=u_range, v_range=v_range,
            resolution=resolution if resolution is not None else (24, 12),
            **kwargs,
        )
        self.shift(center)


def _orthonormal_frame(direction):
    d = direction / _norm(direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, helper)
    e1 /= _norm(e1)
    e2 = np.cross(d, e1)
    return d, e1, e2


@_export
class Arrow3D(_VMobjectFamilyStubs, Group):  # noqa: F821
    """`Arrow3D` de CE: cilindro (thickness = radio) + cono (height, base_radius).

    El cilindro de CE es tan fino que en pantalla se ve como una línea cuyo
    ancho es su diámetro más el contorno de 0.5 de cada cara; acá se dibuja
    directamente como un trazo de ese ancho (en GL un cilindro de 0.008 de
    radio queda sub-pixel y se ve más fino que en CE).
    """

    def __init__(self, start=LEFT, end=RIGHT, thickness: float = 0.02, height: float = 0.3,  # noqa: F821
                 base_radius: float = 0.08, color=WHITE, **kwargs):  # noqa: F821
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        d, e1, e2 = _orthonormal_frame(end - start)
        shaft_end = end - height * d
        shaft = Line(start, shaft_end, color=color, stroke_width=2 * thickness * 100 + 0.5)  # noqa: F821
        cone = _UVSurface(
            lambda u, v: end - v * height * d + v * base_radius * (np.cos(u) * e1 + np.sin(u) * e2),
            u_range=(0, TAU), v_range=(0, 1), resolution=(33, 33),  # noqa: F821
            color=color, shading=_SURFACE_SHADING, **kwargs,
        )
        super().__init__(shaft, cone)
        self.shaft, self.cone = shaft, cone
        # En CE el cilindro son 578 de las 1602 piezas que dibuja Create.
        self.create_split = 578 / 1602
        self.set_z_index(_Z_INDEX_3D)


def _ce_number_line_x(length, x_range, stroke_width=2.0, tip_size=DEFAULT_ARROW_TIP_LENGTH,
                      tick_size=0.1, num_pieces=20):
    """Un eje de ThreeDAxes de CE, construido sobre el eje x.

    Estructura (y orden de dibujo en Create) idéntica a CE:
    [línea (ancho 0), punta, 6 ticks, 20 piezas de la línea].
    """
    x_min, x_max, x_step = x_range
    half = length / 2
    axis = VMobject(color=WHITE, stroke_width=stroke_width)  # noqa: F821
    start, end = np.array([-half, 0, 0.0]), np.array([half, 0, 0.0])
    tip = _ce_triangle_tip(tip_size, tip_size, WHITE)  # noqa: F821
    tip.rotate(-PI)  # noqa: F821  (position_tip de CE: la punta mira hacia +x)
    tip.shift(end - _tip_point(tip))
    tip.set_stroke(WHITE, stroke_width)  # noqa: F821
    base = _tip_base(tip)
    axis.set_points_as_corners([start, base])
    ticks = VGroup()  # noqa: F821
    pos = np.arange(x_step, x_max, x_step)
    neg = -np.arange(x_step, abs(x_min) + 1e-6, x_step)
    for x in np.unique(np.concatenate([neg, pos])):
        alpha = (x - x_min) / (x_max - x_min)
        p = start + alpha * (end - start)
        ticks.add(Line(p + tick_size * DOWN, p + tick_size * UP, color=WHITE, stroke_width=stroke_width))  # noqa: F821
    pieces = VGroup()  # noqa: F821
    alphas = np.linspace(0, 1, num_pieces + 1)
    for a, b in zip(alphas[:-1], alphas[1:]):
        piece = VMobject(color=WHITE, stroke_width=stroke_width)  # noqa: F821
        piece.set_points_as_corners([start + a * (base - start), start + b * (base - start)])
        pieces.add(piece)
    axis.add(tip, ticks, pieces)
    axis.tip, axis.ticks, axis.pieces = tip, ticks, pieces
    axis.set_stroke(width=0, recurse=False)
    return axis


@_export
class ThreeDAxes(VGroup):  # noqa: F821
    """`ThreeDAxes` de CE (sin números), con ticks, puntas y piezas 3D."""

    def __init__(self, x_range=(-6, 6, 1), y_range=(-5, 5, 1), z_range=(-4, 4, 1),
                 x_length=10.5, y_length=10.5, z_length=6.5, **kwargs):
        super().__init__(**kwargs)
        self.x_axis = _ce_number_line_x(x_length, x_range)
        self.y_axis = _ce_number_line_x(y_length, y_range).rotate(PI / 2)  # noqa: F821
        z_axis = _ce_number_line_x(z_length, z_range)
        z_axis.rotate(-PI / 2, axis=UP, about_point=ORIGIN)  # noqa: F821
        z_axis.rotate(-PI / 2, axis=OUT, about_point=ORIGIN)  # noqa: F821
        self.z_axis = z_axis
        self.add(self.x_axis, self.y_axis, self.z_axis)
        self.set_z_index(_Z_INDEX_3D)


# ═════════════════════════════════════════════════════════════════════════════
# Animaciones con el comportamiento de CE
# ═════════════════════════════════════════════════════════════════════════════
def _as_single(mobjects):
    if len(mobjects) == 1:
        return mobjects[0]
    if all(isinstance(m, VMobject) for m in mobjects):  # noqa: F821
        return VGroup(*mobjects)  # noqa: F821
    return Group(*mobjects)  # noqa: F821


@_export
class FadeIn(_GLFadeIn):
    """FadeIn de CE: `scale=s` arranca con el objeto ×s (GL lo invierte)."""

    def __init__(self, *mobjects, shift=None, target_position=None, scale: float = 1, **kwargs):
        mobject = _as_single(mobjects)
        if shift is None:
            shift = ORIGIN  # noqa: F821
            if target_position is not None:
                # CE: con target_position el desplazamiento no se invierte
                if isinstance(target_position, Mobject):  # noqa: F821
                    target_position = target_position.get_center()
                shift = -(np.asarray(target_position) - mobject.get_center())
        super().__init__(mobject, shift=shift, scale=1.0 / scale, **kwargs)


@_export
class FadeOut(_GLFadeOut):
    def __init__(self, *mobjects, shift=None, target_position=None, scale: float = 1, **kwargs):
        mobject = _as_single(mobjects)
        if shift is None:
            shift = ORIGIN  # noqa: F821
            if target_position is not None:
                if isinstance(target_position, Mobject):  # noqa: F821
                    target_position = target_position.get_center()
                shift = np.asarray(target_position) - mobject.get_center()
        super().__init__(mobject, shift=shift, scale=scale, **kwargs)


@_export
class Write(_GLWrite):
    """Write de CE: misma duración/lag_ratio y color de contorno por pieza."""

    def __init__(self, vmobject, rate_func=linear, run_time=None, lag_ratio=None,  # noqa: F821
                 stroke_color=None, **kwargs):
        length = len(vmobject.family_members_with_points())
        if run_time is None:
            run_time = 1 if length < 15 else 2
        if lag_ratio is None:
            lag_ratio = min(4.0 / max(1.0, length), 0.2)
        self._ce_stroke_color = stroke_color
        _GLDrawBorderThenFill.__init__(
            self, vmobject, run_time=run_time, lag_ratio=lag_ratio,
            rate_func=rate_func, stroke_color=stroke_color, **kwargs,
        )

    def get_outline(self):
        outline = self.mobject.copy()
        outline.set_fill(opacity=0)
        for sm in outline.family_members_with_points():
            if self._ce_stroke_color is not None:
                color = self._ce_stroke_color
            elif sm.get_stroke_width() > 0:
                color = sm.get_stroke_color()
            else:
                color = sm.get_fill_color()
            sm.set_stroke(color=color, width=self.stroke_width, behind=self.mobject.stroke_behind)
        return outline


@_export
class GrowArrow(GrowFromPoint):  # noqa: F821
    """GrowArrow de CE: la flecha entera (punta incluida) crece desde su inicio."""

    def __init__(self, arrow, point_color=None, **kwargs):
        super().__init__(arrow, arrow.get_start(), point_color=point_color, **kwargs)

    def create_starting_mobject(self):
        start = self.mobject.copy()
        if isinstance(start, Arrow):
            start.scale(0, scale_tips=True, about_point=self.point)
        else:
            start.scale(0, about_point=self.point)
        if self.point_color:
            start.set_color(self.point_color)
        return start


class _CreateArrow3D(Animation):  # noqa: F821
    """Create(Arrow3D) de CE: el cilindro crece y después el cono, en proporción
    a la cantidad de caras que tiene cada uno en CE (avance lineal)."""

    def __init__(self, arrow: Arrow3D, **kwargs):
        super().__init__(arrow, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        split = self.mobject.create_split
        a1 = float(np.clip(alpha / split, 0, 1))
        a2 = float(np.clip((alpha - split) / (1 - split), 0, 1))
        self.mobject.shaft.pointwise_become_partial(self.starting_mobject.shaft, 0, a1)
        self.mobject.cone.pointwise_become_partial(self.starting_mobject.cone, 0, a2)


@_export
def Create(mobject, lag_ratio: float = 1.0, **kwargs):
    if isinstance(mobject, Arrow3D):
        return _CreateArrow3D(mobject, **kwargs)
    if isinstance(mobject, Surface):
        return _CreateSurface(mobject, **kwargs)
    return ShowCreation(mobject, lag_ratio=lag_ratio, **kwargs)  # noqa: F821


@_export
class TransformMatchingTex(AnimationGroup):  # noqa: F821
    """TransformMatchingTex de CE: empareja partes por su tex_string."""

    def __init__(self, mobject, target_mobject, transform_mismatches: bool = False,
                 fade_transform_mismatches: bool = False, key_map: dict | None = None, **kwargs):
        source_map = self._shape_map(mobject)
        target_map = self._shape_map(target_mobject)
        key_map = key_map or {}
        kwargs["final_alpha_value"] = 0
        transform_source, transform_target = VGroup(), VGroup()  # noqa: F821
        for key in set(source_map).intersection(target_map):
            transform_source.add(source_map[key])
            transform_target.add(target_map[key])
        anims = [Transform(transform_source, transform_target, **kwargs)]  # noqa: F821
        km_source, km_target = VGroup(), VGroup()  # noqa: F821
        for k1, k2 in key_map.items():
            if k1 in source_map and k2 in target_map:
                km_source.add(source_map.pop(k1))
                km_target.add(target_map.pop(k2))
        if len(km_source) > 0:
            anims.append(FadeTransformPieces(km_source, km_target, **kwargs))  # noqa: F821
        fade_source, fade_target = VGroup(), VGroup()  # noqa: F821
        for key in set(source_map).difference(target_map):
            fade_source.add(source_map[key])
        for key in set(target_map).difference(source_map):
            fade_target.add(target_map[key])
        fade_target_copy = fade_target.copy()
        if transform_mismatches:
            anims.append(Transform(fade_source, fade_target, **kwargs))  # noqa: F821
        elif fade_transform_mismatches:
            anims.append(FadeTransformPieces(fade_source, fade_target, **kwargs))  # noqa: F821
        else:
            anims.append(FadeOut(fade_source, target_position=fade_target, **kwargs))
            anims.append(FadeIn(fade_target_copy, target_position=fade_target, **kwargs))
        super().__init__(*anims)
        self.to_remove = [mobject, fade_target_copy]
        self.to_add = target_mobject

    @staticmethod
    def _parts(mobject):
        if isinstance(mobject, MathTex):
            return list(mobject.submobjects)
        return [p for s in mobject.submobjects for p in TransformMatchingTex._parts(s)]

    def _shape_map(self, mobject):
        shape_map = {}
        for part in self._parts(mobject):
            shape_map.setdefault(part.tex_string, VGroup()).add(part)  # noqa: F821
        return shape_map

    def clean_up_from_scene(self, scene):
        for anim in self.animations:
            anim.interpolate(0)
        scene.remove(self.mobject)
        scene.remove(*self.to_remove)
        scene.add(self.to_add)


# ═════════════════════════════════════════════════════════════════════════════
# Escenas
# ═════════════════════════════════════════════════════════════════════════════
@_export
class Scene(_GLScene):
    """Scene con la semántica de CE para agregar objetos al animarlos."""

    def set_background_color(self, color) -> None:
        self.camera.background_rgba = list(color_to_rgba(color, 1.0))  # noqa: F821

    def scene_mobjects(self) -> list:
        """Equivalente a `self.mobjects` de CE (sin el CameraFrame de GL)."""
        return [m for m in self.mobjects if m is not self.frame]

    def _leaf_animations(self, animation):
        if isinstance(animation, AnimationGroup) and not isinstance(animation, TransformMatchingTex):  # noqa: F821
            for sub in animation.animations:
                yield from self._leaf_animations(sub)
        else:
            yield animation

    def begin_animations(self, animations) -> None:
        # GL agrega el `group` de un AnimationGroup como un solo objeto y vuelve
        # a agregar grupos cuyos hijos ya están en escena (los dibuja dos veces).
        # CE agrega cada objeto animado por separado y nunca duplica.
        for animation in animations:
            animation.begin()
        for animation in animations:
            for leaf in self._leaf_animations(animation):
                in_scene = set(self.get_mobject_family_members())
                mob = leaf.mobject
                members = mob.family_members_with_points() or [mob]
                if not all(m in in_scene for m in members):
                    self.add(mob)


@_export
class ThreeDScene(Scene):
    """ThreeDScene de CE: phi/theta en la convención de CE y focal_distance=20."""

    def setup(self):
        super().setup()
        self._ce_phi, self._ce_theta, self._ce_gamma = 0.0, -PI / 2, 0.0  # noqa: F821
        self._ambient_updater = None
        self.frame.set_focal_distance(20.0)
        self._apply_camera()

    def _apply_camera(self):
        # CE: R = Rz(γ)·Rx(-φ)·Rz(-θ-90°)   GL: view = Rz(-γ')·Rx(-φ')·Rz(-θ')
        self.frame.set_euler_angles(
            theta=self._ce_theta + PI / 2,  # noqa: F821
            phi=self._ce_phi,
            gamma=-self._ce_gamma,
        )

    def set_camera_orientation(self, phi=None, theta=None, gamma=None, **kwargs):
        if phi is not None:
            self._ce_phi = phi
        if theta is not None:
            self._ce_theta = theta
        if gamma is not None:
            self._ce_gamma = gamma
        self._apply_camera()

    def move_camera(self, phi=None, theta=None, gamma=None, added_anims=(), **kwargs):
        start = np.array([self._ce_phi, self._ce_theta, self._ce_gamma])
        end = np.array([
            start[0] if phi is None else phi,
            start[1] if theta is None else theta,
            start[2] if gamma is None else gamma,
        ])
        rate_func = kwargs.pop("rate_func", smooth)

        def update(frame, alpha):
            self._ce_phi, self._ce_theta, self._ce_gamma = start + (end - start) * rate_func(alpha)
            self._apply_camera()

        self.play(UpdateFromAlphaFunc(self.frame, update), *added_anims, **kwargs)  # noqa: F821

    def begin_ambient_camera_rotation(self, rate: float = 0.02, about: str = "theta"):
        attr = {"theta": "_ce_theta", "phi": "_ce_phi", "gamma": "_ce_gamma"}[about.lower()]

        def updater(frame, dt):
            setattr(self, attr, getattr(self, attr) + rate * dt)
            self._apply_camera()

        self.stop_ambient_camera_rotation()
        self._ambient_updater = updater
        self.frame.add_updater(updater)

    def stop_ambient_camera_rotation(self, about: str = "theta"):
        if self._ambient_updater is not None:
            self.frame.remove_updater(self._ambient_updater)
            self._ambient_updater = None

    def add_fixed_in_frame_mobjects(self, *mobjects):
        for mob in mobjects:
            mob.fix_in_frame()
        self.add(*mobjects)
