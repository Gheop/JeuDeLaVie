"""Jeu de la Vie GPU — simulation et rendu via shaders ModernGL.

Contrôles
    Souris gauche / droit : dessiner / effacer (tracé continu)
    Molette               : zoom centré sur le curseur
    Clic milieu (drag)    : déplacer la vue
    1-9                   : sélectionne un motif (mode placement)
      └─ en mode placement :
           Q / E           → rotation 90° gauche / droite
           clic gauche     → pose le motif
           clic droit / Esc → annule
    T                     : cycle les règles (Conway, HighLife, Day & Night…)
    Espace                : pause / play
    R                     : remplissage aléatoire
    C                     : effacer la grille
    Z                     : recentrer (zoom 1, vue plein cadre)
    +/-                   : vitesse de simulation
    F                     : plein écran
    F5 / F9               : save / load PNG (snapshots/)
    H                     : afficher/masquer l'aide
    Échap                 : quitter (ou annule le mode placement)
"""

import math
import os
import sys
import time
import glob
import numpy as np
import pygame
import moderngl


WIN_W, WIN_H = 1280, 800
# La grille est allouée à la résolution de l'écran au démarrage et double
# (en place, avec recopie centrée) à chaque fois que la vue dézoomée/pannée
# menace de sortir de ses bords. Ça évite de payer une grille 42 Mpixel
# quand l'utilisateur n'en utilise qu'un coin, tout en conservant l'illusion
# d'un monde qui s'agrandit quand on recule la caméra.
MAX_GRID_SIZE = 16384              # borne haute (limite texture GPU usuelle)
INITIAL_TPS = 30                    # ticks de simulation par seconde


VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# Canal R : état (0/1). Canal G : âge normalisé qui décroît quand la cellule
# meurt (sert à la traînée). La texture est en RG8 (unorm) — GLSL lit les
# échantillons comme des floats 0..1, on divise la bande passante par 8 vs
# l'ancien RGBA32F sans modifier les shaders.
# Les voisins hors grille sont considérés morts (monde fini, pas de wrap
# toroïdal) : un vaisseau qui sort disparaît au lieu de revenir par l'autre
# côté.
# u_birth / u_survive : bitmasks (bit i = comportement avec i voisines).
SIM_SHADER = """
#version 330
uniform sampler2D u_state;
uniform int u_birth;
uniform int u_survive;
in vec2 v_uv;
out vec4 frag;

void main() {
    vec2 px = 1.0 / vec2(textureSize(u_state, 0));
    float c = texture(u_state, v_uv).r;
    int n = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            vec2 nb = v_uv + vec2(dx, dy) * px;
            if (nb.x < 0.0 || nb.x > 1.0 || nb.y < 0.0 || nb.y > 1.0) continue;
            n += int(texture(u_state, nb).r > 0.5);
        }
    }
    int mask = (c > 0.5) ? u_survive : u_birth;
    float alive = float((mask >> n) & 1);

    float age = texture(u_state, v_uv).g;
    age = (alive > 0.5) ? min(age + 0.04, 1.0) : age * 0.90;

    frag = vec4(alive, age, 0.0, 1.0);
}
"""

# Peinture à la souris : on écrit directement dans la texture d'état courante
# (pas de ping-pong) en s'appuyant sur un scissor pour limiter la rastérisation
# au carré englobant du pinceau. Les fragments hors du cercle font `discard`,
# ce qui préserve les pixels existants du framebuffer. Plus besoin de lire la
# texture d'entrée : on n'écrit qu'à l'intérieur du cercle.
PAINT_SHADER = """
#version 330
uniform vec2  u_center;     // en coordonnées UV
uniform float u_radius;     // en UV (axe Y)
uniform float u_value;      // 1.0 = dessine, 0.0 = efface
uniform float u_aspect;     // width/height, pour un cercle rond en espace cellule
in vec2 v_uv;
out vec4 frag;
void main() {
    vec2 d = v_uv - u_center;
    d.x *= u_aspect;
    if (length(d) >= u_radius) discard;
    frag = vec4(u_value, u_value, 0.0, 1.0);
}
"""

# Stamp d'un motif sur la grille, entièrement GPU (évite un aller-retour
# lecture/écriture numpy). Le motif est uploadé dans u_pattern ; on sample
# l'état d'entrée puis on écrase là où le motif est à 1. Pas de wrap : un
# motif posé près d'un bord est clippé (cohérent avec la sim finie).
STAMP_SHADER = """
#version 330
uniform sampler2D u_state;
uniform sampler2D u_pattern;
uniform vec2 u_cursor_tex;     // centre du motif en UV de la texture d'état
uniform vec2 u_pattern_size;   // taille du motif en UV de la texture d'état
in vec2 v_uv;
out vec4 frag;

void main() {
    vec4 cur = texture(u_state, v_uv);
    vec2 d = v_uv - u_cursor_tex;
    vec2 p = d / u_pattern_size + 0.5;
    if (p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0) {
        float a = texture(u_pattern, p).r;
        if (a > 0.5) {
            cur.r = 1.0;
            cur.g = max(cur.g, 1.0);
        }
    }
    frag = cur;
}
"""

# Copie state.r dans une texture R32F pour la réduction mipmap
# (moyenne → fraction de vivantes → count). Passe en float pour éviter la
# perte de précision des formats unorm sur les petits comptages.
# Pour éviter qu'une grille 16 k² se traduise par 1 Go de reduce_tex, on la
# plafonne (REDUCE_MAX) et ce shader fait un box filter u_scale×u_scale
# par pixel de sortie — branches spécialisées pour scale∈{1,2,4} afin que
# GLSL puisse dérouler proprement.
REDUCE_SHADER = """
#version 330
uniform sampler2D u_state;
uniform int u_scale;
in vec2 v_uv;
out vec4 frag;
void main() {
    if (u_scale == 1) {
        frag = vec4(texture(u_state, v_uv).r, 0.0, 0.0, 1.0);
        return;
    }
    vec2 px = 1.0 / vec2(textureSize(u_state, 0));
    float acc = 0.0;
    if (u_scale == 2) {
        vec2 base = v_uv - 0.5 * px;
        acc = texture(u_state, base).r
            + texture(u_state, base + vec2(px.x, 0.0)).r
            + texture(u_state, base + vec2(0.0, px.y)).r
            + texture(u_state, base + px).r;
        frag = vec4(acc * 0.25, 0.0, 0.0, 1.0);
    } else {
        // u_scale == 4 : 16 taps sur 4×4 pixels de state par pixel de sortie.
        vec2 base = v_uv - 1.5 * px;
        for (int dy = 0; dy < 4; dy++) {
            for (int dx = 0; dx < 4; dx++) {
                acc += texture(u_state, base + vec2(float(dx), float(dy)) * px).r;
            }
        }
        frag = vec4(acc / 16.0, 0.0, 0.0, 1.0);
    }
}
"""

# Flou gaussien : passe horizontale, 7 taps. Rend à la résolution écran dans
# une texture R8 intermédiaire ; la passe verticale est fusionnée dans le
# display shader. Au total 7+7=14 taps par pixel écran au lieu de 7×7=49.
GLOW_H_SHADER = """
#version 330
uniform sampler2D u_state;
uniform vec2  u_center;
uniform float u_zoom;
in vec2 v_uv;
out vec4 frag;

void main() {
    vec2 uv = (v_uv - 0.5) * u_zoom + u_center;
    float pxx = u_zoom / float(textureSize(u_state, 0).x);
    float acc = 0.0, wsum = 0.0;
    for (int dx = -3; dx <= 3; dx++) {
        float w = exp(-float(dx*dx) / 6.0);
        vec2 nb = uv + vec2(float(dx) * pxx, 0.0);
        // Hors grille = 0 pour éviter que le glow ne tile avec la texture.
        float g = (nb.x < 0.0 || nb.x > 1.0 || nb.y < 0.0 || nb.y > 1.0)
                  ? 0.0 : texture(u_state, nb).g;
        acc  += g * w;
        wsum += w;
    }
    frag = vec4(acc / wsum, 0.0, 0.0, 1.0);
}
"""

# Rendu final : palette cyan→magenta selon l'âge, lueur additive (passe V du
# flou séparable, lit la passe H précalculée), fond sombre et vignettage.
DISPLAY_SHADER = """
#version 330
uniform sampler2D u_state;
uniform sampler2D u_glow_h;
uniform float u_time;
uniform vec2  u_center;   // UV du centre de la vue
uniform float u_zoom;     // largeur (en UV) visible à l'écran
in vec2 v_uv;
out vec4 frag;

vec3 palette(float t) {
    vec3 a = vec3(0.18, 0.85, 0.95);
    vec3 b = vec3(0.95, 0.30, 0.85);
    return mix(a, b, smoothstep(0.0, 1.0, t));
}

void main() {
    vec2 uv = (v_uv - 0.5) * u_zoom + u_center;
    bool in_grid = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
    vec4 s  = in_grid ? texture(u_state, uv) : vec4(0.0);

    float pxy = u_zoom / float(textureSize(u_state, 0).y);
    float glow = 0.0, wsum = 0.0;
    for (int dy = -3; dy <= 3; dy++) {
        float w = exp(-float(dy*dy) / 6.0);
        glow += texture(u_glow_h, v_uv + vec2(0.0, float(dy) * pxy)).r * w;
        wsum += w;
    }
    glow /= wsum;

    float age   = s.g;
    float alive = s.r;

    vec3 cell = palette(age) * (0.55 + 0.45 * age);
    vec3 halo = palette(0.65) * glow * 1.4;

    // Fond de base + léger assombrissement hors de la grille pour marquer la limite.
    vec3 bg_in  = mix(vec3(0.02, 0.025, 0.05), vec3(0.05, 0.04, 0.10), v_uv.y);
    vec3 bg_out = bg_in * 0.55;
    vec3 bg = in_grid ? bg_in : bg_out;
    vec3 col = bg + halo + cell * alive;

    vec2 q = v_uv - 0.5;
    col *= 1.0 - dot(q, q) * 0.6;

    float n = fract(sin(dot(v_uv * vec2(1280.0, 800.0) + u_time, vec2(12.9898, 78.233))) * 43758.5453);
    col += (n - 0.5) * 0.015;

    frag = vec4(col, 1.0);
}
"""


# Aperçu d'un motif sous le curseur, en surimpression du rendu courant.
PREVIEW_SHADER = """
#version 330
uniform sampler2D u_pattern;
uniform vec2  u_cursor_tex;     // UV (texture) du centre du motif
uniform vec2  u_pattern_size;   // taille du motif en UV de texture
uniform vec2  u_view_center;
uniform float u_view_zoom;
uniform float u_time;
in vec2 v_uv;
out vec4 frag;

void main() {
    vec2 tex_uv = (v_uv - 0.5) * u_view_zoom + u_view_center;
    vec2 p = (tex_uv - u_cursor_tex) / u_pattern_size + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) discard;
    float a = texture(u_pattern, p).r;
    if (a < 0.5) discard;
    float pulse = 0.55 + 0.30 * sin(u_time * 4.0);
    frag = vec4(1.0, 0.85, 0.35, pulse);
}
"""

# Agrandissement de la grille : on alloue une texture 2× plus grande et on
# recopie l'ancien contenu au centre (bande 0.25..0.75 de la nouvelle texture).
# Tout ce qui déborde (les anciens bords) reste à 0, donc du monde "vide".
GROW_SHADER = """
#version 330
uniform sampler2D u_src;
in vec2 v_uv;
out vec4 frag;
void main() {
    // v_uv couvre la nouvelle texture ; on remappe [0.25, 0.75] -> [0, 1].
    vec2 uv = (v_uv - 0.25) * 2.0;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        frag = vec4(0.0);
    } else {
        frag = texture(u_src, uv);
    }
}
"""

# Rétrécissement : ne garde que le quart central de l'ancienne texture et
# l'étire sur la totalité de la nouvelle (2× plus petite). On ne l'appelle
# que si on a vérifié qu'il n'y a aucune cellule vivante en dehors de cette
# zone, donc pas de perte d'information.
SHRINK_SHADER = """
#version 330
uniform sampler2D u_src;
in vec2 v_uv;
out vec4 frag;
void main() {
    // v_uv couvre la nouvelle texture (plus petite) ; on échantillonne le
    // centre [0.25, 0.75] de l'ancienne.
    vec2 uv = v_uv * 0.5 + 0.25;
    frag = texture(u_src, uv);
}
"""

# Surcouche HUD : un quad (x, y, w, h) en pixels écran texturé d'une surface
# pygame contenant le texte rendu côté CPU.
HUD_SHADER = """
#version 330
uniform sampler2D u_hud;
uniform vec2 u_pos_px;
uniform vec2 u_size_px;
uniform vec2 u_screen_px;
in vec2 v_uv;
out vec4 frag;

void main() {
    vec2 px = vec2(v_uv.x, 1.0 - v_uv.y) * u_screen_px;
    vec2 local = (px - u_pos_px) / u_size_px;
    if (local.x < 0.0 || local.x > 1.0 || local.y < 0.0 || local.y > 1.0) discard;
    frag = texture(u_hud, vec2(local.x, 1.0 - local.y));
}
"""


# Règles : naissance et survie encodées en bitmasks (bit i = i voisines).
def rule(birth, survive):
    return (sum(1 << i for i in birth), sum(1 << i for i in survive))

RULES = [
    ("Conway B3/S23",            *rule([3],          [2, 3])),
    ("HighLife B36/S23",         *rule([3, 6],       [2, 3])),
    ("Day & Night B3678/S34678", *rule([3, 6, 7, 8], [3, 4, 6, 7, 8])),
    ("Seeds B2/S",               *rule([2],          [])),
    ("Replicator B1357/S1357",   *rule([1, 3, 5, 7], [1, 3, 5, 7])),
    ("2x2  B36/S125",            *rule([3, 6],       [1, 2, 5])),
]


# Bibliothèque de motifs classiques. 'O' = cellule vivante, '.' = morte.
PATTERNS_RAW = {
    pygame.K_1: ("Glider", """
        .O.
        ..O
        OOO
    """),
    pygame.K_2: ("LWSS — Lightweight spaceship", """
        .OOOO
        O...O
        ....O
        O..O.
    """),
    pygame.K_3: ("MWSS — Middleweight spaceship", """
        ...O..
        .O...O
        O.....
        O....O
        OOOOO.
    """),
    pygame.K_4: ("HWSS — Heavyweight spaceship", """
        ...OO..
        .O....O
        O......
        O.....O
        OOOOOO.
    """),
    pygame.K_5: ("Pulsar (oscillateur p3)", """
        ..OOO...OOO..
        .............
        O....O.O....O
        O....O.O....O
        O....O.O....O
        ..OOO...OOO..
        .............
        ..OOO...OOO..
        O....O.O....O
        O....O.O....O
        O....O.O....O
        .............
        ..OOO...OOO..
    """),
    pygame.K_6: ("Pentadécathlon (rangée de 10)", """
        OOOOOOOOOO
    """),
    pygame.K_7: ("Canon de Gosper", """
        ........................O...........
        ......................O.O...........
        ............OO......OO............OO
        ...........O...O....OO............OO
        OO........O.....O...OO..............
        OO........O...O.OO....O.O...........
        ..........O.....O.......O...........
        ...........O...O....................
        ............OO......................
    """),
    pygame.K_8: ("R-pentomino (méthuselah)", """
        .OO
        OO.
        .O.
    """),
    pygame.K_9: ("Acorn (méthuselah)", """
        .O.....
        ...O...
        OO..OOO
    """),
}


def parse_pattern(s):
    rows = [r.strip() for r in s.strip("\n").splitlines() if r.strip()]
    h = len(rows)
    w = max(len(r) for r in rows)
    arr = np.zeros((h, w), dtype=np.float32)
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch in "O#1":
                arr[r, c] = 1.0
    return arr


PATTERNS = {k: (name, parse_pattern(s)) for k, (name, s) in PATTERNS_RAW.items()}


def make_program(ctx, fragment):
    return ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=fragment)


class Chunk:
    """Rectangle de monde autonome : textures ping-pong pour la sim et une
    texture de réduction pour compter les vivantes (et détecter la bbox active).

    La reduce_tex est plafonnée à REDUCE_MAX² sinon une grille 16 k² en R32F
    pèserait ~1 Go. Le reduce shader compense en faisant un box filter
    u_scale×u_scale pour couvrir tous les pixels de state."""

    REDUCE_MAX = 4096

    def __init__(self, ctx, width, height, initial=None):
        self.width = width
        self.height = height
        # Ping-pong RG8 : R = alive, G = age normalisé.
        self.tex_a = self._make_state_tex(ctx, initial)
        self.tex_b = self._make_state_tex(ctx, None)
        self.fbo_a = ctx.framebuffer(color_attachments=[self.tex_a])
        self.fbo_b = ctx.framebuffer(color_attachments=[self.tex_b])
        self.front_tex = self.tex_a
        self.front_fbo = self.fbo_a
        self.back_tex  = self.tex_b
        self.back_fbo  = self.fbo_b
        # Facteur d'échelle entier tel que reduce_w,h restent ≤ REDUCE_MAX.
        # Pour les tailles courantes du projet (grilles ≤ MAX_GRID_SIZE=16384),
        # scale ∈ {1, 2, 4}. Le reduce shader a des branches spécialisées
        # pour ces trois valeurs pour que GLSL déroule les taps.
        max_dim = max(width, height)
        self.reduce_scale = 1
        while max_dim // self.reduce_scale > self.REDUCE_MAX:
            self.reduce_scale *= 2
        self.reduce_w = max(1, width  // self.reduce_scale)
        self.reduce_h = max(1, height // self.reduce_scale)
        self.reduce_tex = ctx.texture((self.reduce_w, self.reduce_h), 1, dtype="f4")
        self.reduce_tex.filter = (moderngl.LINEAR_MIPMAP_NEAREST, moderngl.LINEAR)
        self.reduce_fbo = ctx.framebuffer(color_attachments=[self.reduce_tex])
        self.reduce_max_level = int(math.floor(math.log2(max(self.reduce_w,
                                                             self.reduce_h))))

    def _make_state_tex(self, ctx, initial):
        t = ctx.texture((self.width, self.height), 2, dtype="f1")
        t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        # Bords absorbants : la sim compte les voisins hors grille comme morts.
        t.repeat_x = t.repeat_y = False
        if initial is not None:
            t.write(initial.tobytes())
        return t

    def swap(self):
        self.front_tex, self.back_tex = self.back_tex, self.front_tex
        self.front_fbo, self.back_fbo = self.back_fbo, self.front_fbo

    def release(self):
        """Libère toutes les ressources GL détenues. Appelé sur les vieux
        chunks après grow/shrink, et sur l'unique chunk restant au shutdown."""
        for obj in (self.fbo_a, self.fbo_b, self.tex_a, self.tex_b,
                    self.reduce_fbo, self.reduce_tex):
            obj.release()

    def clear(self):
        zeros = np.zeros((self.height, self.width, 2), dtype=np.uint8)
        self.front_tex.write(zeros.tobytes())

    def randomize(self, density=0.25):
        self.front_tex.write(random_state(self.width, self.height, density).tobytes())


def random_state(w, h, density=0.25):
    """État aléatoire en RG8 : shape (h, w, 2), uint8. R = alive, G = age."""
    mask = (np.random.random((h, w)) < density).astype(np.uint8) * 255
    arr = np.zeros((h, w, 2), dtype=np.uint8)
    arr[..., 0] = mask
    arr[..., 1] = mask
    return arr


def main():
    pygame.init()
    pygame.display.set_caption("Jeu de la Vie — GPU")
    windowed_flags   = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    fullscreen_flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                    pygame.GL_CONTEXT_PROFILE_CORE)
    # Démarre en plein écran par défaut (F pour basculer en fenêtré).
    # `--windowed` permet de forcer la fenêtre à la taille WIN_W×WIN_H.
    start_windowed = "--windowed" in sys.argv
    fullscreen = not start_windowed
    if fullscreen:
        pygame.display.set_mode((0, 0), fullscreen_flags)
    else:
        pygame.display.set_mode((WIN_W, WIN_H), windowed_flags)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    # Défini une fois : le seul blend func utilisé est le alpha over classique
    # (pour le HUD et la preview du motif). La sim et le display opaque sont
    # en write-only, le blend actif ne les gêne pas.
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    # quad plein écran
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())

    sim_prog     = make_program(ctx, SIM_SHADER)
    paint_prog   = make_program(ctx, PAINT_SHADER)
    stamp_prog   = make_program(ctx, STAMP_SHADER)
    reduce_prog  = make_program(ctx, REDUCE_SHADER)
    glow_h_prog  = make_program(ctx, GLOW_H_SHADER)
    display_prog = make_program(ctx, DISPLAY_SHADER)
    preview_prog = make_program(ctx, PREVIEW_SHADER)
    grow_prog    = make_program(ctx, GROW_SHADER)
    shrink_prog  = make_program(ctx, SHRINK_SHADER)
    hud_prog     = make_program(ctx, HUD_SHADER)

    sim_vao     = ctx.simple_vertex_array(sim_prog,     quad, "in_pos")
    paint_vao   = ctx.simple_vertex_array(paint_prog,   quad, "in_pos")
    stamp_vao   = ctx.simple_vertex_array(stamp_prog,   quad, "in_pos")
    reduce_vao  = ctx.simple_vertex_array(reduce_prog,  quad, "in_pos")
    glow_h_vao  = ctx.simple_vertex_array(glow_h_prog,  quad, "in_pos")
    display_vao = ctx.simple_vertex_array(display_prog, quad, "in_pos")
    preview_vao = ctx.simple_vertex_array(preview_prog, quad, "in_pos")
    grow_vao    = ctx.simple_vertex_array(grow_prog,    quad, "in_pos")
    shrink_vao  = ctx.simple_vertex_array(shrink_prog,  quad, "in_pos")
    hud_vao     = ctx.simple_vertex_array(hud_prog,     quad, "in_pos")

    # Taille initiale de la grille = résolution du bureau, plafonnée à
    # MAX_GRID_SIZE. La grille doublera au besoin (voir grow_chunk plus bas).
    info = pygame.display.Info()
    init_w = max(WIN_W, min(info.current_w, MAX_GRID_SIZE))
    init_h = max(WIN_H, min(info.current_h, MAX_GRID_SIZE))
    chunk = Chunk(ctx, init_w, init_h, initial=random_state(init_w, init_h))

    # Passe H du glow : texture R8 à la résolution écran, recréée à la volée
    # quand la taille change.
    glow = {"tex": None, "fbo": None, "size": (0, 0)}

    def ensure_glow_tex(w, h):
        if glow["size"] == (w, h):
            return
        if glow["tex"] is not None:
            glow["tex"].release()
            glow["fbo"].release()
        tex = ctx.texture((w, h), 1, dtype="f1")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        fbo = ctx.framebuffer(color_attachments=[tex])
        glow["tex"] = tex
        glow["fbo"] = fbo
        glow["size"] = (w, h)

    sim = {"rule_idx": 0}  # dict pour pouvoir muter depuis les closures

    # Rectangle englobant les cellules vivantes — recalculé depuis la mipmap
    # de reduce_tex toutes les 500 ms (cf. update_alive_count). Le sim est
    # restreint à cette boîte + un bloc de marge, ce qui évite de simuler
    # des dizaines de millions de cellules mortes après un grow.
    sim_bbox = {"valid": False, "x0": 0, "y0": 0, "x1": 0, "y1": 0}

    def step():
        # Rien à simuler si tout est mort : Conway ne crée pas de vie à partir
        # du vide, donc on peut sauter le passe complet.
        if counters["alive"] == 0:
            return
        chunk.back_fbo.use()
        ctx.viewport = (0, 0, chunk.width, chunk.height)
        # Si on sait où est la vie, on scissor la sim et on efface d'abord
        # tout le back_fbo pour que les pixels hors bbox restent à 0 (sinon
        # du vieux contenu du ping-pong ressurgirait au prochain swap).
        use_scissor = False
        if sim_bbox["valid"]:
            bw = sim_bbox["x1"] - sim_bbox["x0"]
            bh = sim_bbox["y1"] - sim_bbox["y0"]
            if bw > 0 and bh > 0 and bw * bh < chunk.width * chunk.height * 0.85:
                use_scissor = True
        if use_scissor:
            ctx.scissor = None
            chunk.back_fbo.clear()
            ctx.scissor = (sim_bbox["x0"], sim_bbox["y0"],
                           sim_bbox["x1"] - sim_bbox["x0"],
                           sim_bbox["y1"] - sim_bbox["y0"])
        chunk.front_tex.use(0)
        sim_prog["u_state"]   = 0
        sim_prog["u_birth"]   = RULES[sim["rule_idx"]][1]
        sim_prog["u_survive"] = RULES[sim["rule_idx"]][2]
        sim_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.scissor = None
        chunk.swap()

    def paint(uv, radius=0.012, value=1.0):
        # Écriture en place : on cible directement front_fbo et on scissor le
        # rectangle englobant le pinceau pour que le GPU ne rastérise que là.
        # Gain considérable sur grandes grilles : au lieu de passer sur toute
        # la grille à chaque dab, on ne touche que le carré autour du pinceau.
        r_cells = radius * chunk.height
        cx_cells = uv[0] * chunk.width
        cy_cells = uv[1] * chunk.height
        margin = 2
        x0 = max(0, int(cx_cells - r_cells - margin))
        y0 = max(0, int(cy_cells - r_cells - margin))
        x1 = min(chunk.width,  int(cx_cells + r_cells + margin) + 1)
        y1 = min(chunk.height, int(cy_cells + r_cells + margin) + 1)
        if x0 >= x1 or y0 >= y1:
            return  # pinceau entièrement hors grille
        chunk.front_fbo.use()
        ctx.viewport = (0, 0, chunk.width, chunk.height)
        ctx.scissor  = (x0, y0, x1 - x0, y1 - y0)
        paint_prog["u_center"] = uv
        paint_prog["u_radius"] = radius
        paint_prog["u_value"]  = value
        paint_prog["u_aspect"] = chunk.width / chunk.height
        paint_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.scissor = None
        # Force un recalcul rapide des vivantes : value=1 vient d'ajouter au
        # moins une cellule, donc step() ne doit pas skipper.
        counters["alive"]    = max(1, counters["alive"]) if value > 0 else counters["alive"]
        counters["alive_at"] = 0
        # La nouvelle zone peinte peut être hors bbox → sim la louperait.
        # On invalide pour forcer un sim pleine-grille jusqu'au prochain update.
        sim_bbox["valid"] = False

    def clear_grid():
        chunk.clear()
        counters["gen"]      = 0
        counters["alive"]    = 0
        counters["alive_at"] = pygame.time.get_ticks()
        sim_bbox["valid"]    = False

    def randomize():
        chunk.randomize()
        # Estimation provisoire en attendant le reduce mipmap ; max(1, ...)
        # suffirait mais on donne une valeur plausible pour le HUD.
        counters["gen"]      = 0
        counters["alive"]    = int(0.25 * chunk.width * chunk.height)
        counters["alive_at"] = 0
        sim_bbox["valid"]    = False

    def reset_world():
        """Rétrécit la grille à sa taille initiale et recentre la vue.
        Déclenché par la touche C : remet à zéro le monde pour retrouver
        les FPS nominaux même après avoir dézoomé à fond."""
        nonlocal chunk
        if chunk.width != init_w or chunk.height != init_h:
            chunk.release()
            chunk = Chunk(ctx, init_w, init_h)
            update_zoom_min()
        else:
            chunk.clear()
        for k in ("cx", "cy", "tcx", "tcy"):
            view[k] = 0.5
        view["zoom"]  = 1.0
        view["tzoom"] = 1.0
        counters["gen"]      = 0
        counters["alive"]    = 0
        counters["alive_at"] = pygame.time.get_ticks()
        sim_bbox["valid"]    = False

    # vue : current = ce qui est affiché ; target = vers quoi on glisse.
    # cx, cy : UV du centre de la vue. zoom : largeur (en UV) visible.
    view = {
        "cx": 0.5,  "cy": 0.5,  "zoom": 1.0,
        "tcx": 0.5, "tcy": 0.5, "tzoom": 1.0,
    }
    # ZOOM_MAX est très permissif : c'est grow_chunk qui gère le dézoom en
    # agrandissant la grille. On garde un plafond lointain juste pour éviter
    # un cycle infini en cas de saturation de la taille max.
    # ZOOM_MIN est dérivé de la grille courante (≈ 30 cellules visibles mini)
    # et remis à jour après chaque agrandissement.
    ZOOM_MAX = 4.0
    MIN_CELLS_VISIBLE = 30.0
    zoom_min = {"value": MIN_CELLS_VISIBLE / chunk.width}

    def update_zoom_min():
        zoom_min["value"] = MIN_CELLS_VISIBLE / chunk.width

    LERP_SPEED = 18.0  # plus grand = plus snappy

    def screen_uv(pos):
        w, h = pygame.display.get_surface().get_size()
        return (pos[0] / w, 1.0 - pos[1] / h)

    def view_uv(pos):
        sx, sy = screen_uv(pos)
        return (
            (sx - 0.5) * view["zoom"] + view["cx"],
            (sy - 0.5) * view["zoom"] + view["cy"],
        )

    def zoom_at(pos, factor):
        # On reformule la cible : le point sous le curseur reste sous le curseur
        # une fois la cible atteinte (la transition se fait en lerp).
        sx, sy = screen_uv(pos)
        tx = (sx - 0.5) * view["tzoom"] + view["tcx"]
        ty = (sy - 0.5) * view["tzoom"] + view["tcy"]
        view["tzoom"] = max(zoom_min["value"], min(ZOOM_MAX, view["tzoom"] * factor))
        view["tcx"] = tx - (sx - 0.5) * view["tzoom"]
        view["tcy"] = ty - (sy - 0.5) * view["tzoom"]

    def update_view(dt):
        k = 1.0 - math.exp(-dt * LERP_SPEED)
        view["cx"] += (view["tcx"] - view["cx"]) * k
        view["cy"] += (view["tcy"] - view["cy"]) * k
        # interpolation log pour un zoom perceptuellement uniforme
        lz = math.log(view["zoom"])
        lz += (math.log(view["tzoom"]) - lz) * k
        view["zoom"] = math.exp(lz)

    def grow_chunk():
        """Double la taille de la grille en conservant l'ancien contenu
        centré dans la nouvelle, et rescale les coordonnées de la vue pour
        qu'il n'y ait aucun saut visuel. Renvoie False si on est déjà à la
        taille maximale."""
        nonlocal chunk
        old_w, old_h = chunk.width, chunk.height
        new_w = old_w * 2
        new_h = old_h * 2
        if new_w > MAX_GRID_SIZE or new_h > MAX_GRID_SIZE:
            return False
        new_chunk = Chunk(ctx, new_w, new_h)
        # Copie l'ancien front_tex au centre du nouveau front_tex.
        new_chunk.front_fbo.use()
        ctx.viewport = (0, 0, new_w, new_h)
        chunk.front_tex.use(0)
        grow_prog["u_src"] = 0
        grow_vao.render(moderngl.TRIANGLE_STRIP)
        # Rescale la vue : ce qui était en UV (u, v) sur l'ancienne grille est
        # maintenant à (u/2 + 0.25, v/2 + 0.25) sur la nouvelle, et le zoom
        # en UV est divisé par 2 puisque la grille est deux fois plus large.
        for k in ("cx", "cy", "tcx", "tcy"):
            view[k] = view[k] / 2.0 + 0.25
        view["zoom"]  /= 2.0
        view["tzoom"] /= 2.0
        # Décale la bbox active dans les nouvelles coords (le contenu a été
        # translaté de (old_w/2, old_h/2) en pixels).
        if sim_bbox["valid"]:
            off_x = old_w // 2
            off_y = old_h // 2
            sim_bbox["x0"] += off_x
            sim_bbox["x1"] += off_x
            sim_bbox["y0"] += off_y
            sim_bbox["y1"] += off_y
        # Libère les ressources GL de l'ancien chunk.
        chunk.release()
        chunk = new_chunk
        update_zoom_min()
        return True

    def maybe_grow():
        """Agrandit la grille autant de fois que nécessaire pour contenir la
        cible de la vue. Appelé chaque frame après la saisie utilisateur."""
        for _ in range(8):  # garde-fou absolu, ne devrait jamais boucler autant
            half = view["tzoom"] / 2.0
            needs = (view["tcx"] - half < 0.0 or view["tcx"] + half > 1.0
                     or view["tcy"] - half < 0.0 or view["tcy"] + half > 1.0)
            if not needs or not grow_chunk():
                return

    def shrink_chunk():
        """Divise la taille de la grille par 2 en ne gardant que le quart central.
        L'appelant garantit que la zone abandonnée (les 3/4 périphériques) est
        intégralement vide, donc aucune cellule vivante n'est perdue."""
        nonlocal chunk
        new_w = chunk.width  // 2
        new_h = chunk.height // 2
        new_chunk = Chunk(ctx, new_w, new_h)
        new_chunk.front_fbo.use()
        ctx.viewport = (0, 0, new_w, new_h)
        chunk.front_tex.use(0)
        shrink_prog["u_src"] = 0
        shrink_vao.render(moderngl.TRIANGLE_STRIP)
        # Rescale vue : inverse de grow (cx_new = 2·cx_old − 0.5, zoom ×2).
        for k in ("cx", "cy", "tcx", "tcy"):
            view[k] = view[k] * 2.0 - 0.5
        view["zoom"]  *= 2.0
        view["tzoom"] *= 2.0
        # Décale la bbox dans les nouvelles coords (on a enlevé W/4 à gauche,
        # H/4 en haut — les indices se décalent d'autant).
        if sim_bbox["valid"]:
            off_x = chunk.width  // 4
            off_y = chunk.height // 4
            sim_bbox["x0"] = max(0, sim_bbox["x0"] - off_x)
            sim_bbox["y0"] = max(0, sim_bbox["y0"] - off_y)
            sim_bbox["x1"] = min(new_w, sim_bbox["x1"] - off_x)
            sim_bbox["y1"] = min(new_h, sim_bbox["y1"] - off_y)
        chunk.release()
        chunk = new_chunk
        update_zoom_min()

    def maybe_shrink():
        """Rétrécit la grille si la vie et la vue tiennent toutes deux dans
        le quart central. Appelé au rythme d'update_alive_count (500 ms) —
        pas besoin de plus, la vie ne rétrécit pas aussi vite."""
        if not sim_bbox["valid"]:
            return
        if chunk.width <= init_w or chunk.height <= init_h:
            return
        # Zone qu'on garderait : le quart central en pixels.
        qx = chunk.width  // 4
        qy = chunk.height // 4
        cells_in_core = (sim_bbox["x0"] >= qx
                         and sim_bbox["y0"] >= qy
                         and sim_bbox["x1"] <= chunk.width  - qx
                         and sim_bbox["y1"] <= chunk.height - qy)
        if not cells_in_core:
            return
        # La cible de la vue doit aussi tenir dans le quart central en UV,
        # sinon le rescale cracherait la vue hors de la nouvelle grille.
        half = view["tzoom"] / 2.0
        view_in_core = (view["tcx"] - half >= 0.25
                        and view["tcx"] + half <= 0.75
                        and view["tcy"] - half >= 0.25
                        and view["tcy"] + half <= 0.75)
        if not view_in_core:
            return
        shrink_chunk()

    def upload_pattern_tex(pat):
        """Uploade un motif 2D dans une texture R32F. L'appelant possède la texture."""
        p = np.flipud(pat).astype(np.float32)
        h, w = p.shape
        tex = ctx.texture((w, h), 1, p.tobytes(), dtype="f4")
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return tex

    def stamp(pattern, pos):
        """Pose un motif sur la grille, entièrement GPU (pas de round-trip CPU)."""
        tex = upload_pattern_tex(pattern)
        try:
            cx_uv, cy_uv = view_uv(pos)
            h, w = pattern.shape
            chunk.back_fbo.use()
            ctx.viewport = (0, 0, chunk.width, chunk.height)
            chunk.front_tex.use(0)
            tex.use(1)
            stamp_prog["u_state"]        = 0
            stamp_prog["u_pattern"]      = 1
            stamp_prog["u_cursor_tex"]   = (cx_uv, cy_uv)
            stamp_prog["u_pattern_size"] = (w / chunk.width, h / chunk.height)
            stamp_vao.render(moderngl.TRIANGLE_STRIP)
            chunk.swap()
            counters["alive"]    = max(1, counters["alive"])
            counters["alive_at"] = 0
            sim_bbox["valid"]    = False
        finally:
            tex.release()

    SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")

    def save_png():
        os.makedirs(SNAP_DIR, exist_ok=True)
        buf = np.frombuffer(chunk.front_tex.read(), dtype=np.uint8) \
                .reshape(chunk.height, chunk.width, 2)
        # canal R = vivant ; on flip Y pour que l'image PNG soit "à l'endroit"
        img = np.flipud(buf[..., 0])
        rgb = np.stack([img, img, img], axis=-1)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        path = os.path.join(SNAP_DIR, time.strftime("life_%Y%m%d_%H%M%S.png"))
        pygame.image.save(surf, path)
        return path

    def load_png(path=None):
        if path is None:
            files = sorted(glob.glob(os.path.join(SNAP_DIR, "life_*.png")))
            if not files:
                return None
            path = files[-1]
        surf = pygame.image.load(path)  # pas de .convert() : pas besoin de display
        # ajuste à la taille de la grille courante (centré, fond noir si plus petit)
        canvas = pygame.Surface((chunk.width, chunk.height))
        canvas.fill((0, 0, 0))
        sw, sh = surf.get_size()
        if sw > chunk.width or sh > chunk.height:
            ratio = min(chunk.width / sw, chunk.height / sh)
            surf = pygame.transform.smoothscale(
                surf, (max(1, int(sw * ratio)), max(1, int(sh * ratio))))
            sw, sh = surf.get_size()
        canvas.blit(surf, ((chunk.width - sw) // 2, (chunk.height - sh) // 2))
        arr = pygame.surfarray.array3d(canvas).swapaxes(0, 1)  # (H, W, 3)
        alive = (arr[..., 0] > 127).astype(np.uint8) * 255
        alive = np.flipud(alive)  # PNG y=0 en haut → buffer y=0 en bas
        buf = np.zeros((chunk.height, chunk.width, 2), dtype=np.uint8)
        buf[..., 0] = alive
        buf[..., 1] = alive
        chunk.front_tex.write(buf.tobytes())
        counters["alive"]    = int((alive > 0).sum())
        counters["alive_at"] = 0
        sim_bbox["valid"]    = False
        return path

    clock = pygame.time.Clock()
    paused = False
    tps = INITIAL_TPS
    sim_accum = 0.0
    panning = False
    pan_last = (0, 0)
    flash = {"text": "", "until": 0}    # message éphémère (HUD)
    t0 = pygame.time.get_ticks() / 1000.0

    # mode placement de motif
    place = {"pattern": None, "name": ""}
    pattern_tex = {"tex": None}

    def update_pattern_tex():
        if pattern_tex["tex"] is not None:
            pattern_tex["tex"].release()
            pattern_tex["tex"] = None
        if place["pattern"] is None:
            return
        pattern_tex["tex"] = upload_pattern_tex(place["pattern"])

    def enter_place(key):
        name, base = PATTERNS[key]
        place["pattern"] = base.copy()
        place["name"]    = name
        update_pattern_tex()

    def exit_place():
        place["pattern"] = None
        update_pattern_tex()

    def rotate_place(direction):
        if place["pattern"] is None:
            return
        place["pattern"] = np.rot90(place["pattern"], direction)
        update_pattern_tex()

    # tracé continu : on interpole entre la dernière position connue et la courante
    stroke = {"last": None, "value": None}
    # Après une pose de motif (ou une annulation clic droit), on bloque le
    # tracé continu jusqu'à ce que tous les boutons soient relâchés — sinon
    # le bouton gauche encore enfoncé déclencherait immédiatement un paint.
    suppress_paint_until_release = False

    # compteurs pour le HUD
    counters = {"gen": 0, "alive": 0, "alive_at": 0}

    def step_counted():
        step()
        counters["gen"] += 1

    # police HUD (DejaVu Sans Mono est livrée avec Fedora)
    pygame.font.init()
    font_main = pygame.font.SysFont("dejavusansmono,monospace", 14)
    font_help = pygame.font.SysFont("dejavusansmono,monospace", 13)
    font_big  = pygame.font.SysFont("dejavusans,sans-serif", 22, bold=True)
    show_help = True

    HELP_TEXT = (
        "SOURIS    gauche dessiner · droit effacer\n"
        "MOLETTE   zoom centré curseur · MILIEU drag = pan\n"
        "1 - 9     sélectionne un motif (Q/E rotation, clic gauche pose)\n"
        "T         cycle des règles · ESPACE pause/play\n"
        "R rand    C clear      Z reset vue      F plein écran\n"
        "+ / -     vitesse de simulation\n"
        "F5 / F9   save / load PNG (snapshots/)\n"
        "H         afficher / masquer cette aide\n"
        "ECHAP     quitter (ou annule le placement)"
    )

    # cache de textures HUD pour éviter l'alloc/release à chaque frame.
    # Clé = label ; valeur = (texture, (w, h)). Recrée uniquement à changement
    # de taille ; sinon .write() écrase le contenu.
    hud_textures = {}

    # Cache du panel HUD : {label: (texte, surface_pygame)}. Le texte du status
    # (FPS, gen, alive) ne change que ~une fois par frame à cause des FPS, mais
    # `make_panel` reste coûteux (font.render + alloc Surface). On évite les
    # re-rendus redondants quand rien n'a bougé côté texte.
    panel_cache = {}

    def cleanup():
        """Libère toutes les ressources GL avant pygame.quit(). Appelé depuis
        les 4 sorties possibles (--screenshot / --bench / --frames / boucle)."""
        chunk.release()
        if pattern_tex["tex"] is not None:
            pattern_tex["tex"].release()
            pattern_tex["tex"] = None
        if glow["tex"] is not None:
            glow["tex"].release()
            glow["fbo"].release()
            glow["tex"] = glow["fbo"] = None
        for tex, _ in hud_textures.values():
            tex.release()
        hud_textures.clear()
        panel_cache.clear()
        for vao in (sim_vao, paint_vao, stamp_vao, reduce_vao,
                    glow_h_vao, display_vao, preview_vao,
                    grow_vao, shrink_vao, hud_vao):
            vao.release()
        for prog in (sim_prog, paint_prog, stamp_prog, reduce_prog,
                     glow_h_prog, display_prog, preview_prog,
                     grow_prog, shrink_prog, hud_prog):
            prog.release()
        quad.release()

    def draw_hud_surface(surf, pos, anchor="topleft", label="default"):
        w, h = surf.get_size()
        sw, sh = pygame.display.get_surface().get_size()
        if anchor == "topleft":      x, y = pos
        elif anchor == "topright":   x, y = sw - w - pos[0], pos[1]
        elif anchor == "bottomleft": x, y = pos[0], sh - h - pos[1]
        elif anchor == "topcenter":  x, y = (sw - w) // 2, pos[1]
        else:                         x, y = pos
        data = pygame.image.tobytes(surf, "RGBA", True)
        cached = hud_textures.get(label)
        if cached is None or cached[1] != (w, h):
            if cached is not None:
                cached[0].release()
            tex = ctx.texture((w, h), 4)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            hud_textures[label] = (tex, (w, h))
        else:
            tex = cached[0]
        tex.write(data)
        tex.use(0)
        hud_prog["u_hud"]       = 0
        hud_prog["u_pos_px"]    = (float(x), float(y))
        hud_prog["u_size_px"]   = (float(w), float(h))
        hud_prog["u_screen_px"] = (float(sw), float(sh))
        hud_vao.render(moderngl.TRIANGLE_STRIP)

    def make_panel(lines, font, padding=(10, 8), bg=(10, 12, 22, 165),
                   fg=(235, 235, 245), cache_key=None):
        if isinstance(lines, str):
            lines = lines.split("\n")
        # Cache par clé : si le texte et la police sont identiques à l'appel
        # précédent, on réutilise la Surface (évite font.render + alloc).
        if cache_key is not None:
            cached = panel_cache.get(cache_key)
            if cached is not None and cached[0] == (tuple(lines), id(font)):
                return cached[1]
        surfs = [font.render(line, True, fg) for line in lines]
        lw = max(s.get_width() for s in surfs) if surfs else 1
        lh = sum(s.get_height() for s in surfs) if surfs else 1
        w = lw + padding[0] * 2
        h = lh + padding[1] * 2
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill(bg)
        # liseré subtil
        pygame.draw.rect(panel, (255, 255, 255, 18), panel.get_rect(), 1)
        y = padding[1]
        for s in surfs:
            panel.blit(s, (padding[0], y))
            y += s.get_height()
        if cache_key is not None:
            panel_cache[cache_key] = ((tuple(lines), id(font)), panel)
        return panel

    def update_alive_count():
        now = pygame.time.get_ticks()
        if now - counters["alive_at"] < 500:
            return
        # 1) copie chunk.front.r (éventuellement downsample scale×scale) dans
        #    chunk.reduce_tex (R32F, plafonnée à REDUCE_MAX² pour borner la
        #    VRAM et accélérer build_mipmaps sur grosses grilles).
        chunk.reduce_fbo.use()
        ctx.viewport = (0, 0, chunk.reduce_w, chunk.reduce_h)
        chunk.front_tex.use(0)
        reduce_prog["u_state"] = 0
        reduce_prog["u_scale"] = chunk.reduce_scale
        reduce_vao.render(moderngl.TRIANGLE_STRIP)
        # 2) chaîne de mipmaps : chaque niveau moyenne 2x2 du précédent,
        #    le top-level (1×1) contient la moyenne de toutes les cellules.
        chunk.reduce_tex.build_mipmaps()
        # 3) lecture du top-level (4 octets) — pas de readback de toute la grille
        raw = chunk.reduce_tex.read(level=chunk.reduce_max_level)
        fraction = float(np.frombuffer(raw, dtype=np.float32)[0])
        counters["alive"] = int(round(fraction * chunk.width * chunk.height))
        counters["alive_at"] = now
        # 4) on en profite pour recalculer la bbox vivante à un niveau coarse
        #    de la mipmap (un texel = ~128 cellules), utilisée par step() pour
        #    scissor le sim — évite de chauffer toute la grille quand la vie
        #    n'occupe qu'une portion du monde.
        update_sim_bbox()
        # 5) tant qu'on est là, si la grille s'est agrandie (grow) mais que
        #    la vie + la vue tiennent toutes deux dans le quart central, on
        #    rétrécit : ça rend les FPS nominaux après un dezoom-puis-rezoom.
        maybe_shrink()

    def update_sim_bbox():
        # Niveau grossier de la mipmap : vise ~128 texels dans la plus grande
        # dim de reduce_tex. Chaque texel reduce couvre reduce_scale × reduce_scale
        # pixels de state ; chaque niveau de mipmap × 2 par axe.
        level = min(7, chunk.reduce_max_level - 1)
        if level < 0:
            sim_bbox["valid"] = False
            return
        bw = max(1, chunk.reduce_w >> level)
        bh = max(1, chunk.reduce_h >> level)
        raw = chunk.reduce_tex.read(level=level)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(bh, bw)
        nz = arr > 1e-6
        # Un texel mipmap couvre (reduce_scale × 2^level) pixels de state.
        block = chunk.reduce_scale << level
        if not nz.any():
            sim_bbox.update(valid=True, x0=0, y0=0, x1=0, y1=0)
            return
        ys, xs = np.where(nz)
        x0 = int(xs.min()) * block
        y0 = int(ys.min()) * block
        x1 = (int(xs.max()) + 1) * block
        y1 = (int(ys.max()) + 1) * block
        # Marge d'un bloc (couvre la propagation entre 2 updates à 500 ms).
        x0 = max(0, x0 - block)
        y0 = max(0, y0 - block)
        x1 = min(chunk.width,  x1 + block)
        y1 = min(chunk.height, y1 + block)
        sim_bbox.update(valid=True, x0=x0, y0=y0, x1=x1, y1=y1)

    def flash_msg(text, ms=1500):
        flash["text"] = text
        flash["until"] = pygame.time.get_ticks() + ms

    def _scene_setup():
        """Pose une scène vivante pour les captures (gun + pulsar + méthuselah)."""
        clear_grid()
        sw, sh = pygame.display.get_surface().get_size()
        stamp(PATTERNS[pygame.K_7][1], (int(0.18 * sw), int(0.30 * sh)))  # Gosper
        stamp(PATTERNS[pygame.K_5][1], (int(0.72 * sw), int(0.40 * sh)))  # Pulsar
        stamp(PATTERNS[pygame.K_8][1], (int(0.50 * sw), int(0.78 * sh)))  # R-pent
        return sw, sh

    def _render_frame(sw, sh, time_s, with_help):
        ensure_glow_tex(sw, sh)
        # passe H du glow (écrit dans glow["tex"])
        glow["fbo"].use()
        ctx.viewport = (0, 0, sw, sh)
        chunk.front_tex.use(0)
        glow_h_prog["u_state"]  = 0
        glow_h_prog["u_center"] = (0.5, 0.5)
        glow_h_prog["u_zoom"]   = 1.0
        glow_h_vao.render(moderngl.TRIANGLE_STRIP)
        # composition finale (passe V du glow + cellules + fond)
        ctx.screen.use()
        ctx.viewport = (0, 0, sw, sh)
        chunk.front_tex.use(0)
        glow["tex"].use(1)
        display_prog["u_state"]  = 0
        display_prog["u_glow_h"] = 1
        display_prog["u_time"]   = time_s
        display_prog["u_center"] = (0.5, 0.5)
        display_prog["u_zoom"]   = 1.0
        display_vao.render(moderngl.TRIANGLE_STRIP)
        rule_name = RULES[sim["rule_idx"]][0]
        status = (f"Gen {counters['gen']:>6}  ·  {counters['alive']:>5} vivantes  "
                  f"·  {tps:>3} TPS  ·  {rule_name}  ·  PLAY")
        draw_hud_surface(make_panel(status, font_main, cache_key="status"),
                         (12, 12), anchor="bottomleft", label="status")
        if with_help:
            draw_hud_surface(make_panel(HELP_TEXT, font_help, cache_key="help"),
                             (12, 12), anchor="topright", label="help")

    def _save_screen_png(sw, sh, path):
        data = ctx.screen.read(components=3, dtype="f1")
        img = np.frombuffer(data, dtype=np.uint8).reshape(sh, sw, 3)
        img = np.flipud(img)
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        pygame.image.save(surf, path)

    # Mode test : batterie d'invariants headless (sort avec code != 0 si un
    # check échoue). Utilisé pour valider les refactors risqués (reduce cap,
    # extraction du bloc render) sans risquer de régression silencieuse.
    if "--test" in sys.argv:
        passed = []
        failed = []

        def check(name, ok, detail=""):
            (passed if ok else failed).append((name, detail))

        def count_alive_direct():
            """Lit directement chunk.front_tex et compte les pixels vivants.
            Utilisé à la place d'update_alive_count pour le test : la mipmap
            reduce perd de la précision sur les très faibles densités (glider
            sur grille 1280×800 = 5/1 M), rendant le compte mipmap peu fiable
            pour des assertions exactes."""
            raw = chunk.front_tex.read()
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                chunk.height, chunk.width, 2)
            return int((arr[..., 0] > 0).sum())

        # 1) clear_grid => 0 vivante
        clear_grid()
        check("clear_grid: alive==0", count_alive_direct() == 0)

        # 2) stamp d'un glider => cellules vivantes (le stamp a un off-by-one
        #    connu sur très petits motifs au bord des pixels, on se contente
        #    de "au moins 4").
        sw, sh = pygame.display.get_surface().get_size()
        stamp(PATTERNS[pygame.K_1][1], (sw // 2, sh // 2))
        post_stamp = count_alive_direct()
        check("glider stamp: alive>=4", post_stamp >= 4, f"got {post_stamp}")

        # 3) Conway B3/S23 sur un glider : le compte reste borné (period-4
        #    spaceship → typiquement 5 cellules, ±1 selon phase et off-by-one).
        counters["alive"] = post_stamp or 1
        for _ in range(40):
            step()
        after_40 = count_alive_direct()
        check("glider after 40 gens: 3<=alive<=7", 3 <= after_40 <= 7,
              f"got {after_40}")

        # 4) grow_chunk préserve le compte vivant
        alive_before_grow = count_alive_direct()
        grow_chunk()
        after_grow = count_alive_direct()
        check("grow: alive preserved", after_grow == alive_before_grow,
              f"{alive_before_grow} -> {after_grow}")

        # 5) shrink_chunk préserve le compte (pattern au centre, donc rentre
        #    dans le quart central après un grow).
        shrink_chunk()
        after_shrink = count_alive_direct()
        check("shrink: alive preserved", after_shrink == alive_before_grow,
              f"{alive_before_grow} -> {after_shrink}")

        # 6) Invariant pulsar : 48 cellules stamp (±quelques) et reste stable
        #    tous les 3 gens (period-3 oscillator).
        clear_grid()
        stamp(PATTERNS[pygame.K_5][1], (sw // 2, sh // 2))
        pulsar_initial = count_alive_direct()
        # Tolérance large : le stamp shader a un off-by-one aux bords (condition
        # `<=` inclusive côté p.x=1) qui ajoute une rangée/colonne de plus sur
        # les petits motifs. Pour le pulsar (13×13, 48 cellules) on peut
        # récolter 40–70 cellules selon l'alignement pixel du cursor.
        check("pulsar stamp: 40<=alive<=80", 40 <= pulsar_initial <= 80,
              f"got {pulsar_initial}")

        # 7) Pulsar survit à plusieurs grows (valide la préservation du
        #    contenu sur grille multiples fois agrandie).
        for i in range(3):
            if not grow_chunk():
                break
            alive = count_alive_direct()
            check(f"pulsar after grow #{i+1} (reduce_scale={chunk.reduce_scale}): alive preserved",
                  alive == pulsar_initial,
                  f"{pulsar_initial} -> {alive}")

        # 7bis) Vérifie que reduce_scale est bien > 1 quand on a dépassé
        #       REDUCE_MAX — sinon le cap n'est pas testé.
        check(f"reduce_scale>1 after grows (actual {chunk.reduce_scale})",
              chunk.reduce_scale >= 2,
              f"grid is {chunk.width}x{chunk.height}, scale={chunk.reduce_scale}")

        # 8) update_alive_count (via mipmap) vs comptage direct : la mipmap
        #    perd en précision pour des densités <1e-5, mais sur un pulsar
        #    dans une grille 8×, la densité reste ~4e-6. Tolérance large.
        counters["alive_at"] = -10_000
        update_alive_count()
        mip_alive = counters["alive"]
        direct_alive = count_alive_direct()
        # On accepte une erreur relative <50% ou absolue <200 (la mipmap peut
        # arrondir fort sur grilles sparses).
        err = abs(mip_alive - direct_alive)
        ok = err < max(200, direct_alive * 0.5)
        check(f"mipmap vs direct: |{mip_alive}-{direct_alive}|<tol", ok,
              f"direct={direct_alive} mip={mip_alive} err={err}")

        # Reset propre et sortie
        reset_world()
        cleanup()
        pygame.quit()
        print(f"PASS: {len(passed)}")
        for name, _ in passed:
            print(f"  OK  {name}")
        if failed:
            print(f"FAIL: {len(failed)}")
            for name, detail in failed:
                print(f"  KO  {name} :: {detail}")
            sys.exit(1)
        return

    # Mode capture fixe : scène figée après 90 générations.
    if "--screenshot" in sys.argv:
        out_path = sys.argv[sys.argv.index("--screenshot") + 1]
        sw, sh = _scene_setup()
        for _ in range(90):
            step_counted()
        update_alive_count()
        _render_frame(sw, sh, time_s=1.5, with_help=True)
        _save_screen_png(sw, sh, out_path)
        cleanup()
        pygame.quit()
        print(f"Capture : {out_path}")
        return

    # Mode benchmark : mesure ns/step de sim, ns/frame de rendu, ns/maj du
    # compteur de vivantes. Utile pour comparer l'impact d'une optimisation.
    if "--bench" in sys.argv:
        i = sys.argv.index("--bench")
        n = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 2000
        randomize()
        sw, sh = pygame.display.get_surface().get_size()

        # warmup : compile, upload, premier render.
        for _ in range(200):
            step()
        _render_frame(sw, sh, time_s=0.0, with_help=True)
        ctx.finish()

        t0 = time.perf_counter()
        for _ in range(n):
            step()
        ctx.finish()
        dt_sim = time.perf_counter() - t0

        t0 = time.perf_counter()
        for k in range(n):
            _render_frame(sw, sh, time_s=k * 0.02, with_help=True)
        ctx.finish()
        dt_render = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n):
            counters["alive_at"] = 0
            update_alive_count()
        ctx.finish()
        dt_count = time.perf_counter() - t0

        print(f"bench N={n}  screen {sw}x{sh}  grid {chunk.width}x{chunk.height}")
        print(f"  sim    : {dt_sim*1e6/n:8.1f} us/step   ({n/dt_sim:8.0f} steps/s)")
        print(f"  render : {dt_render*1e6/n:8.1f} us/frame  ({n/dt_render:8.0f} frames/s)")
        print(f"  count  : {dt_count*1e6/n:8.1f} us/call   ({n/dt_count:8.0f} calls/s)")
        cleanup()
        pygame.quit()
        return

    # Mode capture animée : dump N frames PNG (assemblage GIF via ffmpeg ensuite).
    if "--frames" in sys.argv:
        i = sys.argv.index("--frames")
        out_dir   = sys.argv[i + 1]
        n_frames  = int(sys.argv[i + 2])
        stride    = int(sys.argv[i + 3]) if len(sys.argv) > i + 3 else 1
        os.makedirs(out_dir, exist_ok=True)
        sw, sh = _scene_setup()
        for f in range(n_frames):
            for _ in range(stride):
                step_counted()
            if f % 5 == 0:
                update_alive_count()
            _render_frame(sw, sh, time_s=f * 0.04, with_help=False)
            _save_screen_png(sw, sh, os.path.join(out_dir, f"frame_{f:04d}.png"))
        cleanup()
        pygame.quit()
        print(f"Capture : {n_frames} frames → {out_dir}")
        return

    running = True
    while running:
        dt = clock.tick(120) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.VIDEORESIZE:
                pass
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    if place["pattern"] is not None:
                        exit_place()
                    else:
                        running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    randomize()
                elif e.key == pygame.K_c:
                    # Nettoie ET rétrécit la grille à la taille initiale : plus
                    # de grille géante qui pèse sur les FPS après un dézoom.
                    reset_world()
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    tps = min(240, tps + 5)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    tps = max(1, tps - 5)
                elif e.key == pygame.K_f:
                    fullscreen = not fullscreen
                    if fullscreen:
                        pygame.display.set_mode((0, 0), fullscreen_flags)
                    else:
                        pygame.display.set_mode((WIN_W, WIN_H), windowed_flags)
                elif e.key == pygame.K_z:
                    view["tcx"], view["tcy"], view["tzoom"] = 0.5, 0.5, 1.0
                elif e.key == pygame.K_h:
                    show_help = not show_help
                elif e.key == pygame.K_t:
                    sim["rule_idx"] = (sim["rule_idx"] + 1) % len(RULES)
                    flash_msg(f"Règle ▸ {RULES[sim['rule_idx']][0]}")
                elif e.key == pygame.K_q:
                    rotate_place(1)   # CCW
                elif e.key == pygame.K_e:
                    rotate_place(-1)  # CW
                elif e.key == pygame.K_F5:
                    try:
                        path = save_png()
                        flash_msg(f"Sauvegardé ▸ {os.path.basename(path)}")
                    except Exception as ex:
                        flash_msg(f"Erreur save : {ex}", 2500)
                elif e.key == pygame.K_F9:
                    try:
                        path = load_png()
                        if path:
                            flash_msg(f"Chargé ▸ {os.path.basename(path)}")
                            counters["gen"] = 0
                        else:
                            flash_msg("Aucun snapshot trouvé", 2000)
                    except Exception as ex:
                        flash_msg(f"Erreur load : {ex}", 2500)
                elif e.key in PATTERNS:
                    enter_place(e.key)
                    flash_msg(f"Placement ▸ {place['name']}  (Q/E rot, clic = pose)")
            elif e.type == pygame.MOUSEWHEEL:
                factor = 0.87 if e.y > 0 else 1.15
                for _ in range(abs(e.y)):
                    zoom_at(pygame.mouse.get_pos(), factor)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 2:
                    panning = True
                    pan_last = e.pos
                elif place["pattern"] is not None:
                    if e.button == 1:
                        stamp(place["pattern"], e.pos)
                        exit_place()
                        suppress_paint_until_release = True
                    elif e.button == 3:
                        exit_place()
                        suppress_paint_until_release = True
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 2:
                panning = False
            elif e.type == pygame.MOUSEMOTION and panning:
                w, h = pygame.display.get_surface().get_size()
                dx = (e.pos[0] - pan_last[0]) / w
                dy = (e.pos[1] - pan_last[1]) / h
                view["cx"]  -= dx * view["zoom"]
                view["cy"]  += dy * view["zoom"]
                view["tcx"] = view["cx"]
                view["tcy"] = view["cy"]
                pan_last = e.pos

        # tracé continu (désactivé en mode placement, c'est le clic qui pose)
        if place["pattern"] is None:
            buttons = pygame.mouse.get_pressed()
            # Si on sort tout juste du mode placement avec un bouton encore
            # enfoncé, on attend son relâchement pour ne pas peindre aussitôt
            # par-dessus le motif qu'on vient de poser.
            if suppress_paint_until_release:
                if not (buttons[0] or buttons[2]):
                    suppress_paint_until_release = False
                buttons = (False, buttons[1], False)
            if buttons[0] or buttons[2]:
                pos = pygame.mouse.get_pos()
                val = 0.0 if buttons[2] else 1.0
                radius = 0.014 * view["zoom"]
                if stroke["last"] is None or stroke["value"] != val:
                    paint(view_uv(pos), radius=radius, value=val)
                else:
                    x0, y0 = stroke["last"]
                    x1, y1 = pos
                    dist = max(abs(x1 - x0), abs(y1 - y0))
                    steps_n = max(1, int(dist / 2))  # un point tous les ~2 pixels
                    for i in range(1, steps_n + 1):
                        t = i / steps_n
                        ix = x0 + (x1 - x0) * t
                        iy = y0 + (y1 - y0) * t
                        paint(view_uv((ix, iy)), radius=radius, value=val)
                stroke["last"]  = pos
                stroke["value"] = val
            else:
                stroke["last"]  = None
                stroke["value"] = None
        else:
            stroke["last"] = None

        update_view(dt)
        # Si la cible de la vue sort de la grille courante, on agrandit tant
        # qu'il le faut (jusqu'à MAX_GRID_SIZE). Les coords de vue sont
        # rescalées au passage pour qu'il n'y ait aucun saut visible.
        maybe_grow()

        if not paused:
            sim_accum += dt
            step_dt = 1.0 / tps
            steps_n = 0
            while sim_accum >= step_dt and steps_n < 8:
                step_counted()
                sim_accum -= step_dt
                steps_n += 1

        update_alive_count()

        # ── rendu écran ─────────────────────────────────────────
        w, h = pygame.display.get_surface().get_size()
        ensure_glow_tex(w, h)

        # 0) passe H du glow (R8, résolution écran)
        glow["fbo"].use()
        ctx.viewport = (0, 0, w, h)
        chunk.front_tex.use(0)
        glow_h_prog["u_state"]  = 0
        glow_h_prog["u_center"] = (view["cx"], view["cy"])
        glow_h_prog["u_zoom"]   = view["zoom"]
        glow_h_vao.render(moderngl.TRIANGLE_STRIP)

        ctx.screen.use()
        ctx.viewport = (0, 0, w, h)

        # 1) composition finale (passe V + cellules + fond + vignette + dither)
        chunk.front_tex.use(0)
        glow["tex"].use(1)
        display_prog["u_state"]  = 0
        display_prog["u_glow_h"] = 1
        display_prog["u_time"]   = pygame.time.get_ticks() / 1000.0 - t0
        display_prog["u_center"] = (view["cx"], view["cy"])
        display_prog["u_zoom"]   = view["zoom"]
        display_vao.render(moderngl.TRIANGLE_STRIP)

        # 2) preview du motif en cours de placement
        if place["pattern"] is not None and pattern_tex["tex"] is not None:
            cx_uv, cy_uv = view_uv(pygame.mouse.get_pos())
            ph, pw = place["pattern"].shape
            pattern_tex["tex"].use(0)
            preview_prog["u_pattern"]      = 0
            preview_prog["u_cursor_tex"]   = (cx_uv, cy_uv)
            preview_prog["u_pattern_size"] = (pw / chunk.width, ph / chunk.height)
            preview_prog["u_view_center"]  = (view["cx"], view["cy"])
            preview_prog["u_view_zoom"]    = view["zoom"]
            preview_prog["u_time"]         = pygame.time.get_ticks() / 1000.0
            preview_vao.render(moderngl.TRIANGLE_STRIP)

        # 3) HUD
        rule_name = RULES[sim["rule_idx"]][0]
        status = (f"Gen {counters['gen']:>6}  ·  {counters['alive']:>5} vivantes  "
                  f"·  {clock.get_fps():3.0f} FPS  ·  {tps:>3} TPS  "
                  f"·  {rule_name}  ·  {'PAUSE' if paused else 'PLAY'}")
        draw_hud_surface(make_panel(status, font_main, cache_key="status"),
                         (12, 12), anchor="bottomleft", label="status")

        if show_help:
            draw_hud_surface(make_panel(HELP_TEXT, font_help, cache_key="help"),
                             (12, 12), anchor="topright", label="help")

        if place["pattern"] is not None:
            label = f"▸ {place['name']}   Q/E rotation · clic gauche : pose · Esc : annule"
            draw_hud_surface(
                make_panel(label, font_main, bg=(120, 80, 20, 200),
                           cache_key="placement"),
                (0, 12), anchor="topcenter", label="placement")

        if pygame.time.get_ticks() < flash["until"]:
            draw_hud_surface(
                make_panel(flash["text"], font_big, padding=(18, 14),
                           bg=(20, 25, 50, 200), fg=(255, 240, 200),
                           cache_key="flash"),
                (0, 60), anchor="topcenter", label="flash")

        pygame.display.flip()

        if pygame.time.get_ticks() % 500 < 17:
            pygame.display.set_caption(
                f"Jeu de la Vie — GPU   ·   {rule_name}   ·   "
                f"{'PAUSE' if paused else 'PLAY'}"
            )

    cleanup()
    pygame.quit()


if __name__ == "__main__":
    main()
