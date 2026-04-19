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
GRID_W, GRID_H = 2048, 1280        # résolution de la simulation
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

# Peinture à la souris : on écrit dans la texture d'état.
PAINT_SHADER = """
#version 330
uniform sampler2D u_state;
uniform vec2  u_center;     // en coordonnées UV
uniform float u_radius;     // en UV
uniform float u_value;      // 1.0 = dessine, 0.0 = efface
in vec2 v_uv;
out vec4 frag;
void main() {
    vec4 cur = texture(u_state, v_uv);
    vec2 d = (v_uv - u_center);
    d.x *= float(textureSize(u_state, 0).x) / float(textureSize(u_state, 0).y);
    if (length(d) < u_radius) {
        cur.r = u_value;
        cur.g = max(cur.g, u_value);
    }
    frag = cur;
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
REDUCE_SHADER = """
#version 330
uniform sampler2D u_state;
in vec2 v_uv;
out vec4 frag;
void main() {
    frag = vec4(texture(u_state, v_uv).r, 0.0, 0.0, 1.0);
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
    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                    pygame.GL_CONTEXT_PROFILE_CORE)
    screen = pygame.display.set_mode((WIN_W, WIN_H), flags)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)

    # quad plein écran
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())

    sim_prog     = make_program(ctx, SIM_SHADER)
    paint_prog   = make_program(ctx, PAINT_SHADER)
    stamp_prog   = make_program(ctx, STAMP_SHADER)
    reduce_prog  = make_program(ctx, REDUCE_SHADER)
    glow_h_prog  = make_program(ctx, GLOW_H_SHADER)
    display_prog = make_program(ctx, DISPLAY_SHADER)
    preview_prog = make_program(ctx, PREVIEW_SHADER)
    hud_prog     = make_program(ctx, HUD_SHADER)

    sim_vao     = ctx.simple_vertex_array(sim_prog,     quad, "in_pos")
    paint_vao   = ctx.simple_vertex_array(paint_prog,   quad, "in_pos")
    stamp_vao   = ctx.simple_vertex_array(stamp_prog,   quad, "in_pos")
    reduce_vao  = ctx.simple_vertex_array(reduce_prog,  quad, "in_pos")
    glow_h_vao  = ctx.simple_vertex_array(glow_h_prog,  quad, "in_pos")
    display_vao = ctx.simple_vertex_array(display_prog, quad, "in_pos")
    preview_vao = ctx.simple_vertex_array(preview_prog, quad, "in_pos")
    hud_vao     = ctx.simple_vertex_array(hud_prog,     quad, "in_pos")

    def make_tex(initial=None):
        # RG8 : R = alive (0/255), G = age (0..255). 2 o/cellule au lieu de 16.
        tex = ctx.texture((GRID_W, GRID_H), 2, dtype="f1")
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        # Pas de wrap : monde fini, les bords sont absorbants (cf. SIM_SHADER).
        tex.repeat_x = tex.repeat_y = False
        if initial is not None:
            tex.write(initial.tobytes())
        return tex

    tex_a = make_tex(random_state(GRID_W, GRID_H))
    tex_b = make_tex(np.zeros((GRID_H, GRID_W, 2), dtype=np.uint8))
    fbo_a = ctx.framebuffer(color_attachments=[tex_a])
    fbo_b = ctx.framebuffer(color_attachments=[tex_b])

    state = {"front": (tex_a, fbo_a), "back": (tex_b, fbo_b)}

    def swap():
        state["front"], state["back"] = state["back"], state["front"]

    # Réduction pour compter les vivantes : texture R32F à la résolution de
    # la grille + chaîne de mipmaps. Lire le top-level 1×1 donne la fraction
    # de cellules vivantes, qu'on multiplie par GRID_W*GRID_H.
    reduce_tex = ctx.texture((GRID_W, GRID_H), 1, dtype="f4")
    reduce_tex.filter = (moderngl.LINEAR_MIPMAP_NEAREST, moderngl.LINEAR)
    reduce_fbo = ctx.framebuffer(color_attachments=[reduce_tex])
    reduce_max_level = int(math.floor(math.log2(max(GRID_W, GRID_H))))

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

    def step():
        front_tex, _ = state["front"]
        _, back_fbo = state["back"]
        back_fbo.use()
        ctx.viewport = (0, 0, GRID_W, GRID_H)
        front_tex.use(0)
        sim_prog["u_state"]   = 0
        sim_prog["u_birth"]   = RULES[sim["rule_idx"]][1]
        sim_prog["u_survive"] = RULES[sim["rule_idx"]][2]
        sim_vao.render(moderngl.TRIANGLE_STRIP)
        swap()

    def paint(uv, radius=0.012, value=1.0):
        front_tex, _ = state["front"]
        _, back_fbo = state["back"]
        back_fbo.use()
        ctx.viewport = (0, 0, GRID_W, GRID_H)
        front_tex.use(0)
        paint_prog["u_state"] = 0
        paint_prog["u_center"] = uv
        paint_prog["u_radius"] = radius
        paint_prog["u_value"]  = value
        paint_vao.render(moderngl.TRIANGLE_STRIP)
        swap()

    def clear_grid():
        zeros = np.zeros((GRID_H, GRID_W, 2), dtype=np.uint8)
        state["front"][0].write(zeros.tobytes())

    def randomize():
        state["front"][0].write(random_state(GRID_W, GRID_H).tobytes())

    # vue : current = ce qui est affiché ; target = vers quoi on glisse.
    # cx, cy : UV du centre de la vue. zoom : largeur (en UV) visible.
    view = {
        "cx": 0.5,  "cy": 0.5,  "zoom": 1.0,
        "tcx": 0.5, "tcy": 0.5, "tzoom": 1.0,
    }
    ZOOM_MIN, ZOOM_MAX = 0.02, 4.0
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
        view["tzoom"] = max(ZOOM_MIN, min(ZOOM_MAX, view["tzoom"] * factor))
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
            front_tex = state["front"][0]
            _, back_fbo = state["back"]
            back_fbo.use()
            ctx.viewport = (0, 0, GRID_W, GRID_H)
            front_tex.use(0)
            tex.use(1)
            stamp_prog["u_state"]        = 0
            stamp_prog["u_pattern"]      = 1
            stamp_prog["u_cursor_tex"]   = (cx_uv, cy_uv)
            stamp_prog["u_pattern_size"] = (w / GRID_W, h / GRID_H)
            stamp_vao.render(moderngl.TRIANGLE_STRIP)
            swap()
        finally:
            tex.release()

    SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")

    def save_png():
        os.makedirs(SNAP_DIR, exist_ok=True)
        front_tex = state["front"][0]
        buf = np.frombuffer(front_tex.read(), dtype=np.uint8) \
                .reshape(GRID_H, GRID_W, 2)
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
        # ajuste à la taille de la grille (centré, fond noir si plus petit)
        canvas = pygame.Surface((GRID_W, GRID_H))
        canvas.fill((0, 0, 0))
        sw, sh = surf.get_size()
        if sw > GRID_W or sh > GRID_H:
            ratio = min(GRID_W / sw, GRID_H / sh)
            surf = pygame.transform.smoothscale(
                surf, (max(1, int(sw * ratio)), max(1, int(sh * ratio))))
            sw, sh = surf.get_size()
        canvas.blit(surf, ((GRID_W - sw) // 2, (GRID_H - sh) // 2))
        arr = pygame.surfarray.array3d(canvas).swapaxes(0, 1)  # (H, W, 3)
        alive = (arr[..., 0] > 127).astype(np.uint8) * 255
        alive = np.flipud(alive)  # PNG y=0 en haut → buffer y=0 en bas
        buf = np.zeros((GRID_H, GRID_W, 2), dtype=np.uint8)
        buf[..., 0] = alive
        buf[..., 1] = alive
        state["front"][0].write(buf.tobytes())
        return path

    clock = pygame.time.Clock()
    paused = False
    tps = INITIAL_TPS
    sim_accum = 0.0
    fullscreen = False
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
                   fg=(235, 235, 245)):
        if isinstance(lines, str):
            lines = lines.split("\n")
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
        return panel

    def update_alive_count():
        now = pygame.time.get_ticks()
        if now - counters["alive_at"] < 500:
            return
        # 1) copie state.r dans reduce_tex (R32F)
        reduce_fbo.use()
        ctx.viewport = (0, 0, GRID_W, GRID_H)
        state["front"][0].use(0)
        reduce_prog["u_state"] = 0
        reduce_vao.render(moderngl.TRIANGLE_STRIP)
        # 2) chaîne de mipmaps : chaque niveau moyenne 2x2 du précédent,
        #    le top-level (1×1) contient la moyenne de toutes les cellules.
        reduce_tex.build_mipmaps()
        # 3) lecture du top-level (4 octets) — pas de readback de toute la grille
        raw = reduce_tex.read(level=reduce_max_level)
        fraction = float(np.frombuffer(raw, dtype=np.float32)[0])
        counters["alive"] = int(round(fraction * GRID_W * GRID_H))
        counters["alive_at"] = now

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
        state["front"][0].use(0)
        glow_h_prog["u_state"]  = 0
        glow_h_prog["u_center"] = (0.5, 0.5)
        glow_h_prog["u_zoom"]   = 1.0
        glow_h_vao.render(moderngl.TRIANGLE_STRIP)
        # composition finale (passe V du glow + cellules + fond)
        ctx.screen.use()
        ctx.viewport = (0, 0, sw, sh)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        state["front"][0].use(0)
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
        draw_hud_surface(make_panel(status, font_main), (12, 12),
                         anchor="bottomleft", label="status")
        if with_help:
            draw_hud_surface(make_panel(HELP_TEXT, font_help), (12, 12),
                             anchor="topright", label="help")

    def _save_screen_png(sw, sh, path):
        data = ctx.screen.read(components=3, dtype="f1")
        img = np.frombuffer(data, dtype=np.uint8).reshape(sh, sw, 3)
        img = np.flipud(img)
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        pygame.image.save(surf, path)

    # Mode capture fixe : scène figée après 90 générations.
    if "--screenshot" in sys.argv:
        out_path = sys.argv[sys.argv.index("--screenshot") + 1]
        sw, sh = _scene_setup()
        for _ in range(90):
            step_counted()
        update_alive_count()
        _render_frame(sw, sh, time_s=1.5, with_help=True)
        _save_screen_png(sw, sh, out_path)
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

        print(f"bench N={n}  screen {sw}x{sh}  grid {GRID_W}x{GRID_H}")
        print(f"  sim    : {dt_sim*1e6/n:8.1f} us/step   ({n/dt_sim:8.0f} steps/s)")
        print(f"  render : {dt_render*1e6/n:8.1f} us/frame  ({n/dt_render:8.0f} frames/s)")
        print(f"  count  : {dt_count*1e6/n:8.1f} us/call   ({n/dt_count:8.0f} calls/s)")
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
                    counters["gen"] = 0
                elif e.key == pygame.K_c:
                    clear_grid()
                    counters["gen"] = 0
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    tps = min(240, tps + 5)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    tps = max(1, tps - 5)
                elif e.key == pygame.K_f:
                    fullscreen = not fullscreen
                    flags2 = flags | (pygame.FULLSCREEN if fullscreen else 0)
                    pygame.display.set_mode((WIN_W, WIN_H), flags2)
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
        state["front"][0].use(0)
        glow_h_prog["u_state"]  = 0
        glow_h_prog["u_center"] = (view["cx"], view["cy"])
        glow_h_prog["u_zoom"]   = view["zoom"]
        glow_h_vao.render(moderngl.TRIANGLE_STRIP)

        ctx.screen.use()
        ctx.viewport = (0, 0, w, h)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # 1) composition finale (passe V + cellules + fond + vignette + dither)
        state["front"][0].use(0)
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
            preview_prog["u_pattern_size"] = (pw / GRID_W, ph / GRID_H)
            preview_prog["u_view_center"]  = (view["cx"], view["cy"])
            preview_prog["u_view_zoom"]    = view["zoom"]
            preview_prog["u_time"]         = pygame.time.get_ticks() / 1000.0
            preview_vao.render(moderngl.TRIANGLE_STRIP)

        # 3) HUD
        rule_name = RULES[sim["rule_idx"]][0]
        status = (f"Gen {counters['gen']:>6}  ·  {counters['alive']:>5} vivantes  "
                  f"·  {clock.get_fps():3.0f} FPS  ·  {tps:>3} TPS  "
                  f"·  {rule_name}  ·  {'PAUSE' if paused else 'PLAY'}")
        draw_hud_surface(make_panel(status, font_main), (12, 12),
                         anchor="bottomleft", label="status")

        if show_help:
            draw_hud_surface(make_panel(HELP_TEXT, font_help), (12, 12),
                             anchor="topright", label="help")

        if place["pattern"] is not None:
            label = f"▸ {place['name']}   Q/E rotation · clic gauche : pose · Esc : annule"
            draw_hud_surface(
                make_panel(label, font_main, bg=(120, 80, 20, 200)),
                (0, 12), anchor="topcenter", label="placement")

        if pygame.time.get_ticks() < flash["until"]:
            draw_hud_surface(
                make_panel(flash["text"], font_big, padding=(18, 14),
                           bg=(20, 25, 50, 200), fg=(255, 240, 200)),
                (0, 60), anchor="topcenter", label="flash")

        pygame.display.flip()

        if pygame.time.get_ticks() % 500 < 17:
            pygame.display.set_caption(
                f"Jeu de la Vie — GPU   ·   {rule_name}   ·   "
                f"{'PAUSE' if paused else 'PLAY'}"
            )

    pygame.quit()


if __name__ == "__main__":
    main()
