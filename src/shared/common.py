"""
Shared UI toolkit used by both game modes (single-player puzzle and
competitive).  Everything visual that is not specific to one mode lives
here so the two modes look and feel identical.
"""

import os
import json
import pygame


# ── Config ──
TILE = 64
UI_H = 120

# ── Colors ──
C_BG       = (30, 30, 30)
C_WALL     = (100, 100, 100)
C_FLOOR    = (40, 40, 40)
C_GOAL     = (60, 160, 60)
C_BOX      = (200, 180, 140)
C_BOX_DONE = (140, 200, 140)

C_UI_BG    = (20, 20, 20)
C_TEXT     = (255, 255, 255)
C_TEXT_DIM = (180, 180, 180)

C_ACCENT   = (120, 200, 120)   # used for "selected"/primary controls


def load_placeholder_image(path, size, color, label, tint=None):
    """Load an image scaled to size, or fall back to a labeled color tile."""
    try:
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.scale(img, (size, size))
        if tint:
            tint_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            tint_surf.fill((*tint, 255))
            img.blit(tint_surf, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        return img
    except Exception:
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(surf, color, (0, 0, size, size), border_radius=8)
        font_name = "Segoe UI" if pygame.font.match_font("segoeui") else "Arial"
        font = pygame.font.SysFont(font_name, max(10, size // 5), bold=True)
        txt = font.render(label, True, (255, 255, 255))
        surf.blit(txt, txt.get_rect(center=(size // 2, size // 2)))
        return surf


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_pos(p1, p2, t):
    return (lerp(p1[0], p2[0], t), lerp(p1[1], p2[1], t))


def make_fonts():
    """Standard font set shared by every screen."""
    font_name = "Segoe UI" if pygame.font.match_font("segoeui") else "Arial"
    return {
        "title": pygame.font.SysFont(font_name, 48, bold=True),
        "huge":  pygame.font.SysFont(font_name, 72, bold=True),
        "lg":    pygame.font.SysFont(font_name, 28, bold=True),
        "md":    pygame.font.SysFont(font_name, 22),
        "sm":    pygame.font.SysFont(font_name, 16),
    }


def draw_glass_panel(surface, rect, alpha=40, border_alpha=120, radius=15):
    temp = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(temp, (255, 255, 255, alpha), temp.get_rect(), border_radius=radius)
    pygame.draw.rect(temp, (255, 255, 255, border_alpha), temp.get_rect(), width=1, border_radius=radius)
    surface.blit(temp, rect.topleft)


def draw_glow(surface, rect, color=(255, 255, 255), alpha=80, grow=8, radius=12):
    """Soft glowing border behind a widget (SRCALPHA so alpha really applies)."""
    temp = pygame.Surface((rect.width + grow * 2, rect.height + grow * 2), pygame.SRCALPHA)
    pygame.draw.rect(temp, (*color, alpha), temp.get_rect(), border_radius=radius)
    surface.blit(temp, (rect.x - grow, rect.y - grow))


def create_gradient_surface(w, h, c1, c2):
    surf = pygame.Surface((w, h))
    for y in range(h):
        t = y / max(1, h - 1)
        c = (int(lerp(c1[0], c2[0], t)),
             int(lerp(c1[1], c2[1], t)),
             int(lerp(c1[2], c2[2], t)))
        pygame.draw.line(surf, c, (0, y), (w, y))
    return surf


def load_blurred_image(path, w, h):
    """Heavily downscale + upscale an image so it works as a soft background."""
    try:
        raw = pygame.image.load(path).convert()
        small = pygame.transform.scale(raw, (max(1, raw.get_width() // 16),
                                             max(1, raw.get_height() // 16)))
        return pygame.transform.smoothscale(small, (w, h))
    except Exception:
        return None


class Button:
    def __init__(self, x, y, w, h, text, action, selected=False, accent=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action = action
        self.selected = selected
        self.accent = accent          # (r, g, b) border/text accent or None
        font_name = "Segoe UI" if pygame.font.match_font("segoeui") else "Arial"
        self.font = pygame.font.SysFont(font_name, 20, bold=True)
        self.hovered = False

    def draw(self, screen):
        if self.selected:
            draw_glass_panel(screen, self.rect, alpha=70, border_alpha=200, radius=10)
            if self.accent:
                pygame.draw.rect(screen, self.accent, self.rect, width=2, border_radius=10)
        else:
            alpha = 60 if self.hovered else 25
            draw_glass_panel(screen, self.rect, alpha=alpha, border_alpha=150, radius=10)

        color = self.accent if (self.accent and self.selected) else (255, 255, 255)
        txt_shadow = self.font.render(self.text, True, (0, 0, 0))
        txt_surf = self.font.render(self.text, True, color)

        c = self.rect.center
        screen.blit(txt_shadow, txt_shadow.get_rect(center=(c[0] + 1, c[1] + 1)))
        screen.blit(txt_surf, txt_surf.get_rect(center=c))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered and self.action:
                self.action()


class Animator:
    """Businessman sprite-sheet animation player (shared by both modes)."""

    def __init__(self, base_dir):
        self.states = ["Running", "Push", "Die", "Uppercut", "Idle"]
        self.animations = {}
        self.frame_dims = {}
        # 1: South, 3: West, 5: North, 7: East
        self.dir_map = {(0, 1): 1, (-1, 0): 3, (0, -1): 5, (1, 0): 7}

        for state in self.states:
            self.animations[state] = {}
            self.frame_dims[state] = {}
            base_path = f"{base_dir}/{state}/Businessman_{state}"
            for vec, idx in self.dir_map.items():
                img_path = f"{base_path}_dir{idx}.png"
                json_path = f"{base_path}_dir{idx}.json"
                if os.path.exists(img_path) and os.path.exists(json_path):
                    try:
                        sheet = pygame.image.load(img_path).convert_alpha()
                        with open(json_path, 'r') as f:
                            data = json.load(f)

                        # 1. Find union bounding box over all frames
                        union_rect = None
                        for f_data in data['frames']:
                            r = f_data['frame']
                            rect = pygame.Rect(r['x'], r['y'], r['w'], r['h'])
                            sub = sheet.subsurface(rect)
                            bbox = sub.get_bounding_rect()
                            if bbox.width > 0 and bbox.height > 0:
                                if union_rect is None:
                                    union_rect = bbox.copy()
                                else:
                                    union_rect.union_ip(bbox)

                        if union_rect is None:
                            union_rect = pygame.Rect(0, 0, 256, 256)
                        else:
                            union_rect.inflate_ip(4, 4)

                        # 2. Extract and scale cropped frames
                        frames = []
                        scale_factor = (TILE * 0.9) / union_rect.h
                        new_w = int(union_rect.w * scale_factor)
                        new_h = int(union_rect.h * scale_factor)
                        self.frame_dims[state][vec] = (new_w, new_h)

                        for f_data in data['frames']:
                            r = f_data['frame']
                            crop_rect = pygame.Rect(r['x'] + union_rect.x,
                                                    r['y'] + union_rect.y,
                                                    union_rect.w, union_rect.h)
                            frame_surf = sheet.subsurface(crop_rect)
                            frame_surf = pygame.transform.scale(frame_surf, (new_w, new_h))
                            frames.append(frame_surf)
                        self.animations[state][vec] = frames
                    except Exception:
                        self.animations[state][vec] = None
                else:
                    self.animations[state][vec] = None

    def get_frame(self, state, vec, t, time_ms, start_time_ms=0):
        state_anims = self.animations.get(state, {})
        state_dims = self.frame_dims.get(state, {})

        frames = state_anims.get(vec)
        if not frames:
            frames = state_anims.get((0, 1))
        if not frames:
            return None, (TILE, TILE)

        dims = state_dims.get(vec, state_dims.get((0, 1), (TILE, TILE)))

        if state in ["Running", "Push"] and t < 1.0:
            frame_idx = int(t * len(frames)) % len(frames)
            if frame_idx == 0 and t > 0:
                frame_idx = 1
            return frames[frame_idx], dims
        else:
            elapsed = time_ms - start_time_ms
            frame_idx = (elapsed // 62) % len(frames)
            # Clamp end-game animations so they don't loop endlessly
            if state in ["Die", "Uppercut"]:
                if (elapsed // 62) >= len(frames):
                    frame_idx = len(frames) - 1
            return frames[frame_idx], dims


# ── Map preview ──────────────────────────────────────────────────────────

def _preview_data(map_file, mode):
    """Parse a map just enough for a preview.  Returns (walls, goals, boxes,
    agents) where agents is a list of (pos, color)."""
    from src.single.core.parser import parse_map as parse_single
    from src.competitive.parser import parse_competitive_map

    walls, goals, boxes, agents = set(), set(), set(), []
    if mode == "competitive":
        state, board = parse_competitive_map(map_file)
        walls, goals, boxes = set(board.walls), set(board.goals), set(state.boxes)
        agents.append((state.agent_a, (255, 100, 100)))
        agents.append((state.agent_b, (100, 150, 255)))
    else:
        state, board = parse_single(map_file)
        walls, goals, boxes = set(board.walls), set(board.goals), set(state.boxes)
        agents.append((state.agent, (100, 170, 255)))
    return walls, goals, boxes, agents


def draw_map_preview(surface, rect, map_file, mode, fonts=None):
    """Render a scaled-down board inside rect.  Returns the map name, or
    None if the map could not be parsed."""
    draw_glass_panel(surface, rect, alpha=40, border_alpha=80, radius=15)
    try:
        walls, goals, boxes, agents = _preview_data(map_file, mode)
    except Exception:
        if fonts:
            msg = fonts["md"].render("preview unavailable", True, C_TEXT_DIM)
            surface.blit(msg, msg.get_rect(center=rect.center))
        return None

    name = os.path.basename(map_file)

    # Reserve a strip at the bottom for the file name.
    label_h = 24
    board_rect = pygame.Rect(rect.x + 6, rect.y + 6,
                             rect.width - 12, rect.height - 12 - label_h)

    all_cells = walls | goals | boxes | {p for p, _ in agents}
    width = max((x for x, _ in all_cells), default=0) + 1
    height = max((y for _, y in all_cells), default=0) + 1

    tile = max(4, min(board_rect.w // max(1, width), board_rect.h // max(1, height), 24))
    board_w, board_h = tile * width, tile * height
    ox = board_rect.x + (board_rect.w - board_w) // 2
    oy = board_rect.y + (board_rect.h - board_h) // 2

    # Floor
    pygame.draw.rect(surface, (35, 40, 50),
                     pygame.Rect(ox, oy, board_w, board_h), border_radius=3)

    for x in range(width):
        for y in range(height):
            r = pygame.Rect(ox + x * tile, oy + y * tile, tile, tile)
            if (x, y) in walls:
                pygame.draw.rect(surface, C_WALL, r)
                pygame.draw.rect(surface, (130, 130, 130), r, width=1)
            elif (x, y) in goals:
                pygame.draw.rect(surface, (45, 110, 55), r)

    for bx, by in boxes:
        cx, cy = ox + bx * tile + tile // 2, oy + by * tile + tile // 2
        color = C_BOX_DONE if (bx, by) in goals else C_BOX
        pygame.draw.rect(surface, color,
                         pygame.Rect(cx - tile // 3, cy - tile // 3,
                                     max(3, 2 * tile // 3), max(3, 2 * tile // 3)),
                         border_radius=2)

    for (ax, ay), color in agents:
        cx, cy = ox + ax * tile + tile // 2, oy + ay * tile + tile // 2
        pygame.draw.circle(surface, color, (cx, cy), max(3, tile // 3))

    if fonts:
        label = fonts["sm"].render(name, True, C_TEXT)
        surface.blit(label, label.get_rect(center=(rect.centerx,
                                                   rect.bottom - label_h // 2 - 2)))
    return name
