"""
Unified Sokoban launcher — the game's main menu.

Two modes share one shell:
  • 1 Player  — single-agent puzzle solved by A* or UCS (src/single/gui/single.py)
  • 2 Players — competitive mode (src/competitive/gui/app.py)

Both modes use the same visual style; the launcher dispatches to whichever
mode the player picks and loops until the window is closed.
"""

import sys

import pygame

from src.shared.common import make_fonts, draw_glow, C_MENU_BG

MENU_W = 1024
MENU_H = 768

C_PANEL_DARK = (14, 18, 32)        # card container (idle)
C_PANEL_LIT = (26, 32, 52)         # card container (hovered)
C_INK = (8, 12, 24)                # dark text on cyan

C_CARD_ACCENT_SINGLE = (120, 200, 255)
C_CARD_ACCENT_COMP   = (255, 160, 120)


class ModeCard:
    """A large clickable mode card on the main menu."""

    def __init__(self, rect, kicker, title, lines, accent, action):
        self.rect = rect
        self.kicker = kicker      # small label above the title
        self.title = title        # big title
        self.lines = lines        # description lines
        self.accent = accent
        self.action = action
        self.hovered = False

    def draw(self, screen, fonts):
        hovered = self.hovered
        fill = C_PANEL_LIT if hovered else C_PANEL_DARK

        if hovered:
            draw_glow(screen, self.rect, color=self.accent, alpha=110, grow=6,
                      radius=18)
        # Solid, high-contrast container + strong border
        pygame.draw.rect(screen, fill, self.rect, border_radius=18)
        pygame.draw.rect(screen, self.accent if hovered else (0, 0, 0),
                         self.rect, 4 if hovered else 3, border_radius=18)
        # Accent bar across the top of the card
        bar = pygame.Rect(self.rect.x + 4, self.rect.y + 4,
                          self.rect.width - 8, 5)
        pygame.draw.rect(screen, self.accent, bar)

        cx = self.rect.centerx

        kicker = fonts["sm"].render(self.kicker, True, self.accent)
        screen.blit(kicker, kicker.get_rect(center=(cx, self.rect.y + 46)))

        title = fonts["title"].render(self.title, True, (255, 255, 255))
        screen.blit(title, title.get_rect(center=(cx, self.rect.y + 96)))

        y = self.rect.y + 150
        for line in self.lines:
            txt = fonts["md"].render(line, True, (224, 224, 224))
            screen.blit(txt, txt.get_rect(center=(cx, y)))
            y += 32

        hint = fonts["sm"].render("CLICK TO PLAY", True,
                                  (255, 255, 255) if hovered else (185, 185, 185))
        screen.blit(hint, hint.get_rect(center=(cx, self.rect.bottom - 34)))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered and self.action:
                self.action()


class App:
    """Root launcher: main menu -> mode app -> back to main menu."""

    def __init__(self, map_file=None):
        pygame.init()
        pygame.display.set_caption("Sokoban")
        self.screen = pygame.display.set_mode((MENU_W, MENU_H))
        self.clock = pygame.time.Clock()
        self.fonts = make_fonts()

        self.preselect_map = map_file
        self.single = None
        self.competitive = None
        self._choice = None

        card_w, card_h, gap = 400, 340, 60
        x1 = (MENU_W - (card_w * 2 + gap)) // 2
        y = 250
        self.cards = [
            ModeCard(
                pygame.Rect(x1, y, card_w, card_h),
                "1 PLAYER",
                "PUZZLE MODE",
                ["Push every crate onto a goal.",
                 "The AI solves it for you",
                 "with A* or uniform-cost search."],
                C_CARD_ACCENT_SINGLE,
                lambda: self._select("single"),
            ),
            ModeCard(
                pygame.Rect(x1 + card_w + gap, y, card_w, card_h),
                "2 PLAYERS",
                "COMPETITIVE",
                ["Agent A vs Agent B.",
                 "Race for goals within",
                 "the step limit — good luck!"],
                C_CARD_ACCENT_COMP,
                lambda: self._select("competitive"),
            ),
        ]

    def _select(self, choice):
        self._choice = choice

    def run(self):
        while True:
            choice = self._run_menu()
            if choice is None:
                break
            if choice == "single":
                self._run_single()
            else:
                self._run_competitive()
        pygame.quit()

    # ── Main menu ────────────────────────────────────────────────────────

    def _run_menu(self):
        pygame.display.set_caption("Sokoban")
        self.screen = pygame.display.set_mode((MENU_W, MENU_H))
        self._choice = None

        while self._choice is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_1:
                        self._select("single")
                    elif event.key == pygame.K_2:
                        self._select("competitive")
                    elif event.key == pygame.K_RETURN:
                        self._select("single")
                for card in self.cards:
                    card.handle_event(event)

            self._draw_menu()
            self.clock.tick(60)

        return self._choice

    def _draw_menu(self):
        # Full solid cyan background (no image, no overlay)
        self.screen.fill(C_MENU_BG)

        # Title — black on cyan with a white drop-shadow for punch
        fonts = self.fonts
        title = fonts["huge"].render("SOKOBAN", True, (0, 0, 0))
        shadow = fonts["huge"].render("SOKOBAN", True, (255, 255, 255))
        self.screen.blit(shadow, shadow.get_rect(center=(MENU_W // 2 + 2, 107)))
        self.screen.blit(title, title.get_rect(center=(MENU_W // 2, 105)))

        subtitle = fonts["md"].render("AI SEARCH GAME  —  SELECT GAME MODE",
                                      True, C_INK)
        self.screen.blit(subtitle, subtitle.get_rect(center=(MENU_W // 2, 165)))

        # Divider — hard black rule for contrast
        pygame.draw.line(self.screen, (0, 0, 0),
                         (MENU_W // 2 - 220, 200), (MENU_W // 2 + 220, 200), 3)

        for card in self.cards:
            card.draw(self.screen, fonts)

        footer = fonts["sm"].render(
            "Click a mode  •  1 / 2 keys  •  ESC to quit", True, C_INK)
        self.screen.blit(footer, footer.get_rect(center=(MENU_W // 2, MENU_H - 45)))

        pygame.display.flip()

    # ── Dispatch ─────────────────────────────────────────────────────────

    def _run_single(self):
        from src.single.gui.single import SinglePlayerApp
        if self.single is None:
            self.single = SinglePlayerApp(self.preselect_map)
        pygame.display.set_caption("Sokoban — Puzzle Mode")
        self.single.run()

    def _run_competitive(self):
        from src.competitive.gui.app import CompetitiveApp
        if self.competitive is None:
            self.competitive = CompetitiveApp(
                "maps/competitive/arena_open.txt", 50, "AI", "AI",
                launcher=True)
        pygame.display.set_caption("Sokoban — Competitive Mode")
        self.competitive.run()


if __name__ == "__main__":
    App().run()
