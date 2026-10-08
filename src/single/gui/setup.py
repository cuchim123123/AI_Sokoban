"""Setup screen: map list, algorithm choice, start/back buttons."""


import os



from src.shared.common import (
    C_ACCENT,
    Button,
)





class SetupMixin:
    """Setup screen: map list, algorithm choice, start/back buttons."""

    # ── Setup screen ─────────────────────────────────────────────────────

    def _setup_menu_buttons(self):
        self.menu_buttons = []

        start_y = 170
        for m in self.available_maps:
            name = os.path.basename(m)
            b = Button(70, start_y, 260, 48, name, lambda m=m: self._select_map(m))
            self.menu_buttons.append(b)
            start_y += 60

        # Algorithm choice (one row: label + toggles)
        self.btn_astar = Button(575, 452, 145, 48, "A*",
                                lambda: self._set_algorithm("A*"), accent=C_ACCENT)
        self.btn_ucs = Button(740, 452, 145, 48, "UCS",
                              lambda: self._set_algorithm("UCS"), accent=C_ACCENT)

        self.btn_back = Button(70, 640, 125, 48, "< BACK", self._back_to_launcher)
        self.btn_start = Button(770, 640, 170, 48, "START", self._start_game,
                                selected=True, accent=(120, 220, 120))

        self.menu_buttons.extend([self.btn_astar, self.btn_ucs,
                                  self.btn_back, self.btn_start])
        self._refresh_menu_selection()

    def _refresh_menu_selection(self):
        for b in self.menu_buttons:
            if b.text.endswith(".txt"):
                b.selected = (os.path.basename(b.text) == os.path.basename(self.map_file))
        self.btn_astar.selected = (self.algorithm == "A*")
        self.btn_ucs.selected = (self.algorithm == "UCS")

    def _select_map(self, m):
        self.map_file = m
        self._refresh_menu_selection()

    def _set_algorithm(self, algo):
        self.algorithm = algo
        self._refresh_menu_selection()

    def _back_to_launcher(self):
        self._exit_requested = True

