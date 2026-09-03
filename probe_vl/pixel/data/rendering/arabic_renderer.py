"""
PangoCairoArabicTextRenderer  –  bigram-patch-aligned edition
==============================================================

Design contract
---------------
1. **Every new word starts a new patch** – identical to PangoCairoTextRenderer.

2. **Each patch holds at most 2 logical characters** (bigram cadence) – the
   patch layout and all offset calculations are identical in both modes.

3. **Contextual glyph forms are always correct** – the full word is shaped by
   Pango/HarfBuzz before any slicing, so initial / medial / final / isolated
   forms and ligatures (lam-alef, etc.) are always resolved correctly.

4. Two rendering modes, selected by the ``fill_gaps`` constructor argument:

   ``fill_gaps=False`` (default)
       Each slot is blitted to the canvas with a clip rectangle exactly equal
       to the slot's glyph width.  The inter-patch gap (the padding columns
       between the end of the glyphs and the next patch boundary) is left
       blank.  Connections between adjacent character groups are visually
       broken at patch boundaries.  This is the strictest match to the general
       bigram renderer's appearance.

   ``fill_gaps=True``
       Each slot's clip rectangle is **extended rightward** to the next patch
       boundary before blitting.  Because the scratch surface contains the
       fully connected word, the extra columns copied from the scratch contain
       exactly the connecting stroke (the kashida / baseline join) between the
       two character groups.  The patch layout is identical to ``fill_gaps=False``
       — only the visual gap is filled; no offsets or boundaries change.

       Concretely: for a slot whose glyphs end at canvas pixel ``co + w``, the
       clip rectangle is extended to ``next_patch_boundary(co + w)``.  With the
       same blit translation (``canvas_offset - src_x``), the extra canvas
       columns receive the corresponding scratch columns, which contain the
       connection stroke.

RTL pixel geometry
------------------
In a Pango RTL layout rendered at origin x=0, logical character 0 (first in
reading order) is placed at the RIGHT and char N-1 at the LEFT.
``char_x[i]`` = left pixel edge of char i's glyph (decreases with index).

For slot covering logical characters [a, b):
    right_px = total_width     if a == 0   (right edge of char 0)
             = char_x[a-1]    otherwise    (left edge of preceding char)
    left_px  = _last_present_x(char_x, b) (left edge of last char in slot)
    src_x    = left_px
    width    = right_px - left_px

Slots are built in logical order (0 → N-1) and then reversed before blitting
so the last logical characters (leftmost in scratch) land leftmost on canvas,
matching the word-list reversal the caller already applies for RTL.

Unicode blocks treated as "joining script"
------------------------------------------
0600-06FF  Arabic          0750-077F  Arabic Supplement
08A0-08FF  Arabic Ext-A    FB50-FDFF  Arabic Pres. Forms-A
FE70-FEFF  Arabic Pres. Forms-B
0700-074F  Syriac          07C0-07FF  N'Ko     0840-085F  Mandaic
"""

from __future__ import annotations

import logging
import math
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import cairo
import gi

gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo

from .pangocairo_renderer_bigrams_iso_char import PangoCairoTextRenderer
from .rendering_utils import Encoding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Joining-script detection
# ---------------------------------------------------------------------------

_JOINING_SCRIPT_RANGES: List[Tuple[int, int]] = [
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    (0x0700, 0x074F),  # Syriac
    (0x07C0, 0x07FF),  # N'Ko
    (0x0840, 0x085F),  # Mandaic
]


def _has_joining_script(text: str) -> bool:
    """Return True if *text* contains at least one joining-script character."""
    for ch in text:
        cp = ord(ch)
        for lo, hi in _JOINING_SCRIPT_RANGES:
            if lo <= cp <= hi:
                return True
    return False


# ---------------------------------------------------------------------------
# Arabic joining-type: right-joining-only characters
# ---------------------------------------------------------------------------

# Characters with Unicode Joining_Type = R (Right_Joining).
# These letters join ONLY on their right side (toward their logical predecessor),
# meaning they do NOT connect to their logical successor.  No connecting stroke
# (tatweel) should be drawn on the side facing the successor.
#
# Source: Unicode 15.0 ArabicShaping.txt
_RIGHT_JOINING_RANGES: List[Tuple[int, int]] = [
    (0x0622, 0x0625),  # Alef with Madda/Hamza variants
    (0x0627, 0x0627),  # Arabic Letter Alef
    (0x0629, 0x0629),  # Arabic Letter Teh Marbuta
    (0x062F, 0x0632),  # Dal, Thal, Reh, Zain
    (0x0648, 0x0648),  # Arabic Letter Waw
    (0x0649, 0x0649),  # Arabic Letter Alef Maksura
    (0x0671, 0x0673),  # Alef Wasla, Alef with Wavy Hamza variants
    (0x0675, 0x0677),  # High Hamza Alef / Waw variants
    (0x0688, 0x0699),  # Dal / Reh extended variants (Ddal … Rrreh)
    (0x06C4, 0x06CB),  # Waw extended variants
    (0x06CF, 0x06CF),  # Arabic Letter Waw with Dot Above
    (0x06D2, 0x06D3),  # Yeh Barree variants
    (0x06EE, 0x06EF),  # Dal / Reh with Inverted V
    (0x0759, 0x075B),  # Arabic Supplement Dal variants
    (0x076B, 0x076C),  # Arabic Supplement Reh variants
    (0x0771, 0x0771),  # Arabic Supplement Alef Wasla
    (0x0773, 0x0774),  # Arabic Supplement Alef variants
    (0x0778, 0x0779),  # Arabic Supplement Waw variants
]


def _is_right_joining_only(ch: str) -> bool:
    """
    Return True if *ch* has Unicode Joining_Type = R (Right_Joining).

    Right-joining characters connect only toward their logical predecessor
    (the character to their right in visual RTL layout).  They do NOT produce
    a connecting stroke toward their logical successor, so no tatweel should
    be inserted on that side.
    """
    cp = ord(ch)
    for lo, hi in _RIGHT_JOINING_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def _slot_has_connecting_letter(word: str, slot: "_PatchSlot") -> bool:
    """
    Return True if *slot* contains at least one Arabic letter that can
    participate in joining (i.e. can receive a connecting stroke from the
    right side).

    This guards against Arabic-block non-letter characters — punctuation
    (،, ؛, ؟, …), digits, combining marks — that sit in the Arabic Unicode
    block but have no joining behaviour.  ``_has_joining_script`` tests only
    block membership (U+0600–U+06FF etc.) and would incorrectly pass U+060C
    ARABIC COMMA (category ``Po``).

    Any Arabic letter with Unicode category ``Lo`` or ``Lm`` can receive a
    connection from its right side, including right-joining-only (JT=R)
    letters such as ا, د, ر, و — they join on their right side only, but
    that right side IS the side facing the gap.  The only exclusion is the
    rare JT=U (non-joining) letter U+0621 ARABIC LETTER HAMZA, which never
    joins on either side.
    """
    for i in range(slot.char_start, slot.char_end):
        ch = word[i]
        if unicodedata.category(ch) in ("Lo", "Lm") and ord(ch) != 0x0621:
            return True
    return False


def _slot_connects_forward(word: str, next_slot: "_PatchSlot") -> bool:
    """
    Return True if the gap to the LEFT of *next_slot* should be filled with a
    tatweel, based on whether *next_slot*'s leftmost-on-canvas character
    produces a connecting stroke toward the preceding (left) slot.

    After the slot list has been reversed for RTL placement, the leftmost-on-
    canvas character within *next_slot* is the one with the HIGHEST logical
    index in that slot — ``word[next_slot.char_end - 1]``.  In Arabic, the
    connecting stroke across a gap is driven by this character.  Two conditions
    must both be true for a tatweel to be drawn:

    1. The character is an actual **letter** (Unicode category ``Lo`` or ``Lm``).
       Arabic punctuation such as ، (U+060C) and ؛ (U+061B) fall in the Arabic
       Unicode block but have category ``Po`` — they never join anything.
       Digits (``Nd``), combining marks (``Mn``), and format characters (``Cf``)
       are likewise excluded.

    2. The character is **not right-joining-only** (Joining_Type ≠ R).
       JT=R characters (ا, د, ر, و, …) only connect toward their logical
       predecessor (rightward visually); they do not reach leftward across the
       gap toward the preceding slot.

    Examples
    --------
    تقار،  — gap between canvas slot(،) and slot(تق... wait reversed):
        next_slot boundary = ق (Lo, JT=D) → True  ✓
        gap between slot(،) and Arabic slot: current-slot check (in caller) blocks it ✓
    مساء  — gap between canvas slot(اء) and slot(مس):
        next_slot = slot(مس), char_end=2, word[1]=س (Lo, JT=D) → True  ✓
    مرحبا — gap between canvas slot(حب) and slot(مر):
        next_slot = slot(مر), char_end=2, word[1]=ر (Lo, JT=R) → False ✓
    """
    # Leftmost-on-canvas char of next_slot = highest logical index in the slot.
    boundary_char = word[next_slot.char_end - 1]
    # Must be a letter (not punctuation, digit, mark, or format char)
    # and must not be right-joining-only.
    return (unicodedata.category(boundary_char) in ("Lo", "Lm")
            and not _is_right_joining_only(boundary_char))


# ---------------------------------------------------------------------------
# RTL Pango helpers
# ---------------------------------------------------------------------------

def _set_rtl_on_layout(layout: Pango.Layout) -> None:
    """Set RTL base direction on *layout* before set_text is called."""
    layout.set_auto_dir(False)
    layout.get_context().set_base_dir(Pango.Direction.RTL)
    layout.set_alignment(Pango.Alignment.RIGHT)


def _make_rtl_layout(context: cairo.Context, font: Pango.FontDescription) -> Pango.Layout:
    """Return a new RTL Pango layout attached to *context* with *font* loaded."""
    pango_ctx = PangoCairo.create_context(context)
    pango_ctx.set_font_description(font)
    layout = Pango.Layout(pango_ctx)
    _set_rtl_on_layout(layout)
    return layout


# ---------------------------------------------------------------------------
# _PatchSlot
# ---------------------------------------------------------------------------

class _PatchSlot:
    """
    One bigram-sized slot within a shaped Arabic word.

    Attributes
    ----------
    char_start, char_end : int
        Half-open logical character range [char_start, char_end).
        Normally ``char_end == char_start + 2``; wider when a ligature spans
        the boundary.
    src_x : float
        Left pixel edge of this slot's region in the scratch surface.
    width : float
        Pixel width of the region.  ``src_x + width`` = right pixel edge.
    """
    __slots__ = ("char_start", "char_end", "src_x", "width")

    def __init__(self, char_start: int, char_end: int, src_x: float, width: float) -> None:
        self.char_start = char_start
        self.char_end   = char_end
        self.src_x      = src_x
        self.width      = width


# ---------------------------------------------------------------------------
# Slot building
# ---------------------------------------------------------------------------

def _last_present_x(char_x: Dict[int, float], before_idx: int) -> float:
    """
    Return ``char_x[k]`` for the largest ``k < before_idx`` present in
    *char_x*, or ``0.0`` if no such key exists.
    """
    for k in range(before_idx - 1, -1, -1):
        if k in char_x:
            return char_x[k]
    return 0.0


def _build_patch_slots(
    word: str,
    context: cairo.Context,
    font: Pango.FontDescription,
    pixels_per_patch: int,
    rgb: bool,
    pango_scale: int = 1024,
) -> Tuple[cairo.ImageSurface, List[_PatchSlot]]:
    """
    Shape *word* fully on an off-screen surface and partition the result into
    bigram-sized patch slots.

    The scratch surface contains the correctly shaped word at x=0.  Each slot
    records a contiguous, non-overlapping pixel region of that surface; their
    widths sum to ``total_width``.  Slots are in logical order (index 0 = first
    reading-order character = rightmost on screen for RTL).

    See module docstring for the pixel-geometry derivation.
    """
    # ── Shape ────────────────────────────────────────────────────────────
    measure_layout = _make_rtl_layout(context, font)
    measure_layout.set_text(word, -1)
    total_width, total_height = measure_layout.get_pixel_size()

    scratch_w = max(total_width + 4, 1)
    scratch_h = pixels_per_patch
    fmt       = cairo.FORMAT_RGB24 if rgb else cairo.FORMAT_A8
    scratch   = cairo.ImageSurface(fmt, scratch_w, scratch_h)
    sctx      = cairo.Context(scratch)

    if rgb:
        sctx.set_source_rgb(1.0, 1.0, 1.0)
        sctx.rectangle(0, 0, scratch_w, scratch_h)
        sctx.fill()
        sctx.set_source_rgb(0.0, 0.0, 0.0)

    shaped = _make_rtl_layout(sctx, font)
    shaped.set_text(word, -1)
    sctx.move_to(0, scratch_h / 2.0 - total_height / 2.0 - 2)
    PangoCairo.show_layout(sctx, shaped)

    # ── Build char_x ─────────────────────────────────────────────────────
    word_bytes = word.encode("utf-8")
    n_chars    = len(word)
    char_x: Dict[int, float] = {}

    it = shaped.get_iter()
    while True:
        ext      = it.get_cluster_extents()[1]
        x_px     = ext.x / pango_scale
        byte_idx = it.get_index()
        char_idx = len(word_bytes[:byte_idx].decode("utf-8"))
        if char_idx not in char_x:
            char_x[char_idx] = x_px
        if not it.next_cluster():
            break

    # ── Partition into bigram slots ───────────────────────────────────────
    slots: List[_PatchSlot] = []
    logical_idx = 0

    while logical_idx < n_chars:
        slot_end = min(logical_idx + 2, n_chars)

        # Ligature guard: extend slot_end until the boundary character has
        # an independent cluster position (or we reach end of word).
        while slot_end < n_chars and slot_end not in char_x:
            slot_end += 1

        # Right pixel boundary of this slot (= right edge of char[logical_idx]).
        # Use _last_present_x rather than char_x.get(logical_idx - 1, ...) so
        # that ligature-absorbed characters (absent from char_x) are skipped
        # correctly.  e.g. in السلام the لا ligature absorbs char[4]=ا; when
        # the next slot starts at logical_idx=5 (م), char_x[4] is absent and
        # char_x.get(4, total_width) would return total_width (wrong); walking
        # back with _last_present_x correctly returns char_x[3]=14 instead.
        right_px = (
            float(total_width)
            if logical_idx == 0
            else _last_present_x(char_x, logical_idx)
        )

        # Left pixel boundary (= left edge of last char in slot)
        left_px = _last_present_x(char_x, slot_end)

        slots.append(_PatchSlot(
            char_start=logical_idx,
            char_end=slot_end,
            src_x=left_px,
            width=max(0.0, right_px - left_px),
        ))
        logical_idx = slot_end

    return scratch, slots


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PangoCairoArabicTextRenderer(PangoCairoTextRenderer):
    """
    Drop-in replacement for PangoCairoTextRenderer with Arabic-aware rendering.

    Parameters
    ----------
    fill_gaps : bool, optional (default False)
        Controls whether the inter-patch visual gap between adjacent character
        groups within a word is filled with the connecting stroke.

        False
            Each slot is blitted with a clip rectangle equal to its glyph
            width only.  The padding columns between the end of the glyphs and
            the next patch boundary are left blank, producing a visible break
            between connected character groups.

        True
            After blitting each slot, the inter-patch gap that follows it is
            filled with the Arabic tatweel character (ـ, U+0640 ARABIC TATWEEL),
            rendered using the loaded font.  The tatweel is the canonical
            kashida / connecting stroke of the Arabic script; using the actual
            font glyph guarantees that baseline, weight, and style match the
            surrounding letters exactly.  If the gap is wider than one tatweel
            glyph, the glyph is tiled to fill the full gap width.
            Patch positions, offsets, and word_patch_map are **identical** to
            ``fill_gaps=False`` — only the visual gap pixels change.

    All other constructor parameters are identical to PangoCairoTextRenderer.

    Usage
    -----
    >>> # Default: clean bigram boundaries (gaps visible)
    >>> r = PangoCairoArabicTextRenderer(font_file="NotoNaskhArabic.ttf")

    >>> # Filled gaps: visually connected across patch boundaries
    >>> r = PangoCairoArabicTextRenderer(font_file="NotoNaskhArabic.ttf",
    ...                                  fill_gaps=True)

    >>> # Upgrade an existing renderer
    >>> ar = PangoCairoArabicTextRenderer.from_renderer(base, fill_gaps=True)
    """

    def __init__(self, *args, fill_gaps: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fill_gaps: bool = fill_gaps

    # ------------------------------------------------------------------
    # Pickling
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state["fill_gaps"] = self.fill_gaps
        return state

    def __setstate__(self, state_dict: Dict[str, Any]) -> None:
        self.__init__(**state_dict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shape_arabic_word(
        self,
        word: str,
        context: cairo.Context,
    ) -> Tuple[cairo.ImageSurface, List[_PatchSlot]]:
        """Thin wrapper around :func:`_build_patch_slots`."""
        return _build_patch_slots(
            word=word,
            context=context,
            font=self.font,
            pixels_per_patch=self.pixels_per_patch,
            rgb=self.rgb,
            pango_scale=self.PANGO_SCALE,
        )

    def _blit_slot(
        self,
        context: cairo.Context,
        scratch: cairo.ImageSurface,
        slot: "_PatchSlot",
        canvas_offset: int,
    ) -> None:
        """
        Blit one slot's glyph region from *scratch* onto *context*.

        The clip rectangle is exactly ``[canvas_offset, canvas_offset + ceil(width))``,
        so only the glyph pixels are copied — the inter-patch gap is left for
        :meth:`_render_tatweel_gap` to fill when ``fill_gaps=True``.
        """
        slot_w = int(math.ceil(slot.width))
        if slot_w <= 0:
            return
        context.save()
        context.set_source_surface(scratch, canvas_offset - slot.src_x, 0)
        context.rectangle(canvas_offset, 0, slot_w, self.pixels_per_patch)
        context.fill()
        context.restore()

    def _render_tatweel_gap(
        self,
        context: cairo.Context,
        gap_x: int,
        gap_w: int,
    ) -> None:
        """
        Fill the inter-patch gap at ``[gap_x, gap_x + gap_w)`` with a tatweel
        (Arabic kashida, U+0640: ـ).

        The tatweel is the canonical connecting stroke of the Arabic script.
        Using the actual glyph from the loaded font guarantees that the
        baseline, weight, and style match the surrounding letters exactly.

        Strategy
        --------
        A Pango layout containing only ``"ـ"`` is created with RTL direction
        so that Pango/HarfBuzz positions it correctly on the baseline.  The
        layout is rendered into a small scratch surface whose width equals
        *gap_w*, then blitted onto the destination surface at *gap_x*.  Cairo's
        ``rectangle`` clip ensures nothing bleeds outside the gap.

        If the tatweel glyph is narrower than *gap_w*, it is tiled horizontally
        to fill the whole gap.  If the font does not contain a tatweel glyph
        (unknown glyph count > 0) the gap is left blank rather than showing a
        placeholder box.
        """
        if gap_w <= 0:
            return

        # Measure the tatweel glyph
        probe = _make_rtl_layout(context, self.font)
        probe.set_text("\u0640", -1)                    # ـ  ARABIC TATWEEL

        if probe.get_unknown_glyphs_count() > 0:
            # Font has no tatweel; leave the gap blank.
            return

        tw, th = probe.get_pixel_size()
        if tw <= 0:
            return

        # Render tatweel(s) into a gap-width scratch surface
        fmt         = cairo.FORMAT_RGB24 if self.rgb else cairo.FORMAT_A8
        gap_scratch = cairo.ImageSurface(fmt, gap_w, self.pixels_per_patch)
        gctx        = cairo.Context(gap_scratch)

        if self.rgb:
            gctx.set_source_rgb(1.0, 1.0, 1.0)
            gctx.rectangle(0, 0, gap_w, self.pixels_per_patch)
            gctx.fill()
            gctx.set_source_rgb(0.0, 0.0, 0.0)

        y = self.pixels_per_patch / 2.0 - th / 2.0 - 2

        # Tile tatweel glyphs left-to-right until the gap is covered.
        # Each repetition is placed one tatweel-width to the right of the last.
        x = 0
        while x < gap_w:
            t_layout = _make_rtl_layout(gctx, self.font)
            t_layout.set_text("\u0640", -1)
            gctx.move_to(x, y)
            PangoCairo.show_layout(gctx, t_layout)
            x += tw

        # Blit gap_scratch onto the destination, clipped to the gap region
        context.save()
        context.set_source_surface(gap_scratch, gap_x, 0)
        context.rectangle(gap_x, 0, gap_w, self.pixels_per_patch)
        context.fill()
        context.restore()

    # ------------------------------------------------------------------
    # _render_bigram  /  _offset_bigram  (overrides parent)
    # ------------------------------------------------------------------

    def _render_bigram(
        self,
        word: str,
        offset: int,
        context: cairo.Context,
        is_last: bool = False,
    ) -> Tuple[cairo.Context, Pango.Layout, int]:
        """
        Render one token.

        Non-joining text is delegated to the parent.  For joining-script tokens:

        1. The full word is shaped on a scratch surface (correct forms +
           ligatures).
        2. Slots are reversed (RTL: last logical chars → leftmost canvas).
        3. Each slot is blitted at its patch-aligned canvas offset using a clip
           rectangle equal to the slot's glyph width only.
        4. When ``fill_gaps=True`` and the slot is not the last in the word, a
           tatweel (ـ, U+0640) is rendered into the inter-patch gap that follows
           the slot, using the font's own kashida glyph tiled to fit the gap
           width.  This visually connects adjacent character groups across the
           patch boundary without any pixel-level bleed from the scratch surface.
        5. Patch positions and offsets are **identical** in both modes.
        """
        if not _has_joining_script(word):
            if len(word) > 2:
                return super()._render_single_word(word, offset, context, is_last)
            return super()._render_bigram(word, offset, context, is_last)

        scratch, slots = self._shape_arabic_word(word, context)

        placeholder = _make_rtl_layout(context, self.font)
        placeholder.set_text(word, -1)

        # Reverse: last logical chars (leftmost in scratch) → leftmost on canvas
        slots = slots[::-1]

        n_slots = len(slots)
        for i, slot in enumerate(slots):
            is_final_slot_of_word = (i == n_slots - 1)  # last slot of THIS word
            last_slot = is_final_slot_of_word and is_last  # last slot of whole sentence

            slot_w = int(math.ceil(slot.width))

            if slot.width <= 0:
                if last_slot:
                    offset += 2
                offset = self._get_unbounded_offset_to_next_patch(offset)
                continue

            # Blit the glyph region (exact width, no bleed)
            self._blit_slot(context, scratch, slot, offset)

            # Advance offset to next patch boundary
            if last_slot:
                offset += 2
            next_offset = self._get_unbounded_offset_to_next_patch(offset + slot_w)

            # Fill the intra-word inter-patch gap with a tatweel when requested.
            # Conditions for filling (ALL must be true):
            #   (1) fill_gaps mode is enabled
            #   (2) not the final slot of THIS word — the gap after the last slot
            #       of a word is an inter-word gap and must never be filled,
            #       regardless of whether this word is the last in the sentence.
            #   (3) the CURRENT slot contains joining-script characters — if the
            #       left side of the gap is punctuation or Latin (e.g. the "."
            #       in "القرآن." forms its own slot), no tatweel is drawn.
            #   (4) the character at the right visual edge of the NEXT slot
            #       connects leftward — i.e. it has Joining_Type=D (dual), not R.
            #       Right-joining-only chars (ا, د, ر, و, …) do not produce a
            #       connecting stroke toward the preceding slot.
            if (self.fill_gaps
                    and not is_final_slot_of_word
                    and _slot_has_connecting_letter(word, slot)
                    and _slot_connects_forward(word, slots[i + 1])):
                gap_x = offset + slot_w
                gap_w = next_offset - gap_x
                if gap_w > 0:
                    self._render_tatweel_gap(context, gap_x, gap_w)

            offset = next_offset

        return context, placeholder, offset

    def _offset_bigram(
        self,
        word: str,
        offset: int,
        context: cairo.Context,
        is_last: bool = False,
    ) -> int:
        """
        Measure one token without painting.

        Offset accumulation is identical regardless of ``fill_gaps`` because
        ``fill_gaps`` only affects which pixels are written, not where patches
        are placed.
        """
        if not _has_joining_script(word):
            if len(word) > 2:
                return super()._offset_single_word(word, offset, context, is_last)
            return super()._offset_bigram(word, offset, context, is_last)

        _, slots = self._shape_arabic_word(word, context)
        slots = slots[::-1]

        for i, slot in enumerate(slots):
            last_slot = (i == len(slots) - 1) and is_last
            if slot.width <= 0:
                if last_slot:
                    offset += 2
                offset = self._get_unbounded_offset_to_next_patch(offset)
                continue
            if last_slot:
                offset += 2
            offset = self._get_unbounded_offset_to_next_patch(
                offset + int(math.ceil(slot.width))
            )

        return offset

    # ------------------------------------------------------------------
    # _render_single_sentence  (override for Arabic-aware n-gram splitter)
    # ------------------------------------------------------------------

    def _render_single_sentence(
        self,
        words: List[str],
        context: cairo.Context,
        input_offset: int,
        rtl: bool = False,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        """
        Render *words* with an Arabic-aware n-gram splitter.

        For joining-script tokens ``string_to_ngrams`` returns the whole token
        as a single element; the bigram partitioning happens inside
        ``_render_bigram`` after Pango has shaped the full word.
        All other logic is identical to the parent.
        """

        def string_to_ngrams(s: str, n: int = 2, rtl: bool = False) -> List[str]:
            if _has_joining_script(s):
                return [s]   # do not split; do not reverse (Pango handles RTL)
            if rtl:
                return [s[i : i + n][::-1] for i in range(0, len(s), n)][::-1]
            return [s[i : i + n] for i in range(0, len(s), n)]

        if len(words) == 1 and words[0].startswith("http://www.google.com/search"):
            logger.debug("UD Treebanks edge-case URL; clamping.")
            words = ["http://www.google.com/search"]

        word_patch_map: Dict[str, Any] = {
            "sum_len_words":              sum(len(f"{w} ") for w in words) - len(" "),
            "total_number_of_words":      len(words),
            "word_order_on_canvas":       {},
            "text_is_truncated":          False,
            "word_list_truncation_index": len(words) - 1,
            "is_rtl":                     rtl,
        }

        word_counter       = 0
        max_length         = self.max_seq_length if max_length is None else max_length
        offset             = input_offset
        skip_last          = False
        word_start_indices = [math.ceil(offset / self.pixels_per_patch)]
        sentence_len_ltr   = 0
        sentence_len_rtl   = word_patch_map["sum_len_words"]

        # Measure first word
        word_len = len(words[0]) if rtl else len(f"{words[0]} ")
        for chunk in string_to_ngrams(words[0], rtl=rtl):
            offset = self._offset_bigram(chunk, offset, context)
        sentence_len_ltr += word_len
        word_patch_map["word_order_on_canvas"][word_counter] = {
            "start_patch_index": word_start_indices[-1],
            "word":              words[0],
            "word_len":          word_len,
            "sentence_len_ltr":  sentence_len_ltr,
            "sentence_len_rtl":  sentence_len_rtl,
        }
        word_counter     += 1
        sentence_len_rtl -= word_len
        word_start_indices.append(math.ceil(offset / self.pixels_per_patch))

        # Measure middle words
        for word in words[1:-1]:
            word_len          = len(f"{word} ")
            sentence_len_ltr += word_len
            word_patch_map["word_order_on_canvas"][word_counter] = {
                "start_patch_index": word_start_indices[-1],
                "word":              word,
                "word_len":          word_len,
                "sentence_len_ltr":  sentence_len_ltr,
                "sentence_len_rtl":  sentence_len_rtl,
            }
            word_counter     += 1
            sentence_len_rtl -= word_len
            for chunk in string_to_ngrams(word, rtl=rtl):
                offset = self._offset_bigram(chunk, offset, context)
            word_start_indices.append(math.ceil(offset / self.pixels_per_patch))

        # Measure last word
        if len(words) > 1:
            word_len = len(f"{words[-1]} ") if rtl else len(words[-1])
            for chunk in string_to_ngrams(words[-1], rtl=rtl):
                offset = self._offset_bigram(chunk, offset, context)
            sentence_len_ltr += word_len
            word_patch_map["word_order_on_canvas"][word_counter] = {
                "start_patch_index": word_start_indices[-1],
                "word":              words[-1],
                "word_len":          word_len,
                "sentence_len_ltr":  sentence_len_ltr,
                "sentence_len_rtl":  sentence_len_rtl,
            }
            word_counter     += 1
            sentence_len_rtl -= word_len
            word_start_indices.append(math.ceil(offset / self.pixels_per_patch))

        word_patch_map["first_patch_after_last_word"] = word_start_indices[-1]

        # Truncation search
        truncated_word_patch_map = word_patch_map.copy()
        index                = word_patch_map["total_number_of_words"] - 1
        patches              = word_patch_map["first_patch_after_last_word"]
        word_order_on_canvas = word_patch_map["word_order_on_canvas"]

        if max_length is not None and patches > max_length:
            truncated_word_patch_map["text_is_truncated"] = True
            while patches > max_length:
                index  -= 1
                patches = (
                    max_length if index < 0
                    else word_order_on_canvas[index]["start_patch_index"]
                )
            index -= 1
            if index < 0:
                logger.warning(
                    "First word in %s does not fit in max_length=%d; "
                    "returning canvas with only the first word.",
                    words, max_length,
                )
                index = 0

        # Render loop
        offset = input_offset
        layout = None

        for word in words[:index]:
            for chunk in string_to_ngrams(word, rtl=rtl):
                context, layout, offset = self._render_bigram(chunk, offset, context)
            if offset == self.max_pixels_len - self.pixels_per_patch:
                skip_last = True
                break

        if not skip_last:
            for chunk in string_to_ngrams(words[index], rtl=rtl):
                context, layout, offset = self._render_bigram(chunk, offset, context)

        if layout is None:
            layout = PangoCairo.create_layout(context)
            layout.set_font_description(self.font)

        kept = list(range(index + 1))
        truncated_word_patch_map["word_order_on_canvas"] = {
            k: v for k, v in word_patch_map["word_order_on_canvas"].items()
            if k in kept
        }
        truncated_word_patch_map["sum_len_words"] = (
            sum(len(f"{w} ") for w in words[: index + 1]) - len(" ")
        )
        truncated_word_patch_map["total_number_of_words"]      = len(words[: index + 1])
        truncated_word_patch_map["word_list_truncation_index"] = index
        truncated_word_patch_map["first_patch_after_last_word"] = math.ceil(
            offset / self.pixels_per_patch
        )

        return context, layout, offset, (word_patch_map, truncated_word_patch_map)

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_renderer(
        cls,
        renderer: "PangoCairoTextRenderer",
        fill_gaps: bool = False,
    ) -> "PangoCairoArabicTextRenderer":
        """
        Build a PangoCairoArabicTextRenderer from an existing renderer.

        Parameters
        ----------
        renderer  : source PangoCairoTextRenderer to copy settings from
        fill_gaps : passed to the new instance

        Example
        -------
        >>> base = PangoCairoTextRenderer(font_file="NotoNaskhArabic.ttf")
        >>> ar   = PangoCairoArabicTextRenderer.from_renderer(base, fill_gaps=True)
        """
        state = renderer.__getstate__()
        state.pop("__class__", None)
        return cls(**state, fill_gaps=fill_gaps)