"""
PangoCairoArabicTextRenderer
============================
A specialised subclass of PangoCairoTextRenderer with first-class support for
Arabic and other contextually-joining scripts (Syriac, N'Ko, Mandaic, …).

Rendering modes (controlled by ``bigram_align``)
-------------------------------------------------
``bigram_align=False``  (default)
    Each Arabic word is passed to Pango as one complete layout unit.  Pango
    resolves all glyph forms (initial / medial / final / isolated), ligatures
    (lam-alef, etc.), and diacritic placement in one pass.  The word occupies
    however many pixels Pango says it needs; patch boundaries fall naturally
    after the word.  This matches the behaviour of the original
    PangoCairoTextRenderer for Latin text.

``bigram_align=True``
    Spatial layout is aligned to the parent's bigram grid *but* glyph forms
    are still resolved from the full word context.  This is achieved with a
    two-pass strategy:

    1. **Form-resolution pass** – the full word is rendered into a temporary
       off-screen Cairo surface so that Pango resolves correct glyph shapes
       (connected medial كـ, initial forms, lam-alef ligatures, etc.) through
       its normal shaping pipeline.

    2. **Bigram-placement pass** – the shaped glyphs are split into consecutive
       2-character (logical) groups.  Each group's pixel extent is read from
       Pango's glyph-run iterator, and the *pre-shaped* surface region is
       copied (blitted) to the real output surface at the bigram-aligned
       offset, exactly as the parent would place a Latin bigram.

    Result: the canvas is laid out with the same patch cadence as the parent
    renderer, yet every Arabic letter still carries its contextually-correct
    connected form.  Ligatures that span a bigram boundary are kept together
    in the *earlier* bigram slot.

Why ligatures must stay intact across bigram boundaries
-------------------------------------------------------
A lam-alef ligature (لا) is a *single glyph* in the font.  If we tried to
split it at the bigram boundary we would have to render incomplete half-glyphs,
which is not possible with a standard raster font pipeline.  The approach taken
here is: detect when a ligature's logical character range crosses a 2-character
boundary and allocate the whole ligature to the earlier slot; the next slot
begins after the ligature.

Unicode blocks treated as "joining script"
------------------------------------------
Range           Block name
0600–06FF       Arabic
0750–077F       Arabic Supplement
08A0–08FF       Arabic Extended-A
FB50–FDFF       Arabic Presentation Forms-A
FE70–FEFF       Arabic Presentation Forms-B
0700–074F       Syriac
07C0–07FF       N'Ko
0840–085F       Mandaic
"""

from __future__ import annotations

import logging
import math
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
# Module-level constants and helpers
# ---------------------------------------------------------------------------

_JOINING_SCRIPT_RANGES: List[Tuple[int, int]] = [
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
    (0x0700, 0x074F),   # Syriac
    (0x07C0, 0x07FF),   # N'Ko
    (0x0840, 0x085F),   # Mandaic
]


def _has_joining_script(text: str) -> bool:
    """
    Return True if *text* contains at least one character from a Unicode block
    that uses contextual joining (Arabic, Syriac, N'Ko, …).

    These scripts MUST be passed to Pango as a complete character run so that
    initial / medial / final / isolated glyph substitutions are applied.
    Splitting them into sub-word chunks causes every letter to render in its
    isolated form, breaking joins and suppressing ligatures.
    """
    for ch in text:
        cp = ord(ch)
        for lo, hi in _JOINING_SCRIPT_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def _set_rtl_on_layout(layout: Pango.Layout) -> None:
    """
    Set RTL base direction on *layout* **before** calling set_text.

    Pango performs bidi itemisation at set_text time, so the paragraph
    direction must be fixed beforehand.  Calling this afterwards has no effect.
    """
    layout.set_auto_dir(False)
    ctx = layout.get_context()
    ctx.set_base_dir(Pango.Direction.RTL)
    layout.set_alignment(Pango.Alignment.RIGHT)


def _make_rtl_layout(context: cairo.Context, font: Pango.FontDescription) -> Pango.Layout:
    """
    Create a Pango layout attached to *context* with RTL base direction and
    *font* pre-loaded, ready for set_text.
    """
    pango_ctx = PangoCairo.create_context(context)
    pango_ctx.set_font_description(font)
    layout = Pango.Layout(pango_ctx)
    _set_rtl_on_layout(layout)
    return layout


# ---------------------------------------------------------------------------
# _BigramSlot: one bigram-aligned rendering slot
# ---------------------------------------------------------------------------

class _BigramSlot:
    """
    Describes one bigram-aligned slot in a shaped Arabic word.

    Attributes
    ----------
    char_start, char_end : int
        Half-open range [char_start, char_end) of *logical* Unicode character
        indices that this slot covers.  For a normal 2-char bigram
        char_end = char_start + 2.  For a ligature that spans a boundary,
        char_end may cover additional characters.
    src_x : float
        X-coordinate (pixels) in the shaped off-screen surface where this
        slot's glyphs begin.
    width : float
        Pixel width of the glyphs belonging to this slot.
    """
    __slots__ = ("char_start", "char_end", "src_x", "width")

    def __init__(self, char_start: int, char_end: int, src_x: float, width: float):
        self.char_start = char_start
        self.char_end   = char_end
        self.src_x      = src_x
        self.width      = width


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PangoCairoArabicTextRenderer(PangoCairoTextRenderer):
    """
    Drop-in replacement for PangoCairoTextRenderer with Arabic-aware rendering.

    Parameters
    ----------
    bigram_align : bool, optional (default False)
        Controls the spatial layout mode for joining-script words.

        False
            Whole-word layout.  Each Arabic word occupies its natural pixel
            width as measured by Pango.  Patch boundaries fall after the word.
            Glyph forms and ligatures are always correct.

        True
            Bigram-aligned layout.  The canvas cadence matches the parent
            renderer (each 2-character chunk aligns to a patch boundary) but
            glyph forms are still resolved from the full word context via a
            two-pass render-then-blit strategy.  Ligatures that span a bigram
            boundary are kept whole in the earlier slot.

    All other parameters are identical to PangoCairoTextRenderer.

    Usage
    -----
    >>> # Whole-word mode (default) – best fidelity
    >>> r = PangoCairoArabicTextRenderer(font_file="NotoNaskhArabic.ttf")

    >>> # Bigram-aligned mode – matches parent spatial rhythm
    >>> r = PangoCairoArabicTextRenderer(font_file="NotoNaskhArabic.ttf",
    ...                                  bigram_align=True)

    >>> # Upgrade an existing renderer
    >>> r = PangoCairoArabicTextRenderer.from_renderer(base, bigram_align=True)
    """

    def __init__(self, *args, bigram_align: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.bigram_align: bool = bigram_align

    # ------------------------------------------------------------------
    # Pickling – must round-trip bigram_align
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state["bigram_align"] = self.bigram_align
        return state

    def __setstate__(self, state_dict: Dict[str, Any]) -> None:
        self.__init__(**state_dict)

    # ------------------------------------------------------------------
    # Whole-word path  (bigram_align=False)
    # ------------------------------------------------------------------

    def _render_single_word_arabic(
        self,
        word: str,
        offset: int,
        context: cairo.Context,
        is_last: bool = False,
        unbounded: bool = True,
    ) -> Tuple[cairo.Context, Pango.Layout, int]:
        """
        Render *word* as one Pango layout unit with RTL base direction.

        The RTL direction is set before set_text so that Pango's bidi engine
        itemises the run with the correct paragraph direction and HarfBuzz sees
        the full character context for glyph-form selection and ligature
        formation.

        Parameters
        ----------
        word      : token to render (may include diacritics, ZWJ/ZWNJ)
        offset    : leftmost pixel position on the destination surface
        context   : active Cairo drawing context
        is_last   : if True, 2 px end-of-sequence padding is added
        unbounded : if True the returned offset may exceed max_pixels_len
                    (measure pass); if False it is clamped (bounded pass).
        """
        layout = _make_rtl_layout(context, self.font)
        layout.set_text(word, -1)

        if layout.get_unknown_glyphs_count() > 0:
            logger.warning(
                "Found %d unknown glyphs in Arabic word %r (%s). "
                "Consider loading a fallback font.",
                layout.get_unknown_glyphs_count(),
                word,
                [hex(ord(c)) for c in word],
            )

        width, height = layout.get_pixel_size()
        y = self.pixels_per_patch / 2.0 - height / 2.0 - 2
        context.move_to(offset, y)
        PangoCairo.show_layout(context, layout)

        if is_last:
            offset += 2
        if unbounded:
            offset = self._get_unbounded_offset_to_next_patch(offset + width)
        else:
            offset = self._get_offset_to_next_patch(offset + width)
        return context, layout, offset

    def _offset_single_word_arabic(
        self,
        word: str,
        offset: int,
        context: cairo.Context,
        is_last: bool = False,
    ) -> int:
        """
        Measure the rendered width of *word* with RTL base direction without
        painting anything.  Returns the updated patch-aligned offset.
        """
        layout = _make_rtl_layout(context, self.font)
        layout.set_text(word, -1)
        width, _ = layout.get_pixel_size()
        if is_last:
            offset += 2
        return self._get_unbounded_offset_to_next_patch(offset + width)

    # ------------------------------------------------------------------
    # Two-pass bigram-aligned path  (bigram_align=True)
    # ------------------------------------------------------------------

    def _build_bigram_slots(
        self,
        word: str,
        context: cairo.Context,
    ) -> Tuple[cairo.ImageSurface, List[_BigramSlot]]:
        """
        Shape *word* on an off-screen surface and partition the glyphs into
        bigram-aligned slots.

        Pass 1 – shape
        ~~~~~~~~~~~~~~
        Render the full *word* onto a scratch surface.  Because the whole word
        is given to Pango, every letter gets its contextually-correct form
        (initial كـ, medial ـكـ, final ـك, isolated ك) and all ligatures are
        formed (lam-alef لا, etc.).

        Pass 2 – partition into bigram slots
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Walk Pango's cluster iterator to record the pixel x-position of each
        logical character.  Group the characters into consecutive 2-character
        slots (matching the parent's bigram granularity).

        Ligature handling
        ~~~~~~~~~~~~~~~~~
        A ligature glyph covers multiple logical characters.  If the covered
        range straddles a 2-character boundary, the ligature is assigned
        entirely to the *earlier* slot; the next slot starts after the
        ligature.  This is necessary because a ligature is a single indivisible
        glyph in the font – there is no way to render half of it.

        Returns
        -------
        scratch_surface
            Off-screen surface containing the correctly shaped word.
        slots
            List of _BigramSlot in logical (reading) order, where each slot
            records the src_x / width region within the scratch surface.
        """
        # ── Pass 1: shape onto scratch surface ────────────────────────────
        measure_layout = _make_rtl_layout(context, self.font)
        measure_layout.set_text(word, -1)
        total_width, total_height = measure_layout.get_pixel_size()

        scratch_w = max(total_width + 4, 1)
        scratch_h = self.pixels_per_patch

        cairo_fmt = cairo.FORMAT_RGB24 if self.rgb else cairo.FORMAT_A8
        scratch = cairo.ImageSurface(cairo_fmt, scratch_w, scratch_h)
        scratch_ctx = cairo.Context(scratch)

        if self.rgb:
            scratch_ctx.set_source_rgb(1.0, 1.0, 1.0)
            scratch_ctx.rectangle(0, 0, scratch_w, scratch_h)
            scratch_ctx.fill()
            scratch_ctx.set_source_rgb(0.0, 0.0, 0.0)

        shaped_layout = _make_rtl_layout(scratch_ctx, self.font)
        shaped_layout.set_text(word, -1)
        y = scratch_h / 2.0 - total_height / 2.0 - 2
        scratch_ctx.move_to(0, y)
        PangoCairo.show_layout(scratch_ctx, shaped_layout)

        # ── Pass 2: map logical char index → pixel x ──────────────────────
        # char_x[i] = leftmost pixel of the cluster that starts at logical
        # character index i within the scratch surface.
        word_bytes = word.encode("utf-8")
        n_chars    = len(word)
        char_x: Dict[int, float] = {}

        layout_iter = shaped_layout.get_iter()
        while True:
            ext = layout_iter.get_cluster_extents()[1]          # logical extents
            x_px = ext.x / self.PANGO_SCALE
            byte_idx = layout_iter.get_index()                  # UTF-8 byte offset

            # Convert UTF-8 byte offset → Unicode character index
            char_idx = len(word_bytes[:byte_idx].decode("utf-8"))
            # Only record the first (smallest x) occurrence for each char_idx
            # to handle RTL visual ordering: in a RTL layout Pango's logical
            # extents start from the right, so x increases leftward.
            if char_idx not in char_x:
                char_x[char_idx] = x_px

            if not layout_iter.next_cluster():
                break

        # Sentinel: pixel just past the last glyph
        char_x[n_chars] = float(total_width)

        # ── Partition into bigram slots ───────────────────────────────────
        #
        # Arabic is stored in logical (reading) order: index 0 = first read
        # character = rightmost on screen.  We group in logical pairs (0-1,
        # 2-3, …) which correspond to visual pairs from right to left.
        #
        # For each slot we need the pixel x-range in the scratch surface.
        # Because of RTL layout, lower logical index → higher pixel x.
        # min(x_values) is therefore the *left* edge and max(x_values) is
        # the *right* edge of the slot in the scratch surface.
        slots: List[_BigramSlot] = []
        logical_idx = 0

        while logical_idx < n_chars:
            slot_end = min(logical_idx + 2, n_chars)

            # Collect x-positions for the characters in this slot.
            x_values = [char_x[i] for i in range(logical_idx, slot_end + 1) if i in char_x]

            if not x_values:
                # All characters consumed by a preceding ligature slot.
                logical_idx = slot_end
                if slots:
                    slots[-1].char_end = logical_idx
                continue

            slot_x_min = min(x_values)
            slot_x_max = max(x_values)

            # Ligature guard ───────────────────────────────────────────────
            # A ligature glyph maps multiple logical characters to the same
            # cluster position.  This shows up as char_x[slot_end] being
            # absent (the character has no independent glyph) or equal to an
            # adjacent value.  Extend the slot until all characters of the
            # ligature are included.
            while slot_end < n_chars:
                next_x = char_x.get(slot_end)
                if next_x is None:
                    # slot_end character is part of a ligature starting in
                    # this slot – absorb it.
                    slot_end += 1
                    absorb_x = char_x.get(slot_end, slot_x_max)
                    slot_x_min = min(slot_x_min, absorb_x)
                    slot_x_max = max(slot_x_max, absorb_x)
                else:
                    break

            slots.append(_BigramSlot(
                char_start=logical_idx,
                char_end=slot_end,
                src_x=slot_x_min,
                width=max(0.0, slot_x_max - slot_x_min),
            ))
            logical_idx = slot_end

        return scratch, slots

    def _render_bigram_aligned_arabic(
        self,
        word: str,
        offset: int,
        context: cairo.Context,
        is_last: bool = False,
    ) -> Tuple[cairo.Context, Pango.Layout, int]:
        """
        Render *word* in bigram-aligned mode.

        Shapes the full word off-screen (preserving all glyph forms and
        ligatures), then blits each bigram-sized region onto *context* at the
        appropriate patch-aligned destination offset.

        Each blit is equivalent to the parent rendering one Latin bigram:
        - The slot's pixel region is copied from the scratch surface.
        - The destination offset advances to the start of the next patch.

        Ligatures that span a bigram boundary are blitted as a single region
        at the earlier slot's offset, ensuring they always appear whole.
        """
        scratch, slots = self._build_bigram_slots(word, context)

        # Placeholder layout for the return value (callers only use it for
        # metrics on the whole sentence level; we always return one).
        placeholder_layout = _make_rtl_layout(context, self.font)
        placeholder_layout.set_text(word, -1)

        for i, slot in enumerate(slots):
            is_last_slot = (i == len(slots) - 1) and is_last

            if slot.width <= 0:
                if is_last_slot:
                    offset += 2
                offset = self._get_unbounded_offset_to_next_patch(offset)
                continue

            slot_w = int(math.ceil(slot.width))

            # Blit: translate the scratch surface so that scratch pixel
            # (slot.src_x, 0) lands at (offset, 0) on the destination.
            context.save()
            context.set_source_surface(scratch, offset - slot.src_x, 0)
            context.rectangle(offset, 0, slot_w, self.pixels_per_patch)
            context.fill()
            context.restore()

            if is_last_slot:
                offset += 2
            offset = self._get_unbounded_offset_to_next_patch(offset + slot_w)

        return context, placeholder_layout, offset

    def _offset_bigram_aligned_arabic(
        self,
        word: str,
        offset: int,
        context: cairo.Context,
        is_last: bool = False,
    ) -> int:
        """
        Measure *word* in bigram-aligned mode without painting.

        Builds bigram slots (shaping is required to know glyph widths) and
        accumulates the patch-aligned offset without blitting anything.
        """
        _, slots = self._build_bigram_slots(word, context)

        for i, slot in enumerate(slots):
            is_last_slot = (i == len(slots) - 1) and is_last
            if slot.width <= 0:
                if is_last_slot:
                    offset += 2
                offset = self._get_unbounded_offset_to_next_patch(offset)
                continue
            if is_last_slot:
                offset += 2
            offset = self._get_unbounded_offset_to_next_patch(
                offset + int(math.ceil(slot.width))
            )
        return offset

    # ------------------------------------------------------------------
    # Dispatch: _render_bigram / _offset_bigram
    # ------------------------------------------------------------------

    def _render_bigram(
        self,
        word: str,
        offset: int,
        context: cairo.Context,
        is_last: bool = False,
    ) -> Tuple[cairo.Context, Pango.Layout, int]:
        """
        Render one token, routing through the correct Arabic path.

        Routing table
        ~~~~~~~~~~~~~
        joining-script + bigram_align=True  → two-pass blit path
        joining-script + bigram_align=False → whole-word Pango path
        len > 2 (grapheme cluster / trigram) → whole-word Pango path
        everything else                      → parent's Latin bigram path
        """
        if _has_joining_script(word):
            if self.bigram_align:
                return self._render_bigram_aligned_arabic(word, offset, context, is_last)
            return self._render_single_word_arabic(word, offset, context, is_last)
        if len(word) > 2:
            # Grapheme clusters with combining marks – must stay together.
            return self._render_single_word_arabic(word, offset, context, is_last)
        return super()._render_bigram(word, offset, context, is_last)

    def _offset_bigram(
        self,
        word: str,
        offset: int,
        context: cairo.Context,
        is_last: bool = False,
    ) -> int:
        """
        Measure one token, routing through the correct Arabic path.
        Mirrors the dispatch logic of _render_bigram exactly.
        """
        if _has_joining_script(word):
            if self.bigram_align:
                return self._offset_bigram_aligned_arabic(word, offset, context, is_last)
            return self._offset_single_word_arabic(word, offset, context, is_last)
        if len(word) > 2:
            return self._offset_single_word_arabic(word, offset, context, is_last)
        return super()._offset_bigram(word, offset, context, is_last)

    # ------------------------------------------------------------------
    # Overridden _render_single_sentence
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

        This override exists solely to replace the inner ``string_to_ngrams``
        closure defined in the parent with the Arabic-aware version below.
        All patch-accounting, truncation search, and rendering loop logic is
        verbatim from the parent.

        The Arabic-aware splitter returns ``[word]`` for any token containing
        a joining-script character.  The full token is then passed to
        ``_offset_bigram`` / ``_render_bigram``, which dispatch to either the
        whole-word or the two-pass blit path depending on ``self.bigram_align``.

        For all other tokens the splitter is identical to the parent: split
        into ``n``-character chunks with the RTL reversal applied when needed.

        Note on bigram_align=True
        ~~~~~~~~~~~~~~~~~~~~~~~~~
        When bigram_align is True we still return ``[word]`` from
        ``string_to_ngrams`` for Arabic words.  The per-bigram splitting
        happens *inside* ``_render_bigram_aligned_arabic`` after Pango has
        shaped the full word.  This is intentional: n-gram splitting must not
        occur before shaping.
        """

        def string_to_ngrams(s: str, n: int = 2, rtl: bool = False) -> List[str]:
            if _has_joining_script(s):
                # Return the whole token; do NOT reverse – Pango handles
                # logical → visual reordering internally.
                return [s]
            if rtl:
                return [s[i:i + n][::-1] for i in range(0, len(s), n)][::-1]
            return [s[i:i + n] for i in range(0, len(s), n)]

        # ── Edge case: UD Treebanks v2.10 long URL ────────────────────────
        if len(words) == 1 and words[0].startswith("http://www.google.com/search"):
            logger.debug("UD Treebanks edge case URL detected; clamping.")
            words = ["http://www.google.com/search"]

        # ── word_patch_map skeleton ───────────────────────────────────────
        word_patch_map: Dict[str, Any] = {
            "sum_len_words":            sum(len(f"{w} ") for w in words) - len(" "),
            "total_number_of_words":    len(words),
            "word_order_on_canvas":     {},
            "text_is_truncated":        False,
            "word_list_truncation_index": len(words) - 1,
            "is_rtl":                   rtl,
        }

        word_counter       = 0
        max_length         = self.max_seq_length if max_length is None else max_length
        offset             = input_offset
        skip_last          = False
        word_start_indices = [math.ceil(offset / self.pixels_per_patch)]
        sentence_len_ltr   = 0
        sentence_len_rtl   = word_patch_map["sum_len_words"]

        # ── Measure first word ────────────────────────────────────────────
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

        # ── Measure middle words ──────────────────────────────────────────
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

        # ── Measure last word ─────────────────────────────────────────────
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

        # ── Truncation search ─────────────────────────────────────────────
        truncated_word_patch_map = word_patch_map.copy()
        index                    = word_patch_map["total_number_of_words"] - 1
        patches                  = word_patch_map["first_patch_after_last_word"]
        word_order_on_canvas     = word_patch_map["word_order_on_canvas"]

        if max_length is not None and patches > max_length:
            truncated_word_patch_map["text_is_truncated"] = True
            while patches > max_length:
                index  -= 1
                patches = max_length if index < 0 else word_order_on_canvas[index]["start_patch_index"]
            index -= 1  # step back once more to be safe for long last words
            if index < 0:
                logger.warning(
                    "First word in %s does not fit in max_length=%d; "
                    "returning canvas with only the first word.",
                    words, max_length,
                )
                index = 0

        # ── Render loop ───────────────────────────────────────────────────
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

        # ── Finalise truncated_word_patch_map ─────────────────────────────
        kept = list(range(index + 1))
        truncated_word_patch_map["word_order_on_canvas"] = {
            k: v for k, v in word_patch_map["word_order_on_canvas"].items()
            if k in kept
        }
        truncated_word_patch_map["sum_len_words"] = (
            sum(len(f"{w} ") for w in words[:index + 1]) - len(" ")
        )
        truncated_word_patch_map["total_number_of_words"]      = len(words[:index + 1])
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
        bigram_align: bool = False,
    ) -> "PangoCairoArabicTextRenderer":
        """
        Build a PangoCairoArabicTextRenderer from an existing renderer,
        copying all font / DPI / geometry settings.

        Parameters
        ----------
        renderer     : source PangoCairoTextRenderer to copy settings from
        bigram_align : passed through to the new instance

        Example
        -------
        >>> base = PangoCairoTextRenderer(font_file="NotoNaskhArabic.ttf")
        >>> ar = PangoCairoArabicTextRenderer.from_renderer(base)
        >>> ar_bigram = PangoCairoArabicTextRenderer.from_renderer(
        ...     base, bigram_align=True)
        """
        state = renderer.__getstate__()
        state.pop("__class__", None)
        return cls(**state, bigram_align=bigram_align)