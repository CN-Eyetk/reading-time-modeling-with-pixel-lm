"""
PangoCairoArabicTextRenderer  –  bigram-patch-aligned edition
==============================================================

Design contract (mirrors PangoCairoTextRenderer._render_bigram / _offset_bigram)
---------------------------------------------------------------------------
1. **Every new word starts a new patch.**
   A word boundary always lands on a patch boundary, identical to the general
   bigram renderer.

2. **Each patch holds at most 2 logical characters.**
   Within one word the characters are laid out 2-at-a-time, each pair occupying
   exactly one patch slot, same cadence as the Latin bigram renderer.

3. **Contextual glyph forms are preserved.**
   Arabic letters appear in their correct initial / medial / final / isolated
   forms because the *full word* is shaped by Pango/HarfBuzz before slicing.

4. **Cross-patch connection breaks gracefully.**
   When a join breaks at a patch boundary the first character retains its
   "connects-right" shape and the next character retains its "connects-left"
   shape, because both were shaped inside the full-word context.

5. **Ligatures (e.g. lam-alef لا) are never split.**
   A ligature is a single indivisible glyph.  When one spans a bigram boundary
   the slot is widened to contain all its constituent characters.

RTL pixel geometry
------------------
Pango renders RTL text so that logical character 0 (first in reading order) is
placed at the *right* side of the layout and char N-1 at the *left*.  Each
character's cluster extent reports the LEFT edge of its glyph box measured from
the left of the layout origin.

For a slot covering logical characters [a, b):

    right_px = total_width        if a == 0
             = char_x[a - 1]     otherwise   (left edge of the preceding char
                                               = right edge of char a)
    left_px  = char_x[k]         where k is the largest index < b
                                  present in char_x  (= left edge of the last
                                  char in the slot, or the last independent
                                  cluster before a ligature absorbs the rest)

    src_x  = left_px
    width  = right_px - left_px

These two values give a contiguous, non-overlapping partition of the scratch
surface whose widths sum exactly to total_width.

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
# RTL Pango layout helpers
# ---------------------------------------------------------------------------

def _set_rtl_on_layout(layout: Pango.Layout) -> None:
    """Set RTL base direction on *layout* **before** calling set_text."""
    layout.set_auto_dir(False)
    ctx = layout.get_context()
    ctx.set_base_dir(Pango.Direction.RTL)
    layout.set_alignment(Pango.Alignment.RIGHT)


def _make_rtl_layout(
    context: cairo.Context,
    font: Pango.FontDescription,
) -> Pango.Layout:
    """Create a Pango layout with RTL direction and *font* pre-loaded."""
    pango_ctx = PangoCairo.create_context(context)
    pango_ctx.set_font_description(font)
    layout = Pango.Layout(pango_ctx)
    _set_rtl_on_layout(layout)
    return layout


# ---------------------------------------------------------------------------
# _PatchSlot: one 1-or-2-char patch slot within a shaped word
# ---------------------------------------------------------------------------

class _PatchSlot:
    """
    One patch slot produced by :func:`_build_patch_slots`.

    Attributes
    ----------
    char_start, char_end : int
        Half-open logical character range [char_start, char_end).
        Normally ``char_end == char_start + 2`` (bigram), wider when a
        ligature spans the boundary.
    src_x : float
        Left pixel edge of this slot's region inside the shaped scratch surface.
    width : float
        Pixel width of the region.  src_x + width = right pixel edge.
    """
    __slots__ = ("char_start", "char_end", "src_x", "width")

    def __init__(
        self,
        char_start: int,
        char_end: int,
        src_x: float,
        width: float,
    ) -> None:
        self.char_start = char_start
        self.char_end   = char_end
        self.src_x      = src_x
        self.width      = width


# ---------------------------------------------------------------------------
# Core: shape a word and partition it into bigram patch slots
# ---------------------------------------------------------------------------

def _last_present_x(char_x: Dict[int, float], before_idx: int) -> float:
    """
    Return ``char_x[k]`` for the largest ``k < before_idx`` present in
    *char_x*, or ``0.0`` if no such key exists.

    Used to find the left pixel boundary of the last *independent* cluster
    inside a slot when ligatures have absorbed some intermediate characters.
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
    bigram-sized patch slots that can each be blitted to one output patch.

    Parameters
    ----------
    word             : Arabic (or other joining-script) token
    context          : active Cairo context (used only to create a matching
                       Pango context; nothing is painted here)
    font             : Pango font description
    pixels_per_patch : height of the output strip (= patch height)
    rgb              : True → RGB24 scratch surface, False → A8
    pango_scale      : Pango's internal unit scale (always 1024)

    Returns
    -------
    scratch_surface
        Off-screen surface containing the fully shaped word rendered at x=0.
    slots
        List of _PatchSlot in logical order (index 0 = first reading-order
        character = rightmost on screen for RTL).  Each slot contains:
        ``src_x``  – left pixel edge of the region inside the scratch surface
        ``width``  – pixel width of that region
        Together they define a contiguous, non-overlapping partition of the
        scratch surface; their widths sum to ``total_width``.

    Pixel geometry (RTL)
    --------------------
    Pango places char 0 (first logical) at the RIGHT of the layout and char
    N-1 at the LEFT.  ``char_x[i]`` = left edge of char i's glyph, so values
    DECREASE as logical index increases.

    For slot [a, b):
        right_px = total_width        if a == 0
                 = char_x[a - 1]     otherwise
        left_px  = last_present_x(char_x, b)   (= char_x[b-1] normally,
                                                  wider if b-1 is a ligature)
        src_x  = left_px
        width  = right_px - left_px
    """

    # ── Shape onto off-screen scratch surface ─────────────────────────────
    # Give the FULL word to Pango so every letter receives its contextually-
    # correct form (initial / medial / final / isolated) and all ligatures
    # (lam-alef, etc.) are formed.  Splitting before shaping would yield
    # isolated forms only.
    measure_layout = _make_rtl_layout(context, font)
    measure_layout.set_text(word, -1)
    total_width, total_height = measure_layout.get_pixel_size()

    scratch_w = max(total_width + 4, 1)
    scratch_h = pixels_per_patch

    fmt     = cairo.FORMAT_RGB24 if rgb else cairo.FORMAT_A8
    scratch = cairo.ImageSurface(fmt, scratch_w, scratch_h)
    sctx    = cairo.Context(scratch)

    if rgb:
        sctx.set_source_rgb(1.0, 1.0, 1.0)
        sctx.rectangle(0, 0, scratch_w, scratch_h)
        sctx.fill()
        sctx.set_source_rgb(0.0, 0.0, 0.0)

    shaped = _make_rtl_layout(sctx, font)
    shaped.set_text(word, -1)
    y_pos = scratch_h / 2.0 - total_height / 2.0 - 2
    sctx.move_to(0, y_pos)
    PangoCairo.show_layout(sctx, shaped)

    # ── Build char_x: logical character index → left pixel edge ───────────
    # Pango's cluster iterator yields UTF-8 byte offsets; convert to Unicode
    # character indices.  We only record the FIRST (earliest logical) entry
    # per character index to handle any ambiguous cluster ordering.
    word_bytes = word.encode("utf-8")
    n_chars    = len(word)
    char_x: Dict[int, float] = {}

    it = shaped.get_iter()
    while True:
        ext      = it.get_cluster_extents()[1]          # logical extents
        x_px     = ext.x / pango_scale                  # left edge, pixels
        byte_idx = it.get_index()
        char_idx = len(word_bytes[:byte_idx].decode("utf-8"))
        if char_idx not in char_x:
            char_x[char_idx] = x_px
        if not it.next_cluster():
            break

    # ── Partition into bigram patch slots ──────────────────────────────────
    slots: List[_PatchSlot] = []
    logical_idx = 0

    while logical_idx < n_chars:
        slot_end = min(logical_idx + 2, n_chars)  # tentative end (bigram)

        # Ligature guard: if a character between logical_idx and slot_end has
        # no independent cluster entry it has been fused into a ligature with
        # a preceding character.  Extend slot_end until every member of that
        # ligature is included.
        while slot_end < n_chars and slot_end not in char_x:
            slot_end += 1

        # RIGHT boundary of this slot (pixel x) ───────────────────────────
        # = right edge of char[logical_idx]
        # = left edge of char[logical_idx - 1]  for idx > 0
        # = total_width                           for idx == 0 (first char
        #   is rightmost; its right edge is the end of the layout)
        if logical_idx == 0:
            right_px = float(total_width)
        else:
            right_px = char_x.get(logical_idx - 1, float(total_width))

        # LEFT boundary of this slot (pixel x) ────────────────────────────
        # = left edge of the LAST character in the slot
        # = char_x[slot_end - 1] in the normal case
        # When slot_end - 1 has no entry (ligature hole), walk backwards to
        # the last independent cluster entry before slot_end.
        left_px = _last_present_x(char_x, slot_end)

        src_x = left_px
        width = max(0.0, right_px - left_px)

        if width > 0 or slot_end > logical_idx:  # always append, even zero-width
            slots.append(_PatchSlot(
                char_start=logical_idx,
                char_end=slot_end,
                src_x=src_x,
                width=width,
            ))

        logical_idx = slot_end

    return scratch, slots


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PangoCairoArabicTextRenderer(PangoCairoTextRenderer):
    """
    Drop-in replacement for PangoCairoTextRenderer with Arabic-aware rendering.

    Rendering contract
    ------------------
    * Every new **word** starts a new patch (identical to the general bigram
      renderer).
    * Each patch holds at most **2 logical characters** (bigram cadence).
    * **Contextual glyph forms** (initial / medial / final / isolated) and
      **ligatures** are resolved by shaping the *full word* before slicing.
    * Cross-patch connection breaks preserve the "connects-right" shape on the
      first character of each pair because it was shaped in full-word context.
    * Ligatures that span a bigram boundary are kept whole in the earlier slot.

    All constructor parameters are identical to PangoCairoTextRenderer.

    Usage
    -----
    >>> r = PangoCairoArabicTextRenderer(font_file="NotoNaskhArabic.ttf")
    >>> encoding = r("مرحبا بالعالم")          # single string
    >>> encoding = r(["مرحبا", "بالعالم"])      # pre-tokenised list

    >>> # Upgrade an existing renderer
    >>> ar = PangoCairoArabicTextRenderer.from_renderer(base_renderer)
    """

    # ------------------------------------------------------------------
    # Pickling
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        return super().__getstate__()

    def __setstate__(self, state_dict: Dict[str, Any]) -> None:
        self.__init__(**state_dict)

    # ------------------------------------------------------------------
    # Internal: shape-then-slice
    # ------------------------------------------------------------------

    def _shape_arabic_word(
        self,
        word: str,
        context: cairo.Context,
    ) -> Tuple[cairo.ImageSurface, List[_PatchSlot]]:
        """Thin wrapper around :func:`_build_patch_slots` using renderer settings."""
        return _build_patch_slots(
            word=word,
            context=context,
            font=self.font,
            pixels_per_patch=self.pixels_per_patch,
            rgb=self.rgb,
            pango_scale=self.PANGO_SCALE,
        )

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

        Routing
        -------
        joining-script token   → shape-then-slice (Arabic path)
        len > 2, non-joining   → parent _render_single_word (grapheme cluster)
        everything else        → parent _render_bigram (Latin ≤2 chars)

        Arabic path
        -----------
        Shapes the full word on a scratch surface (preserving all contextual
        forms and ligatures), then blits each bigram-sized pixel region onto
        the output surface at the next patch-aligned offset — exactly as the
        parent does for Latin bigrams.
        """
        if not _has_joining_script(word):
            if len(word) > 2:
                return super()._render_single_word(word, offset, context, is_last)
            return super()._render_bigram(word, offset, context, is_last)

        # ── Arabic: shape-then-slice ───────────────────────────────────────
        scratch, slots = self._shape_arabic_word(word, context)

        # Placeholder layout for callers that inspect layout-level metrics.
        placeholder = _make_rtl_layout(context, self.font)
        placeholder.set_text(word, -1)

        for i, slot in enumerate(slots):
            last_slot = (i == len(slots) - 1) and is_last

            if slot.width <= 0:
                if last_slot:
                    offset += 2
                offset = self._get_unbounded_offset_to_next_patch(offset)
                continue

            slot_w = int(math.ceil(slot.width))

            # Blit: translate scratch so that scratch pixel (slot.src_x, 0)
            # lands at (offset, 0) on the destination canvas.
            context.save()
            context.set_source_surface(scratch, offset - slot.src_x, 0)
            context.rectangle(offset, 0, slot_w, self.pixels_per_patch)
            context.fill()
            context.restore()

            if last_slot:
                offset += 2
            offset = self._get_unbounded_offset_to_next_patch(offset + slot_w)

        return context, placeholder, offset

    def _offset_bigram(
        self,
        word: str,
        offset: int,
        context: cairo.Context,
        is_last: bool = False,
    ) -> int:
        """
        Measure one token without painting, using the same routing logic as
        :meth:`_render_bigram` so measurement and render passes agree exactly.
        """
        if not _has_joining_script(word):
            if len(word) > 2:
                return super()._offset_single_word(word, offset, context, is_last)
            return super()._offset_bigram(word, offset, context, is_last)

        _, slots = self._shape_arabic_word(word, context)

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

        The only difference from the parent is the inner ``string_to_ngrams``
        closure: for joining-script tokens it returns ``[word]`` so the entire
        token is passed to :meth:`_render_bigram`, which performs the
        shape-then-slice partitioning *after* Pango has resolved all contextual
        forms.  For all other tokens the behaviour is identical to the parent.
        """

        def string_to_ngrams(s: str, n: int = 2, rtl: bool = False) -> List[str]:
            if _has_joining_script(s):
                # Do NOT split or reverse; Pango handles RTL reordering.
                return [s]
            if rtl:
                return [s[i : i + n][::-1] for i in range(0, len(s), n)][::-1]
            return [s[i : i + n] for i in range(0, len(s), n)]

        # Edge case: UD Treebanks v2.10 long URL
        if len(words) == 1 and words[0].startswith("http://www.google.com/search"):
            logger.debug("UD Treebanks edge-case URL detected; clamping.")
            words = ["http://www.google.com/search"]

        # ── word_patch_map skeleton ────────────────────────────────────────
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

        # Finalise truncated_word_patch_map
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
    ) -> "PangoCairoArabicTextRenderer":
        """
        Build a PangoCairoArabicTextRenderer from an existing renderer,
        copying all font / DPI / geometry settings.

        Example
        -------
        >>> base = PangoCairoTextRenderer(font_file="NotoNaskhArabic.ttf")
        >>> ar   = PangoCairoArabicTextRenderer.from_renderer(base)
        """
        state = renderer.__getstate__()
        state.pop("__class__", None)
        return cls(**state)