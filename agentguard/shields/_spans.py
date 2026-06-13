"""Shared span-merging used by the regex redaction shields.

Detection produces possibly-overlapping ``(start, end, label)`` spans. Replacing
them naively corrupts the text and — worse for a security tool — can leave part
of a sensitive value exposed when a narrower span shadows a wider overlapping
one. The safe rule is to redact the **union** of any overlapping spans (never
leak a covered character) and label each merged region by its widest
contributing span.
"""


def merge_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Merge overlapping/adjacent spans into a non-overlapping cover.

    Returns spans sorted descending by start, so a caller can splice
    replacements from the end of the string without invalidating offsets.
    """
    spans = [s for s in spans if s[1] > s[0]]
    if not spans:
        return []

    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

    merged: list[tuple[int, int, str]] = []
    cur_start, cur_end, cur_label = spans[0]
    cur_width = cur_end - cur_start

    for start, end, label in spans[1:]:
        if start < cur_end:  # overlaps the open region — extend the union
            if end > cur_end:
                cur_end = end
            # Keep the label of the widest single contributing span.
            if (end - start) > cur_width:
                cur_label = label
                cur_width = end - start
        else:
            merged.append((cur_start, cur_end, cur_label))
            cur_start, cur_end, cur_label = start, end, label
            cur_width = end - start
    merged.append((cur_start, cur_end, cur_label))

    merged.sort(key=lambda s: s[0], reverse=True)
    return merged
