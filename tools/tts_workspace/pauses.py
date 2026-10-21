"""Choose audible TTS pause boundaries before upstream short-text merging."""


CUT_METHODS = {
    "不切": "cut0",
    "凑四句一切": "cut1",
    "凑50字一切": "cut2",
    "按中文句号。切": "cut3",
    "按英文句号.切": "cut4",
    "按标点符号切": "cut5",
}
PUNCTUATION_CUTS = {"按中文句号。切", "按英文句号.切", "按标点符号切"}


def segments_with_boundaries(text, cut):
    from TTS_infer_pack.text_segmentation_method import get_method

    if cut not in CUT_METHODS:
        raise ValueError("请选择有效的文本切分方式")
    method = get_method(CUT_METHODS[cut])
    result = []
    # A typed line break is always an explicit boundary. Apply the chosen
    # punctuation rule within each line, before the TTS preprocessor can merge
    # short pieces back together.
    for line in text.splitlines():
        if not line.strip():
            continue
        first_in_line = True
        for piece in method(line.strip()).splitlines():
            piece = piece.strip()
            if piece and any(char.isalnum() for char in piece):
                boundary = None if not result else (
                    "line" if first_in_line else
                    "punctuation" if cut in PUNCTUATION_CUTS else "segment"
                )
                result.append((piece, boundary))
                first_in_line = False
    if not result:
        raise ValueError("请输入有效的目标文本")
    return result


def segments(text, cut):
    return [piece for piece, _ in segments_with_boundaries(text, cut)]
