from __future__ import annotations

from dataclasses import dataclass, field


COLOR_MAP: dict[str, tuple[tuple[int, int, int], tuple[str, ...]]] = {
    "black": ((24, 24, 24), ("black", "검정", "검은", "까만")),
    "white": ((238, 238, 232), ("white", "흰", "하얀", "화이트")),
    "blue": ((42, 92, 190), ("blue", "파랑", "파란", "남색", "네이비")),
    "red": ((190, 42, 42), ("red", "빨강", "빨간")),
    "yellow": ((225, 190, 48), ("yellow", "노랑", "노란")),
    "green": ((42, 135, 82), ("green", "초록", "녹색")),
    "gray": ((128, 128, 128), ("gray", "grey", "회색", "그레이")),
    "brown": ((118, 76, 44), ("brown", "갈색", "브라운")),
    "beige": ((190, 170, 128), ("beige", "베이지")),
}

ITEM_ALIASES: dict[str, tuple[str, ...]] = {
    "jacket": ("jacket", "jumper", "점퍼", "자켓", "재킷", "패딩", "가디건", "cardigan", "coat", "코트"),
    "hoodie": ("hoodie", "후드", "후드티"),
    "tshirt": ("tshirt", "t-shirt", "티셔츠", "반팔", "긴팔"),
    "shirt": ("shirt", "셔츠", "남방"),
    "top": ("상의", "윗옷", "top", "상반신"),          # 일반 상의 지칭
    "pants": ("pants", "바지", "슬랙스", "반바지", "트레이닝"),
    "jeans": ("jeans", "청바지", "데님"),
    "skirt": ("skirt", "치마", "스커트"),
    "bottom": ("하의", "아랫옷", "bottom"),             # 일반 하의 지칭
    "dress": ("dress", "원피스"),
    "sneakers": ("sneakers", "shoes", "운동화", "신발"),
}

UPPER_ITEMS = {"jacket", "hoodie", "tshirt", "shirt", "top"}
LOWER_ITEMS = {"pants", "jeans", "skirt", "bottom"}
FULL_BODY_ITEMS = {"dress"}
SHOE_ITEMS = {"sneakers"}


@dataclass(slots=True)
class OutfitSpec:
    raw_text: str
    colors: list[str] = field(default_factory=list)
    items: list[str] = field(default_factory=list)
    upper_color: str | None = None
    lower_color: str | None = None
    shoe_color: str | None = None
    category: str = "full_body"

    def primary_color(self) -> str:
        return self.upper_color or self.lower_color or self.shoe_color or (self.colors[0] if self.colors else "gray")


def _all_positions(lowered: str, aliases: tuple[str, ...]) -> list[int]:
    """문장 내 모든 alias 등장 위치."""
    out: list[int] = []
    for alias in aliases:
        a = alias.lower()
        start = 0
        while True:
            idx = lowered.find(a, start)
            if idx < 0:
                break
            out.append(idx)
            start = idx + 1
    return out


def _nearest_color(item_pos: int, color_positions: list[tuple[int, str]]) -> str | None:
    """옷 키워드 위치에 가장 가까운 색상. 한국어는 색이 옷 앞에 오므로
    item_pos 이전 색을 우선, 없으면 가장 가까운 색."""
    if not color_positions:
        return None
    before = [(item_pos - p, name) for p, name in color_positions if p <= item_pos]
    if before:
        return min(before, key=lambda x: x[0])[1]
    return min(color_positions, key=lambda x: abs(x[0] - item_pos))[1]


def parse_outfit(text: str) -> OutfitSpec:
    lowered = (text or "").lower()

    # 모든 색상 등장 위치 (부위별 근접 배정에 사용)
    color_positions: list[tuple[int, str]] = []
    for name, (_rgb, aliases) in COLOR_MAP.items():
        for pos in _all_positions(lowered, aliases):
            color_positions.append((pos, name))
    color_positions.sort()

    # 모든 옷 키워드 위치
    item_positions: list[tuple[int, str]] = []
    for name, aliases in ITEM_ALIASES.items():
        ps = _all_positions(lowered, aliases)
        if ps:
            item_positions.append((min(ps), name))
    item_positions.sort()

    colors = [name for _pos, name in dict.fromkeys((n, p) for p, n in color_positions)] if color_positions else []
    # 중복 제거하며 순서 유지
    seen_c: set[str] = set()
    colors = [n for _p, n in color_positions if not (n in seen_c or seen_c.add(n))]
    items = [name for _pos, name in item_positions]

    # 부위별로 가장 가까운 색 배정 (위치 기반)
    upper_color = lower_color = shoe_color = None
    for pos, name in item_positions:
        if name in UPPER_ITEMS or name in FULL_BODY_ITEMS:
            if upper_color is None:
                upper_color = _nearest_color(pos, color_positions)
        elif name in LOWER_ITEMS:
            if lower_color is None:
                lower_color = _nearest_color(pos, color_positions)
        elif name in SHOE_ITEMS:
            if shoe_color is None:
                shoe_color = _nearest_color(pos, color_positions)

    # 옷 키워드가 전혀 없으면 첫 색을 상의로 가정 (하위 호환)
    if upper_color is None and lower_color is None and shoe_color is None and colors:
        upper_color = colors[0]

    category = "full_body"
    if items and all(item in UPPER_ITEMS for item in items):
        category = "upper_body"
    elif items and all(item in LOWER_ITEMS for item in items):
        category = "lower_body"

    return OutfitSpec(
        raw_text=text or "",
        colors=colors,
        items=items,
        upper_color=upper_color,
        lower_color=lower_color,
        shoe_color=shoe_color,
        category=category,
    )


def color_rgb(name: str | None) -> tuple[int, int, int]:
    if not name:
        name = "gray"
    return COLOR_MAP.get(name, COLOR_MAP["gray"])[0]
