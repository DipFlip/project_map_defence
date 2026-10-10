"""Two players at the short ends of the table, each holding a pair of wands."""
import numpy as np

# wand pairs: tags G+B and D+C
PAIRS = ((16, 11), (13, 12))
COLORS = ((70, 220, 120), (255, 150, 50))  # no blue: blue is the wand buttons' colour


class Player:
    def __init__(self, ids, side, field, color):
        self.ids, self.side, self.color = ids, side, color
        self.end_x = field.left if side == "left" else field.right
        self.dir = 1 if side == "left" else -1  # +x points from this player towards the opponent
        self.facing = side  # text drawn upright for someone standing at this end

    def wands(self, tags):
        return [tags[k] for k in self.ids if k in tags]


def make_players(tags, field):
    """Left/right assignment from where the pairs are now (pair 0 left if unknown)."""
    def mean_x(pair):
        xs = [tags[k].pos[0] for k in pair if k in tags]
        return np.mean(xs) if xs else None

    a, b = mean_x(PAIRS[0]), mean_x(PAIRS[1])
    left = 0
    if a is not None and b is not None:
        left = 0 if a <= b else 1
    elif b is not None:
        left = 1 if b < field.centerx else 0
    elif a is not None:
        left = 0 if a < field.centerx else 1
    right = 1 - left
    return [Player(PAIRS[left], "left", field, COLORS[left]), Player(PAIRS[right], "right", field, COLORS[right])]
