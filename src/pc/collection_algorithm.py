import heapq

from settings import MAP_HEIGHT, MAP_WIDTH


ROW = MAP_HEIGHT
COL = MAP_WIDTH
DIRECTIONS = (
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


class PathCell:
    def __init__(self):
        self.parent_row = 0
        self.parent_col = 0
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0


def destination(row, col, dest_y, dest_x):
    return col == dest_x and row == dest_y


def get_h(ry, rx, by, bx):
    return ((rx - bx) ** 2 + (ry - by) ** 2) ** 0.5


def trace_path(cell, dx, dy):
    path = []
    row = dy
    col = dx

    while not (cell[row][col].parent_row == row and cell[row][col].parent_col == col):
        path.append((row, col))
        temp_row = cell[row][col].parent_row
        temp_col = cell[row][col].parent_col
        row = temp_row
        col = temp_col

    path.append((row, col))
    path.reverse()

    for i in path:
        print("->", i, end=" ")
    print()
    return path


def is_valid(row, col):
    return 0 <= row < ROW and 0 <= col < COL


def is_unblocked(grid, row, col):
    return grid[row][col] == '.' or grid[row][col] == 'W'


def A_star(grid, src, dest):
    if not is_valid(src[0], src[1]) or not is_valid(dest[0], dest[1]):
        return "The specified rows and columns are not valid"

    if (
        not is_unblocked(grid, src[0], src[1])
        or not is_unblocked(grid, dest[0], dest[1])
    ):
        return "Source or destination is blocked"

    if destination(src[0], src[1], dest[0], dest[1]):
        return "Source is already at destination"

    closed_list = [[False for _ in range(COL)] for _ in range(ROW)]
    cell_details = [[PathCell() for _ in range(COL)] for _ in range(ROW)]

    row = src[0]
    col = src[1]
    cell_details[row][col].f = 0
    cell_details[row][col].g = 0
    cell_details[row][col].h = 0
    cell_details[row][col].parent_row = row
    cell_details[row][col].parent_col = col

    open_list = []
    heapq.heappush(open_list, (0.0, row, col))

    while open_list:
        p = heapq.heappop(open_list)

        row = p[1]
        col = p[2]

        closed_list[row][col] = True

        for row_delta, col_delta in DIRECTIONS:
            new_row = row + row_delta
            new_col = col + col_delta

            if (
                is_valid(new_row, new_col)
                and is_unblocked(grid, new_row, new_col)
                and not closed_list[new_row][new_col]
            ):
                if destination(new_row, new_col, dest[0], dest[1]):
                    cell_details[new_row][new_col].parent_row = row
                    cell_details[new_row][new_col].parent_col = col
                    print("destination cell found")
                    path = trace_path(cell_details, dest[1], dest[0])
                    print("Pathfinding done")
                    return path
                else:
                    g_new = cell_details[row][col].g + 1.4
                    h_new = get_h(new_row, new_col, dest[0], dest[1])
                    f_new = g_new + h_new

                    if (
                        cell_details[new_row][new_col].f == float('inf')
                        or cell_details[new_row][new_col].f > f_new
                    ):
                        heapq.heappush(open_list, (f_new, new_row, new_col))
                        cell_details[new_row][new_col].f = f_new
                        cell_details[new_row][new_col].g = g_new
                        cell_details[new_row][new_col].h = h_new
                        cell_details[new_row][new_col].parent_row = row
                        cell_details[new_row][new_col].parent_col = col

    return "No path found"
