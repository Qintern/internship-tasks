from typing import List


def num_of_islands(grid: List[List[int]]) -> int:
    result = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == 1:
                dfs_grid(grid, row, col)
                result += 1
    return result


def dfs_grid(grid: List[List[int]], row: int, col: int):
    grid[row][col] = 0
    if row - 1 >= 0 and grid[row - 1][col] == 1:
        dfs_grid(grid, row - 1, col)
    if row + 1 < len(grid) and grid[row + 1][col] == 1:
        dfs_grid(grid, row + 1, col)
    if col - 1 >= 0 and grid[row][col - 1] == 1:
        dfs_grid(grid, row, col - 1)
    if col + 1 < len(grid[0]) and grid[row][col + 1] == 1:
        dfs_grid(grid, row, col + 1)


def main():
    mn = input().split()
    m = int(mn[0])
    # n = int(mn[1])
    map_input = []
    for _ in range(m):
        map_input.append(list(map(int, input().rstrip().split())))
    result = num_of_islands(map_input)
    print(result)


if __name__ == '__main__':
    main()
