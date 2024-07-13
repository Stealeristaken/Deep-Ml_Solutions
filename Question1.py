def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float]:
    if len(a[0]) != len(b):
        return [-1]  # Return a list with -1 instead of just -1
    
    return [sum(x * y for x, y in zip(row, b)) for row in a]