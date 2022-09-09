from typing import List, Iterable

from variable import Variable


def dfs(u: Variable, l: List[Variable], visited: dict) -> None:
    """depth first search"""
    visited[u.label] = True
    for v in u.prev:
        if v.label not in visited:
            dfs(v, l, visited)
    l.insert(0, u)


def topo_sort(
    nodes: Iterable[Variable], l: List[Variable] = None, visited: dict = None
) -> Iterable[Variable]:
    """topological search"""
    l = [] if l is None else l
    visited = {k: False for k in nodes} if visited is None else visited
    for u in nodes:
        if not visited[u]:
            dfs(u, l, visited)
    return l
