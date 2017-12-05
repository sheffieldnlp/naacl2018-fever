from __future__ import print_function
import os


def main():
    """Examples: Multiple-choice Vector Bin Packing"""
    from pyvpsolver.solvers import mvpsolver
    os.chdir(os.path.dirname(__file__) or os.curdir)

    # Example 1:
    # Bins:
    test = (3333, 3333, 3333)
    dev = (3333,3333,3333)
    train = (1e10,1e10,1e10)

    Ws = [test, dev, train]    # Capacities
    Cs = [1, 1, 1]       # Costs
    Qs = [1, 1, 1]    # Number of available bins

    # Items:
    ws1, b1 = [(50, 25,20)], 1
    b = [b1]
    ws = [ws1]

    # Solve Example 1:
    solution = mvpsolver.solve(
        Ws, Cs, Qs, ws, b,
        svg_file="tmp/graphA_mvbp.svg",
        script="vpsolver_glpk.sh",
        verbose=True
    )
    mvpsolver.print_solution(solution)

    # check the solution objective value
    obj, patterns = solution
    assert obj == 1



if __name__ == "__main__":
    main()
