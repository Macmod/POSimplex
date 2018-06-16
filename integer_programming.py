#!/usr/bin/env python
from fractions import Fraction as ff
from simplex import StdLP, Status
import numpy as np
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        fin = sys.stdin
    else:
        infile = sys.argv[1]
        fin = open(infile)

    # Input.
    k = int(fin.readline())
    m = int(fin.readline())
    n = int(fin.readline())
    matrix_line = fin.readline().strip()

    # Build input matrix & canonical basis.
    matrix = np.matrix(matrix_line).reshape(m+1, n+1) + ff()

    A = matrix[1:]
    c = matrix[0]
    std_A, std_c = StdLP.leq_to_std(A, c)
    std_matrix = np.concatenate((-std_c, std_A), axis=0)

    # Initial basis from slack variables
    basis_map = {}
    for i in range(1, m+1):
        basis_map[i] = m+n+i-1

    # Instantiate LP problem.
    problem = StdLP(
        std_matrix, logfile='pivoting.txt',
        debug=False, pretty=True
    )

    # Solve
    if k == 1:
        print('\n[Branch & bound method]')
        status = problem.apply_branch_and_bound(basis_map)
    else:
        print('\n[Cutting plane method]')
        status = problem.apply_cutting_plane(basis_map)

    # Write simplex output.
    outfile = 'conclusao.txt'
    with open(outfile, 'w') as fout:
        fout.write('%d\n' % status)
        cert = StdLP.printable(problem.cert)

        if status == Status.OPTIMAL:
            opt_vec = StdLP.printable(problem.opt_vec)
            opt_val = problem.opt_val
            lr_cert = StdLP.printable(problem.lr_cert)
            lr_opt_vec = StdLP.printable(problem.lr_opt_vec)
            lr_opt_val = problem.lr_opt_val

            print('\n-\n[+] Optimal found:', opt_val)
            print('[Sol]', opt_vec)
            print('[+] LR Optimal found:', lr_opt_val)
            print('[LR Sol]', lr_opt_vec)
            print('[LR Cert]', lr_cert)

            fout.write(
                '%s' % opt_vec + '\n'
                '%s' % opt_val + '\n'
                '%s' % lr_opt_vec + '\n'
                '%s' % lr_opt_val + '\n'
                '%s' % lr_cert + '\n'
            )
        elif status == Status.UNBOUNDED:
            print('[x] Unbounded.')
            print('[Cert]', cert)

            fout.write('%s' % cert + '\n')
        elif status == Status.INFEASIBLE:
            print('[x] Infeasible')

    fin.close()
