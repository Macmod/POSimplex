#!/usr/bin/env python3
from fractions import Fraction as ff
from math import floor, ceil
from enum import IntEnum
import numpy as np
import sys

'''

    [DCC-UFMG]
    Assignment: TP1 & TP2 - Linear & Integer Programming
    Student: Artur Marzano
    Class: Operations Research (18/1)

'''


# Custom exceptions
class FormatException(Exception):
    pass


class UnboundedException(Exception):
    pass


class InfeasibleException(Exception):
    pass


class WrongCanonicalBasis(Exception):
    pass


# Possible LP outcomes
class Status(IntEnum):
    INFEASIBLE = 0
    UNBOUNDED = 1
    OPTIMAL = 2


class StdLP:
    '''
        This class represents LPs in standard form.
        The M input is the corresponding tableau
        for the problem.
    '''

    def __init__(self, M, debug=False, pretty=False, logfile=None):
        ''' Stores the LP and preferences. '''

        self.M = M
        self.Mx = StdLP.append_log(M.copy())

        self.opt_vec = None
        self.opt_val = None
        self.cert = None

        self.lr_opt_vec = None
        self.lr_opt_val = None
        self.bb_best_vec = None
        self.bb_best_val = None

        self.debug = debug
        self.pretty = pretty

        if logfile:
            self.logfile = open(logfile, 'w')
        else:
            self.logfile = None

        if debug:
            print('[Input]')
            print(self)

    def __del__(self):
        ''' Close logfile. '''
        if self.logfile:
            self.logfile.close()

    def __repr__(self):
        ''' Print current tableau. '''

        if self.pretty:
            sep = '\t| '
            wrap = ['', '']
        else:
            sep = ', '
            wrap = ['[', ']']

        return StdLP.printable(self.Mx, sep, wrap)

    @staticmethod
    def fidentity(n):
        ''' Identity matrix of Fraction objects. '''
        return np.identity(n, dtype='object') + ff()

    @staticmethod
    def printable(M, sep=', ', wrap=['[', ']']):
        ''' Print fraction matrix. '''

        h, w = M.shape
        out = []

        for row in range(h):
            out.append([])
            for col in range(w):
                elem = str(M[row, col])
                out[-1].append(elem)
            out[-1] = wrap[0] + sep.join(out[-1]) + wrap[1]

        return '\n'.join(out)

    @staticmethod
    def append_log(A):
        ''' Appends the logging matrix to the tableau. '''

        h, w = A.shape
        log = np.concatenate(
            (np.matrix([ff(0)]*(h-1)), StdLP.fidentity(h-1)),
            axis=0
        )

        return np.concatenate((log, A), axis=1)

    @staticmethod
    def build_aux(M):
        ''' Builds the auxiliary tableau. '''

        h, w = M.shape

        # Clear target function.
        M[0] -= M[0]

        # Appended matrix
        aux = np.concatenate(
            (np.matrix([ff(1)]*(h-1)), StdLP.fidentity(h-1)),
            axis=0
        )

        new_M = np.concatenate((M[:, :-1], aux, M[:, -1]), axis=1)
        return new_M

    @staticmethod
    def leq_to_std(A, c):
        ''' Converts an instance of
            max  c^Tx
            s.t. Ax <= b
                 x >= 0
            into standard form. '''

        dim, w = A.shape

        c_new = np.concatenate(
            (c[:, :-1], np.matrix([0]*dim), c[:, -1]),
            axis=1
        )

        A_new = np.concatenate(
            (A[:, :-1], StdLP.fidentity(dim), A[:, -1]),
            axis=1
        )

        return A_new, c_new

    def pivot(self, i, j, autofix=False):
        ''' Pivots M[i, j].
            autofix automatically adds a non-zero element
            from another row if M[i, j] is zero.
        '''

        M = self.Mx
        h, w = M.shape
        elem = M[i, j]

        if autofix and elem == 0:
            for y in range(h):
                if M[y, j] != 0:
                    M[i] += M[y]
                    elem = M[i, j]
                    break

        if self.debug:
            print('\n[Pivot]', i, j)

        for x in range(w):
            M[i, x] /= elem

        for y in range(h):
            if y != i:
                k = M[y, j]
                for x in range(w):
                    M[y, x] -= M[i, x] * k

        if self.logfile:
            self.logfile.write(
                '[Pivot] %d %d\n' % (i, j) + self.__repr__() + '\n\n'
            )

    def pick_primal_pivot(self):
        ''' Picks a pivot to be used by a primal Simplex. '''

        M = self.Mx
        h, w = M.shape

        min_ratio = -1
        i = j = -1

        for y in range(1, h):
            if M[y, -1] < 0:
                raise FormatException(
                    'This LP cannot be solved by primal '
                    'Simplex (negative value in b vector).'
                )

        for x in range(h-1, w-1):
            if M[0, x] >= 0:
                continue

            for y in range(1, h):
                if M[y, x] <= 0:
                    continue

                ratio = M[y, -1] / M[y, x]
                if min_ratio == -1 or ratio < min_ratio:
                    min_ratio = ratio
                    i = y
                    j = x

            if min_ratio != -1:
                return i, j
            else:
                self.neg_col = x
                raise UnboundedException('This LP is unbounded.')

        # Reachable only if M[0] is non-negative (optimal tableau).
        return None

    def pick_dual_pivot(self):
        ''' Picks a pivot to be used by a dual Simplex. '''

        M = self.Mx
        h, w = M.shape

        min_ratio = -1
        i = j = -1

        for x in range(h-1, w-1):
            if M[0, x] < 0:
                raise FormatException(
                    'This LP cannot be solved by dual '
                    'Simplex (lacks optimality condition).'
                )

        for y in range(1, h):
            if M[y, -1] >= 0:
                continue

            for x in range(h-1, w-1):
                if M[y, x] >= 0:
                    continue
                ratio = M[0, x] / -M[y, x]

                if ratio < min_ratio or min_ratio == -1:
                    min_ratio = ratio
                    i = y
                    j = x

            if min_ratio != -1:
                return i, j
            else:
                self.pos_row = y
                raise InfeasibleException('This LP is infeasible.')

        # Reachable only if M[:, -1] >= 0 (optimal tableau).
        return None

    def setup_optimal_result(self, basis_map):
        ''' Sets up the optimal result given a final tableau
            and the basis. '''

        h, w = self.Mx.shape

        # Optimal value.
        self.opt_val = self.Mx[0, -1]

        # Optimal vector.
        self.opt_vec = np.matrix([ff(0)]*w)
        for row in basis_map:
            self.opt_vec[0, basis_map[row]] = self.Mx[row, -1]

        # Delete log and slack entries from optimal vector.
        self.opt_vec = self.opt_vec[0, h-1:-h]

        # Certificate of optimality.
        self.cert = self.Mx[0, :h-1]

    def pivoting_loop(self, pivot_picker, basis_map):
        ''' Generic pivoting loop for all kinds of Simplexes.
            pivot_picker is a function that picks the next pivot.
            basis_map is a dictionary used to keep the current basis map.
        '''

        # Attempt to pick first pivot.
        pivot = pivot_picker()

        while pivot:
            # Update basis map.
            basis_map[pivot[0]] = pivot[1]

            # Pivot matrix.
            self.pivot(pivot[0], pivot[1])
            if self.debug:
                print('[Basis]', basis_map)

            # Pick next pivot.
            pivot = pivot_picker()

    def check_basis_map(self, basis_map):
        ''' Checks if a basis_map really
            corresponds to a canonical basis. '''

        M = self.Mx
        h, w = M.shape

        for pivot in range(1, h):
            col = basis_map[pivot]

            for row in range(1, h):
                r = 1 if row == pivot else 0
                if M[row, col] != r:
                    raise WrongCanonicalBasis(
                        'The set you have provided is not '
                        'a canonical basis.'
                    )

    def apply_primal_simplex(self, basis_map):
        ''' Applies primal simplex to the matrix
            using a feasible basis. '''

        M = self.Mx
        h, w = M.shape

        try:
            # Check input basis
            self.check_basis_map(basis_map)

            # Apply pivoting loop using primal pivot picker
            self.pivoting_loop(self.pick_primal_pivot, basis_map)

            # Setup optimal result
            self.setup_optimal_result(basis_map)

            status = Status.OPTIMAL
        except UnboundedException as e:
            # LP is unbounded.
            # Build unboundedness certificate.
            idx = self.neg_col

            self.cert = np.matrix([ff(0)]*(w-h))
            self.cert[0, idx-h+1] = ff(1)

            for row in basis_map:
                self.cert[0, basis_map[row]-h+1] = -M[row, idx]

            self.cert = self.cert[0, :-h+1]
            status = Status.UNBOUNDED

        return status

    def apply_dual_simplex(self, basis_map):
        ''' Applies dual simplex to the matrix. '''
        M = self.Mx
        h, w = M.shape

        try:
            # Apply pivoting loop using dual pivot picker
            self.pivoting_loop(self.pick_dual_pivot, basis_map)

            # Setup optimal result
            self.setup_optimal_result(basis_map)

            status = Status.OPTIMAL
        except InfeasibleException as e:
            # LP is infeasible.
            # Build infeasibility certificate.
            idx = self.pos_row

            self.cert = M[idx, :h-1]
            status = Status.INFEASIBLE

        return status

    def find_feasible_basis(self):
        ''' Solves an auxiliary LP to find a
            feasible basis for some original LP. '''

        h, w = self.Mx.shape

        # Save original matrix.
        self.Mxp = self.Mx.copy()

        # Make b positive.
        for y in range(1, h):
            if self.Mx[y, -1] < 0:
                self.Mx[y] *= -1

        # Build aux from tableau.
        self.Mx = StdLP.build_aux(self.Mx)

        # Build aux_basis_map and restore canonical basis
        # pivoting the aux columns.
        aux_basis_map = {}
        for y in range(0, h-1):
            aux_basis_map[y+1] = w-1+y
            self.pivot(y+1, aux_basis_map[y+1])

        if self.debug:
            print('\n[Aux LP Input]')
            print(self)

        # Apply primal simplex to aux LP.
        self.apply_primal_simplex(aux_basis_map)

        if self.debug:
            print('\n[Aux LP Output]')
            print(self)

        return aux_basis_map

    def apply_simplex(self, basis_map={}):
        ''' Solves this LP using the Simplex method (dual and primal).

            The basis_map parameter is a row->col map
            representing the canonical basis for the primal Simplex,
            if the LP is in primal format. Otherwise, basis_map is ignored.
        '''

        try:
            dual_basis_map = {}

            # Apply dual simplex.
            status = self.apply_dual_simplex(dual_basis_map)

            basis_map.update(dual_basis_map)
        except FormatException as e:
            try:
                # Apply primal simplex.
                status = self.apply_primal_simplex(basis_map)
            except FormatException as e:
                # Apply simplex to aux LP to obtain feasible basis.
                h, w = self.Mx.shape

                aux_basis_map = self.find_feasible_basis()

                if self.opt_val == 0:
                    # Original LP feasible.
                    # Restore original matrix applying autofix to pivoting.
                    self.Mx = self.Mxp
                    for line in aux_basis_map:
                        col = aux_basis_map[line]
                        self.pivot(line, col, autofix=True)

                    # Apply simplex with canonical basis.
                    status = self.apply_primal_simplex(aux_basis_map)
                    basis_map.update(aux_basis_map)
                else:
                    # Original LP infeasible.

                    # The optimality certificate for our aux LP (self.cert)
                    # already certifies infeasibility for the original LP.
                    status = Status.INFEASIBLE

        if self.debug:
            print('\n[Output]')
            print(self, '\n')

        return status

    def restore_solution_after_restriction(self, basis_map):
        '''
            Adjust tableau to restore solution after
            adding a new restriction.
        '''

        # Adjust map and restore basis.
        for row in basis_map:
            col = basis_map[row]
            self.pivot(row, col)

        if self.debug:
            print('[Basis]', basis_map)

        # Restore solution if needed.
        status = self.apply_dual_simplex(basis_map)
        return status

    def add_restriction(self, restriction_row, basis_map, slack=0):
        '''
            Adds a new restriction to an already found solution.
            slack may be set to 0 (=), 1 (<=) or -1 (>=).
        '''

        if self.opt_val is None:
            return
        h, w = self.Mx.shape

        # Add the restriction and slack variables.
        slack_col = np.matrix(np.zeros(shape=(h+1, 1), dtype='object'))
        slack_col[-1, 0] = ff(slack)

        self.Mx = np.concatenate([self.Mx, restriction_row])
        self.Mx = np.concatenate([
            self.Mx[:, :h-1], slack_col, self.Mx[:, h-1:-1],
            slack_col, self.Mx[:, -1]
        ], axis=1)

        if self.debug:
            print('[Add restriction (' + str(slack) + ')]', StdLP.printable(restriction_row))
            print(self)

        basis_map[h] = w-1
        for row in basis_map:
            basis_map[row] += 1

    @staticmethod
    def all_integer(vec):
        h, w = vec.shape

        for x in range(w):
            if vec[0, x].denominator != 1:
                return False

        return True

    def _apply_bb_rec(self, Mo, basis_map, lvl=0):
        ''' Heavy work for recursive branch and bound. '''

        M = Mo.copy()
        h, w = M.shape

        for row in range(1, h):
            # Skip integer entries.
            if M[row, -1].denominator == 1:
                continue

            # Skip slack variables.
            col = basis_map[row]
            if col >= w-h:
                continue

            # Build 1st restriction
            r1 = M[row, :].copy()
            for x in range(w):
                r1[0, x] = ff(0)
            r1[0, col] = ff(1)
            r1[0, -1] = ff(floor(M[row, -1]))

            print((' ' * lvl) + '[Branch] x_' + str(col-h+1) + ' <= ' + str(r1[0, -1]))

            # Copy basis and input matrix
            basis = dict()
            basis.update(basis_map)
            self.Mx = M.copy()

            # Add 1st restriction
            self.add_restriction(r1, basis, slack=1)
            # Restore basis (optimized, just need to pivot branch var)
            self.pivot(row, basis[row])
            # Restore solution if needed
            status = self.apply_dual_simplex(basis)
            if self.debug:
                print(self)

            # Left branch
            if status == Status.OPTIMAL:
                if not self.bb_best_val or self.opt_val > self.bb_best_val:
                    if StdLP.all_integer(self.opt_vec):
                        # Update solution
                        self.bb_best_val = self.opt_val
                        self.bb_best_vec = self.opt_vec
                    else:
                        # Explore branch
                        self._apply_bb_rec(self.Mx, basis, lvl=lvl+1)
                    print((' ' * lvl) + '[Optimal]', self.bb_best_val)
                else:
                    # Prune branch
                    print((' '*lvl) + '[Pruned]')

            # Build 2nd restriction
            r2 = M[row, :].copy()
            for x in range(w):
                r2[0, x] = ff(0)
            r2[0, col] = ff(1)
            r2[0, -1] = ff(ceil(M[row, -1]))

            print((' ' * lvl) + '[Branch] x_' + str(col-h+1) + ' >= ' + str(r2[0, -1]))

            # Copy basis and input matrix
            basis = dict()
            basis.update(basis_map)
            self.Mx = M.copy()

            # Add 2nd restriction
            self.add_restriction(r2, basis, slack=-1)
            # Restore basis (optimized, just need to pivot branch var and slack)
            self.pivot(row, basis[row])
            self.pivot(h, w)
            # Restore solution if needed
            status = self.apply_dual_simplex(basis)
            if self.debug:
                print(self)
                print('[' + str(status) + ']')

            # Right branch
            if status == Status.OPTIMAL:
                if not self.bb_best_val or self.opt_val > self.bb_best_val:
                    if StdLP.all_integer(self.opt_vec):
                        # Update solution
                        self.bb_best_val = self.opt_val
                        self.bb_best_vec = self.opt_vec
                    else:
                        # Explore branch
                        self._apply_bb_rec(self.Mx, basis, lvl=lvl+1)
                        print((' ' * lvl) + '[Optimal]', self.opt_val)
                else:
                    # Prune branch
                    print((' '*lvl) + '[Pruned]', self.opt_val, '<=', self.bb_best_val)

    def apply_branch_and_bound(self, basis_map):
        ''' Applies the branch and bound method recursively
            to solve the integer programming problem.
        '''

        # Apply Simplex to linear relaxation.
        status = self.apply_simplex(basis_map)
        if status != Status.OPTIMAL:
            return status

        # Save results for linear relaxation.
        self.lr_cert = self.cert.copy()
        self.lr_opt_vec = self.opt_vec.copy()
        self.lr_opt_val = self.opt_val

        if not StdLP.all_integer(self.opt_vec):
            self._apply_bb_rec(self.Mx, basis_map)

            self.opt_val = self.bb_best_val
            self.opt_vec = self.bb_best_vec.copy()
        else:
            self.opt_val = self.lr_opt_val
            self.opt_vec = self.lr_opt_vec.copy()

        return Status.OPTIMAL

    def apply_cutting_plane(self, basis_map):
        ''' Applies the cutting plane method to solve
            the integer programming problem. '''

        # Apply simplex to linear relaxation.
        status = self.apply_simplex(basis_map)

        # Cancel method if not optimal
        if status != Status.OPTIMAL:
            return status

        # Save results for linear relaxation.
        self.lr_cert = self.cert.copy()
        self.lr_opt_vec = self.opt_vec.copy()
        self.lr_opt_val = self.opt_val

        last = status
        integer = False
        c = 1
        while not integer:
            # Get dimensions.
            h, w = self.Mx.shape

            # Assume this is the last iteration of the loop.
            integer = True
            for y in range(1, h):
                # Skip integer entries and slack variables
                if self.Mx[y, -1].denominator == 1 or basis_map[y] >= w-h:
                    continue

                # Break loop assumption.
                integer = False

                # Pick fractional entry.
                row = y

                print('[Cutting plane', c, 'for row '+ str(row) + ']')

                # Build restriction.
                restriction = self.Mx[row, :].copy()
                for x in range(w):
                    if x >= h-1:
                        restriction[0, x] = ff(floor(restriction[0, x]))
                    else:
                        restriction[0, x] = 0

                # Enforce restriction on existing solution.
                self.add_restriction(restriction, basis_map, slack=1)
                last = self.restore_solution_after_restriction(basis_map)
                print('[Optimal]', self.Mx[0, -1])

                if self.debug:
                    print(self)

                c += 1

        return last


if __name__ == '__main__':
    if len(sys.argv) < 2:
        fin = sys.stdin
    else:
        infile = sys.argv[1]
        fin = open(infile)

    # Input.
    m = int(fin.readline())
    n = int(fin.readline())
    matrix_line = fin.readline().strip()

    # Build input matrix & canonical basis.
    matrix = np.matrix(matrix_line, dtype='object').reshape(m+1, n+1)
    for i in range(0, m+1):
        for j in range(0, n+1):
            matrix[i, j] = ff(matrix[i, j])

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
        debug=False, pretty=False
    )

    # Solve
    status = problem.apply_simplex(basis_map)

    # Write simplex output.
    outfile = 'conclusao.txt'
    with open(outfile, 'w') as fout:
        fout.write('%d\n' % status)
        cert = StdLP.printable(problem.cert)

        if status == Status.OPTIMAL:
            opt_vec = StdLP.printable(problem.opt_vec)
            opt_val = problem.opt_val
            print('[+] Optimal found:', opt_val)
            print('[Sol]', opt_vec)
            print('[Cert]', cert)

            fout.write(
                '%s' % opt_vec + '\n'
                '%s' % opt_val + '\n'
                '%s' % cert + '\n'
            )
        elif status == Status.UNBOUNDED:
            print('[x] Unbounded.')
            print('[Cert]', cert)

            fout.write('%s' % cert + '\n')
        elif status == Status.INFEASIBLE:
            print('[x] Infeasible')
            print('[Cert]', cert)

            fout.write('%s' % cert + '\n')

    fin.close()
