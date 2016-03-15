from math import sqrt
import numpy as np

EPSILON = 0.000001

# Reads a matrix from a file and returns that matrix. #
def loadMatrix():
	f = open('input.txt')
	size = int(f.readline());
	matrix = np.zeros((size, size))
	for i in range(size):
		matrix[i][:] = [float(num) for num in f.readline().split(" ")]
	f.close()
	return matrix

# Checks if the matrix is symmetric. #
def isSymmetric(matrix):
	size = len(matrix)
	return np.allclose(transpose(matrix), matrix)

# Returns the transpose of a given matrix. #
def transpose(matrix):
	size = len(matrix)
	return [[matrix[i][j] for i in range(size)] for j in range(size)]

# Returns the Euclidian Norm of a given matrix. #
def euclidianNorm(v):
	size = len(v)
	elemSum = 0
	for i in range(size):
		elemSum += v[i]**2
	return sqrt(elemSum)

# Function that implements the QR Decomposition.  #
# Returns Q, an orthogonal matrix and R, an upper #
# triangular matrix such that A = QR.             #
def QRdecompose(matrix):
	size = len(matrix)
	Q = np.zeros((size, size))
	R = np.zeros((size, size))
	for i in range(size):
		v = matrix[:,i]
		for j in range(i):
			R[j,i] = np.dot(Q[:,j], matrix[:,i])
			v = v - np.dot(R[j,i], (Q[:,j]))
		R[i,i] = euclidianNorm(v)
		Q[:,i] = v / R[i,i]
	return Q, R

# Returns true when the computation of the eigenValues #
# of a matrix are accurate enough to stop.             #
def accurateResult(matrix):
	size = len(matrix)
	for i in range(size):
		for j in [x for x in range(size) if x != i]:
			if matrix[i][j] >= EPSILON:
				return False
	return True

# Returns the diagonal of the square matrix given. #
def getMatrixDiagonal(matrix):
	size = len(matrix)
	return [matrix[i][i] for i in range(size)]

# Function that implements the QR iterations to   #
# compute the eigenvalues and eigenvectors of a   #
# matrix. Instead of taking the matrix as an      #
# argument, the function takes matrices Q and R   #
# which result from A after the QR decomposition. #
def computeVV(Q, R):
	eVectors = Q
	A = np.dot(R, Q)
	while not accurateResult(A):
		A = np.dot(R, Q)
		Q, R = QRdecompose(A)
		eVectors = np.dot(eVectors, Q)
	eValues = getMatrixDiagonal(A)
	return eVectors, eValues

# Outputs the given arguments to "output.txt". #
def outputResults(eVectors, eValues):
	f = open('output.txt', 'w')
	f.write("EIGENVECTORS:\n")
	f.write(str(eVectors) + "\n\n")
	f.write("-----------------------\n\n")
	f.write("EIGENVALUES:\n")
	f.write(str(eValues))
	f.close()
