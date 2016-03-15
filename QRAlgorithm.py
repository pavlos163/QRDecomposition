import Utils

## Main program ##

matrix = Utils.loadMatrix()
if not Utils.isSymmetric(matrix):
	raise ValueError("Not a symmetric matrix.")
Q, R = Utils.QRdecompose(matrix)
eVectors, eValues = Utils.computeVV(Q, R)
Utils.outputResults(eVectors, eValues)
