v = 0.25
z = -0.25
test =np.array([
[v,v,v,v,v,v,v,v],
[z,z,z,z,z,z,z,z],
[v,v,v,v,v,v,v,v],
[z,z,z,z,z,z,z,z],
[v,v,v,v,v,v,v,v],
[z,z,z,z,z,z,z,z],
[v,v,v,v,v,v,v,v],
[z,z,z,z,z,z,z,z]
])
print(cv.dct(test))
