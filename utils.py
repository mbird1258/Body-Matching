import numpy as np

COLORS = [(105, 105, 105), 
          ( 85, 107,  47),
          (127,   0,   0),
          ( 72,  61, 139),
          (  0, 128,   0),
          (  0, 139, 139),
          (  0,   0, 128),
          (210, 105,  30),
          (218, 165,  32),
          (143, 188, 143),
          (139,   0, 139),
          (176,  48,  96),
          (255,   0,   0),
          (255, 255,   0),
          (222, 184, 135),
          (  0, 255,   0),
          (138,  43, 226),
          (  0, 255, 127),
          (220,  20,  60),
          (  0, 255, 255),
          (  0, 191, 255),
          (  0,   0, 255),
          (240, 128, 128),
          (173, 255,  47),
          (176, 196, 222),
          (255,   0, 255),
          ( 30, 144, 255),
          (144, 238, 144),
          (255,  20, 147),
          (238, 130, 238)]

def SphereCenter2(xArr, yArr, zArr):
    # Source: https://jekel.me/2015/Least-Squares-Sphere-Fit/
    # fit a sphere to X,Y, and Z data points
    # returns the radius and center points of
    # the best fit sphere

    #   Assemble the A matrix
    xArr = np.array(xArr)
    yArr = np.array(yArr)
    zArr = np.array(zArr)
    A = np.zeros((len(xArr),4))
    A[:,0] = xArr*2
    A[:,1] = yArr*2
    A[:,2] = zArr*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(xArr),1))
    f[:,0] = (xArr*xArr) + (yArr*yArr) + (zArr*zArr)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = np.sqrt(t)

    return radius, C[0], C[1], C[2]

def SphereCenter(xArr, yArr, zArr):
    matrix1 = 2*np.hstack((xArr[0]-xArr[1:, np.newaxis], yArr[0]-yArr[1:, np.newaxis], zArr[0]-zArr[1:, np.newaxis]))
    matrix2 = (xArr[0]**2+yArr[0]**2+zArr[0]**2)-np.sum([xArr[1:]**2, yArr[1:]**2, zArr[1:]**2], axis=0)[:, np.newaxis]
    x, y, z = np.linalg.solve(matrix1, matrix2).flatten()
    r = np.sqrt((x-xArr[0])**2+(y-yArr[0])**2+(z-zArr[0])**2)

    return x, y, z, r

def CircleCenter(xArr, yArr):
    matrix1 = 2*np.hstack((xArr[0]-xArr[1:, np.newaxis], yArr[0]-yArr[1:, np.newaxis]))
    matrix2 = (xArr[0]**2+yArr[0]**2)-np.sum([xArr[1:]**2, yArr[1:]**2], axis=0)[:, np.newaxis]
    x, y = np.linalg.solve(matrix1, matrix2).flatten()
    r = np.sqrt((x-xArr[0])**2+(y-yArr[0])**2)
    
    return (x, y), r

def PointsInCircle(radius, x0, y0, shape=None):
    if shape:
        x_ = np.arange(max(x0 - radius - 1, 0), min(x0 + radius + 1, shape[0]-1), dtype=int)
        y_ = np.arange(max(y0 - radius - 1, 0), min(y0 + radius + 1, shape[1]-1), dtype=int)
        x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2)
    else:
        x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
        y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
        x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2)
    
    for x, y in zip(x_[x], y_[y]):
        yield x, y

def d(p1, p2, p3, p4):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    x4, y4, z4 = p4

    return (x1-x2)*(x3-x4)+(y1-y2)*(y3-y4)+(z1-z2)*(z3-z4)

def intersection(p1, p2, p3, p4, GetError = False):
    """
    p --> point
    source: Paul Bourke
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    x4, y4, z4 = p4

    mu_a = (d(p1, p3, p4, p3)*d(p4, p3, p2, p1)-d(p1, p3, p2, p1)*d(p4, p3, p4, p3))/(d(p2, p1, p2, p1)*d(p4, p3, p4, p3)-d(p4, p3, p2, p1)**2)
    mu_b = (d(p1, p3, p4, p3)+mu_a*d(p4, p3, p2, p1))/d(p4, p3, p4, p3)

    x = (x1+mu_a*(x2-x1)+x3+mu_b*(x4-x3))/2
    y = (y1+mu_a*(y2-y1)+y3+mu_b*(y4-y3))/2
    z = (z1+mu_a*(z2-z1)+z3+mu_b*(z4-z3))/2

    if GetError:
        dx = (x1+mu_a*(x2-x1)) - (x3+mu_b*(x4-x3))
        dy = (y1+mu_a*(y2-y1)) - (y3+mu_b*(y4-y3))
        dz = (z1+mu_a*(z2-z1)) - (z3+mu_b*(z4-z3))
        dist = np.sqrt(dx**2+dy**2+dz**2)
        return (x, y, z), dist

    return x, y, z

def ransac(input, InlierFunction, threshold=1, count=np.inf, iterations=np.inf, samples=None, axes=None, args=None):
    shape = np.array(input.shape)
    
    iteration = 0
    BestInliers, BestModel = [], None
    
    while True:
        if axes:
            randind = list(np.random.randint(0, shape[axes], np.append([samples], shape[axes].shape).astype(np.int32)).T)
        else:
            randind = list(np.random.randint(0, shape, np.append([samples], shape.shape)).T)
        
        ind = [slice(None)]*input.ndim
        if axes:
            for i, ax in enumerate(axes):
                ind[ax] = randind[i]
        
        points = input[*ind]
        
        
        if args:
            inliers, model = InlierFunction(input, points, threshold, *args)
        else:
            inliers, model = InlierFunction(input, points, threshold)
        
        if len(inliers) >= len(BestInliers):
            BestInliers = inliers
            BestModel = model

        iteration += 1
        if len(BestInliers) >= count or iteration >= iterations:
            return BestInliers, BestModel

from rtmlib import RTMO

device = 'cpu'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
BodyPoseDetector = RTMO(onnx_model='end2end.onnx',  # download link or local path
                        backend=backend, device=device)

def BodyPose(img):
    DetectionResult = BodyPoseDetector(img)
    return DetectionResult[0]

def BodyPoseConnections():
    return ((1, 2),
            (0, 1),
            (1, 3),
            (3, 5),
            (5, 7),
            (7, 9),
            (0, 2),
            (2, 4),
            (4, 6),
            (6, 8),
            (8, 10),
            (5, 6),
            (6, 12),
            (12, 11),
            (11, 5),
            (12, 14),
            (14, 16),
            (11, 13),
            (13, 15))

def BoneLength(ind1, ind2):
    match (ind1,ind2):
        case (5,6):
            return 40

        case (11,12):
            return 14
        
        case _:
            return False

def GetCourtConnections():
    return ((0, 1), 
            (0, 2), 
            (1, 3), 
            (2, 3), 
            (2, 4), 
            (3, 5), 
            (4, 5), 
            (4, 6), 
            (5, 7), 
            (6, 7), 
            (6, 8), 
            (7, 9), 
            (8, 9), 
            (10, 12), 
            (11, 13), 
            (12, 13), 
            (12, 14), 
            (13, 15), 
            (14, 15))

def RotationMatrix3D(pitch, yaw, roll):
    return np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                     [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                     [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])

def CourtLocalCoordinates(NetHeight):
    return np.array([[0   , 0   , 0            ],
                     [900 , 0   , 0            ],
                     [0   , 600 , 0            ],
                     [900 , 600 , 0            ],
                     [0   , 900 , 0            ],
                     [900 , 900 , 0            ],
                     [0   , 1200, 0            ],
                     [900 , 1200, 0            ],
                     [0   , 1800, 0            ],
                     [900 , 1800, 0            ],
                     [-91 , 900 , 0            ],
                     [991 , 900 , 0            ],
                     [-91 , 900 , NetHeight-100],
                     [991 , 900 , NetHeight-100],
                     [-91 , 900 , NetHeight    ],
                     [991 , 900 , NetHeight    ]])

def GetPlaneNorm(p1, p2, p3):
    return np.cross(p2-p1, p3-p1)

def GetVectorVectorRotationMatrix(vec1, vec2):
    # Source: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    RotationMatrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return RotationMatrix

def HomographyMatrix(x0, y0, x1, y1, x2, y2, x3, y3, X0, Y0, X1, Y1, X2, Y2, X3, Y3):
    return np.linalg.solve(np.array([[-x0, -y0, -1,  0 ,  0 ,  0, x0*X0, y0*X0, X0], 
                                     [ 0 ,  0 ,  0, -x0, -y0, -1, x0*Y0, y0*Y0, Y0],
                                     [-x1, -y1, -1,  0 ,  0 ,  0, x1*X1, y1*X1, X1], 
                                     [ 0 ,  0 ,  0, -x1, -y1, -1, x1*Y1, y1*Y1, Y1],
                                     [-x2, -y2, -1,  0 ,  0 ,  0, x2*X2, y2*X2, X2], 
                                     [ 0 ,  0 ,  0, -x2, -y2, -1, x2*Y2, y2*Y2, Y2],
                                     [-x3, -y3, -1,  0 ,  0 ,  0, x3*X3, y3*X3, X3], 
                                     [ 0 ,  0 ,  0, -x3, -y3, -1, x3*Y3, y3*Y3, Y3],
                                     [ 0 ,  0 ,  0,  0 ,  0 ,  0, 0    , 0    , 1 ]]),

                           np.array([[0], 
                                     [0], 
                                     [0], 
                                     [0], 
                                     [0], 
                                     [0],  
                                     [0], 
                                     [0], 
                                     [1]])).reshape(3, 3)