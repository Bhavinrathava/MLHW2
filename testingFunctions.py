import starter
import numpy as np

def testEuDistance():
    a = np.array([1,2,1,1,2,3])
    b = np.array([2,6,7,8,5,6])

    euFunctionVal = starter.euclidean(a,b)
    npVal = np.linalg.norm(a-b)

    if(euFunctionVal == npVal):
        print("The Euclidean Distance test was successful !")

if __name__=="__main__":
    testEuDistance()
