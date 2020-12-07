from svm import *
from utils import *

def test():
    '''test SVM with different kernels'''
    x, y = load_csv('data/data2.csv', add_intercept=False)
    y[y == 0] = -1
    y_ = (y==1)
    
    
    kernels = [('poly',{'order': 2, 'bias': 1}), ('rbf',{'radius': 0.5})]
    
    for kernel in kernels:
        model = SVM(C = 5, tol=1e-4, kernel = kernel[0])
        model.fit(x, y, **kernel[1])
        
        plt.figure(figsize=(12, 8))
        plot_contour(lambda x: model.predict(x))
        plot_points(x, y_)
        plt.savefig('data/result_of_'+kernel[0]+'_kernel.png')
        plt.show()

if __name__ == "__main__":
	test()