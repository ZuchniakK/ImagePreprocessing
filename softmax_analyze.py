import numpy as np
import matplotlib.pyplot as plt

def softmax_analyzer(softmax_file):
    data = np.load(softmax_file)
    print data.shape
    x = []
    y=[]
    k=0
    for dat in data:
        y.extend([k for i in range(94720)])
        k+=1
        yyy=[]
        for d in dat:
            x.extend(d)
            yyy.extend(d)
        plt.hist(yyy,bins=40)
        print d.shape
        print len(x)
    print (len(x), len(y))
    # plt.hist2d(x,y,bins=15)
    plt.show()

folder_name = 'params/'
file_name = 'softmax_test_values_vs_time_cifar10_cropp_to_24_resized_m0_c32_c32_f192_f96_e2.1_b128_tf0.2_tp0.95.npy'
softmax_analyzer(folder_name+file_name)



