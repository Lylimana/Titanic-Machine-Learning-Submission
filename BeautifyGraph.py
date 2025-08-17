import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def BeautifyGraph(array_X,array_Y,GraphTitle: str,x_label: str,y_label: str)-> None: 
    plt.plot(array_X,array_Y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(GraphTitle)
    plt.show()
    
    
def main()-> None: 
    BeautifyGraph()
    
    
if __name__ == '__main__': 
    main()