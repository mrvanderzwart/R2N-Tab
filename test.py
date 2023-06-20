'''
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
 
 
# Creating dataset
data = [[34, 44, 41, 47, 45, 38, 37, 33, 50, 40],
        [57, 52, 58, 63, 58, 60, 55, 61, 58, 50],
        [105, 100, 100, 102, 102, 102, 99, 98, 96, 100],
        [105, 100, 100, 102, 102, 102, 99, 98, 96, 100]]
data = np.array(data)

print(data)
 
fig = plt.figure(figsize =(10, 7))
 
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(data)
 
# show plot
plt.show()
'''
import matplotlib.pyplot as plt
import numpy as np

data_adult = [[34, 44, 41, 47, 45, 38, 37, 33, 50, 40],
              [57, 52, 58, 63, 58, 60, 55, 61, 58, 50],
              [105, 100, 100, 102, 102, 102, 99, 98, 96, 100]]
 
plt.boxplot([data_adult[0], data_adult[1], data_adult[2]], labels=['cancelrate 1e-5', 'cancelrate 1e-4', 'cancelrate 1e-3'])
plt.grid(axis='y', alpha=0.5)
plt.ylabel('# rules cancelled')
plt.title('# rules cancelled per cancelrate for the adult dataset')
plt.savefig('boxplotadult.png')
plt.show()

data_heloc = [[30, 31, 31, 32, 33, 30, 30, 33, 30, 29],
              [32, 31, 31, 30, 30, 30, 30, 32, 29, 30],
              [32, 32, 34, 33, 35, 33, 32, 34, 33, 33]]
              
plt.boxplot([data_heloc[0], data_heloc[1], data_heloc[2]], labels=['cancelrate 1e-5', 'cancelrate 1e-4', 'cancelrate 1e-3'])
plt.grid(axis='y', alpha=0.5)
plt.ylabel('# rules cancelled')
plt.title('# rules cancelled per cancelrate for the heloc dataset')
plt.savefig('boxplotheloc.png')
plt.show()

data_house = [[48, 48, 52, 47, 43, 47, 45, 46, 45, 46],
              [49, 45, 45, 45, 50, 49, 47, 46, 46, 52],
              [53, 54, 53, 57, 48, 51, 51, 55, 46, 54]]
              
plt.boxplot([data_house[0], data_house[1], data_house[2]], labels=['cancelrate 1e-5', 'cancelrate 1e-4', 'cancelrate 1e-3'])
plt.grid(axis='y', alpha=0.5)
plt.ylabel('# rules cancelled')
plt.title('# rules cancelled per cancelrate for the house dataset')
plt.savefig('boxplothouse.png')
plt.show()

data_magic = [[20, 16, 23, 21, 19, 21, 16, 19, 21, 19],
              [19, 18, 20, 25, 22, 17, 21, 19, 19, 19],
              [28, 30, 25, 27, 28, 27, 31, 31, 27, 25]]
              
plt.boxplot([data_magic[0], data_magic[1], data_magic[2]], labels=['cancelrate 1e-5', 'cancelrate 1e-4', 'cancelrate 1e-3'])
plt.grid(axis='y', alpha=0.5)
plt.ylabel('# rules cancelled')
plt.title('# rules cancelled per cancelrate for the magic dataset')
plt.savefig('boxplotmagic.png')
plt.show()
