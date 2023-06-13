import tk as tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
class OdomPlot():
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.scatter(0,0)
        self.scatter = self.ax.scatter(0, 0)
    def OdomPlot(self,x,y,z):
        # Create a figure and a 3D axis

        # Plot the 3D graph
        self.scatter.remove()
        self.scatter = self.ax.scatter(x, y)
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')
        self.ax.set_title('3D Graph of Positions')
        plt.draw()
        plt.pause(0.0001)

        # Display the plot
        #plt.show()