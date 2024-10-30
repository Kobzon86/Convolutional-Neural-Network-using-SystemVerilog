# Convolutional-Neural-Network-using-SystemVerilog
Synthesizable RTL-Based video stream Convolutional Neural Network ( non HLS )

The testbench's images are taken from MNIST digits dataset.

![image](https://github.com/user-attachments/assets/447386e3-ac5d-4a59-b600-3b1323c77b01)



All weights and reference calculated by "cnn_behind.py"  

Numbers and size of Convolution and Fully Connected layers are parameterizable.

In testbench I used this CNN structure: 

        conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)
        conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0)
        fc1 = nn.Linear(200, 64)
        fc2 = nn.Linear(64, 10)


Simulation Results: 
![image](https://github.com/user-attachments/assets/6dc68ca5-9127-4118-94c8-77288268b2d6)
