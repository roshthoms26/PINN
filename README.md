Overview

The related document is https://www.vizuaranewsletter.com/p/teach-your-neural-network-to-respect?utm_campaign=post&utm_medium=web.

The Core idea of this is that one can imagine teaching a child what will happen if they drop a ball. They don't need to see a thousand videos to learn that it will fall, not float up. However, in the case of an Artificial Intelligence (AI) model, it would need to analyze millions of pictures and videos of falling objects, and even after that, it is prone to making mistakes. It learns from patterns in data, but doesn't understand the fundamental rules of our world. This central problem is that standard neural networks are excellent pattern recognizers, but are ignorant of the basic laws of physics, which is where the application of PINN, or Physics-Informed Neural Network, comes in. The goal is to "teach" AI models to respect the laws of physics, making them more accurate, efficient, and trustworthy, especially for scientific and engineering tasks.
Traditional AI has two major weaknesses when applied to real-world scientific problems. One of them is Data hunger, which is that training an AI model to simulate a complex system requires a large amount of high-quality data. This is often expensive, time-consuming, or sometimes impossible to collect. For example, we can't run a million crash tests to get data for a car safety AI. The other issue is that the models only understand statistical correlations in data; they can produce results that are completely impossible in the real world. These "unphysical" results make them unreliable for critical applications. Hence, we can argue that we can't rely on data alone. We need to infuse our models with the centuries of scientific knowledge we already possess.
Therefore, teaching a Neural network to "respect" physics can be done in two ways
> Data score: This is a traditional method. We give the AI some real-world data and penalize it when its predictions don't match the experimental data.
> The Physics Score: This is the new method. We give the AI the known physics equations that govern the system( like the laws of motion or fluid dynamics) during training, and the AI is also penalized whenever its prediction violates a fundamental equation.

Example

Here, in this example https://www.youtube.com/watch?v=1AyAia_NZhQ, we consider a hypothetical set of experiments. We throw a ball up at a certain angle and note down the height of the ball at different points in time. Then we train a neural network on this dataset so that we can predict the height of the ball even at time points where we don't know.
Hence, when we plot Height vs Time, we get
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/80caecf1-d224-4f9a-b6bd-859e91f056a0" />
We can construct a neural network with a few or multiple hidden layers. The input is time(t), and the output predicted by the neural network is the height of the ball (h). This NN will be initialized with random weights. Hence, the predictions by the NN for h(t) will be bad, or the loss will be more, as plotted below
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/d00387bf-d933-4fac-a19b-7f791bb77a2a" />
Hence, we need to penalize the neural network in the form of Loss functions, which is given by the Mean Squared Error equation:

<img width="500" height="280" alt="image" src="https://github.com/user-attachments/assets/21dad1d7-50b6-4b61-8b2f-11b6fa7e51a4" />
Hence, when the loss functions are added, we will get a perfectly fitting prediction of the model, which, in other words, over-fitting. This over-fitting can result in unrealistic predictions, which is shown below.

<img width="500" height="475" alt="image" src="https://github.com/user-attachments/assets/3df25214-658f-4d37-a1a9-ecc55f1861a9" />

Hence, we bring Physics into the picture in this case, by adding Newton's laws of motion. This says that we may be making simplifications by assuming that the effect of wind, air drag, or buoyancy is negligible. But that does not take away that we have a decent knowledge about this system, even in the absence of a trained neural network. In other words, the physics we assume may not be in perfect agreement with the experimental data, but it makes sense to think that the experiments will not deviate too much from the Physics, which is shown below:

<img width="500" height="475" alt="image" src="https://github.com/user-attachments/assets/24832813-f697-4ae1-b3f2-b062ac1e78e8" />

Hence, we can implement PINN into our example. Hence, when a ball is thrown up, its trajectory varies according to the following Ordinary Differential Equation(ODE) 
dh/dt = u₀ - gt
Where:
• dh/dt = Rate of change of height with respect to time (velocity)
• u₀ = Initial velocity
• g = Acceleration due to gravity
• t = Time

However, ODE alone cannot fully describe h(t). We also need the Initial condition. Hence, knowing height as a function of time, we need to know the starting height from which the ball was thrown. So we need to know h(t=0) for fully describing the height of the ball as a function of time. We know the expected dh/dt because we know the initial velocity and acceleration due to gravity. After all, it is predicting height h, not velocity v or dh/dt. The answer is Automatic differentiation (AD). Most of the Machine learning Frameworks like Tensorflow, Pytorch, and JAX support Automatic differentiation. Thus, we have a predicted dh/dt for every experimental time point, and we have an actual dh/dt based on the physics.

<img width="500" height="310" alt="image" src="https://github.com/user-attachments/assets/85b50bfa-a9fb-4b89-9fcc-8a9eecc991b7" />

Hence, the loss due to the difference between the predicted and physics-based dh/dt is

 <img width="500" height="360" alt="image" src="https://github.com/user-attachments/assets/567dce62-0015-4c34-8902-be6828dc1fd9" />

The Initial Condition loss is given by

<img width="500" height="412" alt="image" src="https://github.com/user-attachments/assets/cc7686c0-ac12-423d-aded-9c2e6896add8" />

Hence, the Combined Loss Function is 

<img width="500" height="225" alt="image" src="https://github.com/user-attachments/assets/78a53e0a-079b-488a-98a8-7fb086d5c12c" />

The different cases where lambda is set:

Case 1:
When the Loss function of data is set and the Loss of ODE and the Loss of the initial function are set to zero, then overfitting is noticed
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/54d90f74-e013-43bf-b6a4-346016ed97b7" />
The related graph is 
<img width="800" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/f2b6fdfc-3b72-4181-8b34-fd6aa436c165" />

Case 2: 
When the Loss function of data is set to zero, and the other two are set to positive values. This is the case when we are 100% sure that the system follows the laws of physics and there are no external factors or error chances. 
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/41f50c9d-69ce-4a86-a39b-e174ab42e5c5" />
The related graph is 
<img width="800" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/855fdd80-8367-46b5-83e0-0036d0b6baab" />

Case 3: 
When all the loss functions are set to check the PINN case
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/70716b18-f5f5-40ea-9bdf-61913a99bee4" />
The related graph is 
<img width="800" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/ed727796-9692-457b-8ba0-9f115e29fcd9" />

Conclusion

The potential of PINNs is indeed undeniable, but they can be computationally complex and sometimes tricky to train stably. Choosing the right physics laws to enforce is also crucial. They can represent a fundamental shift in how we build AI. Instead of creating pure models that learn only from data, we are moving towards AI partners that integrate our human scientific knowledge. This creates models that are More data-efficient, interpretable, robust, and reliable.
