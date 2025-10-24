The related document is https://www.vizuaranewsletter.com/p/teach-your-neural-network-to-respect?utm_campaign=post&utm_medium=web.

The Combined Loss Function is 
<img width="805" height="225" alt="image" src="https://github.com/user-attachments/assets/78a53e0a-079b-488a-98a8-7fb086d5c12c" />

The different cases where lambda is set are
Case 1
When the Loss function of data is set and the Loss of ODE and Loss of initial function is set to Zero , then overfitting is noticed
<img width="1142" height="522" alt="image" src="https://github.com/user-attachments/assets/54d90f74-e013-43bf-b6a4-346016ed97b7" />
the related graph is 
<img width="800" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/f2b6fdfc-3b72-4181-8b34-fd6aa436c165" />

Case 2
When the Loss function of data is set to zero and the the other two are set to positive values. This is the case when are 100% sure that the system follows the laws of phsyics and there are no external factors or error chances. 
<img width="1136" height="525" alt="image" src="https://github.com/user-attachments/assets/41f50c9d-69ce-4a86-a39b-e174ab42e5c5" />
The related graph is 
<img width="800" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/855fdd80-8367-46b5-83e0-0036d0b6baab" />

Case 3
When the all the loss functions are set to check the PINN case
<img width="1153" height="521" alt="image" src="https://github.com/user-attachments/assets/70716b18-f5f5-40ea-9bdf-61913a99bee4" />
the related graph is 
<img width="806" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/ed727796-9692-457b-8ba0-9f115e29fcd9" />
