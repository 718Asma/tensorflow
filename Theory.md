<style>
   * {font-family: Bookman Old Style; text-align: justify;}
   H1 {font-weight: 600; font-style: italic; font-size: 35px;}
   H2 {font-weight: 500; margin-left: 20px; font-size: 28px;}
   H3 {font-weight: 400; margin-left: 40px; font-size: 22px;}
   H4 {font-weight: 300; margin-left: 60px; font-size: 18px;}
   H5 {font-size: 18px;}
   SPAN {font-weight: bold; color:#00A487;}
   LI {margin-left: 60px; font-size: 16px; color: 'white';} 
   P {margin-left: 20px;}
</style>

# Math

-   <span>Scalar:</span> A single number (0D): `5`
-   <span>Vector:</span> A one-dimensional array (1D): `[1, 2, 3]`
-   <span>Matrix:</span> A two-dimensional array (2D): `[[1, 2], [3, 4]]`
-   <span>Tensor:</span> Can extend to higher dimensions (>= 3D): `[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]`

# TensorFlow (TF)

## Key Concepts

-   <span>Tensors:</span> Containers for data.
-   <span>Operations:</span> These are the fundamental computations TF performs on tensors. They can be basic (addition, matrix multiplication) or complex (convolution, activation functions).
-   <span>Graphs:</span> TF builds a computational graph—a roadmap that defines how data flows through operations.
<br/>&emsp;In graph execution mode, TF first constructs this blueprint and then runs it efficiently.
<br/>&emsp;In eager execution mode, computations run immediately, making debugging easier.

## How it works (Basics)

1. **Prepare the Data:** Set of data TF learns from.
   <br/><p>e.g.: Images, text, or numbers</p>
2. **Build a Model:** Set of instructions to learn patterns from the data.
   <br/><p>e.g.: A model to tell cats from dogs learns shapes, colors and patterns</p>
3. **Train the Model:** TF looks at the data, makes guesses and adjusts itself to get better.
4. **Use the Model:** Once trained, it can make predictions.
   <br/><p>e.g.: Identify pictures as cats or dogs</p>

## Components
- ##### Think of a neural network as a simplified version of a human brain. It’s made up of layers, and each layer contains neurons.

### Layers: Organized Groups of Neurons

-   Layers are groups of neurons stacked together to process data in stages. Each layer learns something different about the input data.

1. **Input Layer:** The first layer that takes raw data and passes it to the next layer
   <br/><p>e.g.: In an image classification model, each pixel value in the image is an input.</p>
2. **Hidden Layer(s):** The *"processing"* layer(s) that take input, perform calculations and pass the results to the next layer.
   <br/><p>The more layers and neurons, the more complex patterns the model can learn.
   <br/> e.g.: One layer might learn edges in an image, the next might learn shapes, and so on.</p>
3. **Output Layer:** The last layer gives the final prediction.
   <br/><p>e.g.: For digit recognition, the output layer has 10 neurons (one for each digit 0–9). The neuron with the highest value is the predicted digit.</p>

### Neurons: The Building Blocks

-   <div style="color: red;"> &emsp;Neurons on the same do NOT communicate with each other! </div>
-   A neuron is like a tiny decision-maker.
-   How they work:
    <br/>&ensp;1. **Take input:** Numbers from the previous layer (or raw data for the first layer).
    <br/>&ensp;2. **Process the inputs:** Multiplies each input by a weight, adds a bias, and then passes the result through an activation function.
    <br/>&ensp;3. **Output a value:** Sends the processed numbers to the next layer.
    <br/>&emsp;&emsp;The output of a neuron can be written as:
    <br/><center>*Output = **Activation Function**(( w<sub>1</sub> * x<sub>1</sub> ) + ( w<sub>2</sub> * x<sub>2</sub> ) + ... + ( w<sub>n</sub> * x<sub>n</sub> ) + b)*</center>

- **Weights (w<sub>n</sub>):** How much influence each input has on a neuron's output.
<br/><p>&emsp;&emsp;They allow the model to "learn" which features matter most.</p>
- **Bias (b):** A baseline value added to the weighted sum.
<br/><p>&emsp;&emsp;It shifts the output up or down, ensuring the neuron can make predictions even if all inputs are zero.
<br/>&emsp;&emsp;It helps the model handle cases where data relationships are more complex.</p>


### Activation Function: Decides whether or not the neuron "fires"
- It allows the model to learn complex non-linear patterns.
- Without it, the model would just perform linear calculations.


#### Common Activation Functions

- <span>ReLU (Rectified Linear Unit):</span> Outputs 0 for negative values and the input itself for positive values.
  - Example: If x = 2, ReLU gives 2, but if x = −1, ReLU gives 0.
  - Formula: `ReLU(x) = max(0, x)`

- <span>Sigmoid:</span> Outputs a value between 0 and 1.
  - Formula: `Sigmoid(x) = 1 / (1 + e^(-x))`
  - Example: If x = 2, the output might be 0.88; if x = −2, the output is 0.12.

- <span>Softmax:</span> Converts raw outputs into probabilities that sum to 1. Used for multi-class classification.
  - Example: For inputs `[2, 1, 0]`, Softmax might output `[0.7, 0.2, 0.1]`, meaning the first class has a 70% chance.

### Loss Function: Measuring Mistakes

- A *loss function* measures how far the model's predictions are from the actual value
- It helps the model learn by giving feedback on its errors.

#### Types of Loss Functions
   - Where:
      - *$y_i$* is the actual value.
      - *$\hat{y}_i$* is the predicted value.
      - <span>*n*</span> is the number of samples.

- <span>Regression Loss Functions:</span> Used to predict continuous values.

   1. **Mean Squared Error (MSE)**
      - **Formula:** *$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$*
      - Squaring the errors has the effect of penalizing larger mistakes more heavily than smaller ones.
      It’s useful when you want to put more emphasis on correcting big errors, especially in cases where large deviations are particularly undesirable.
      - **Example:** Predicting house prices; if the predicted price is far off, MSE gives a large penalty.

   2. **Mean Absolute Error (MAE)**
      - **Formula:** *$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$*
      - It treats all errors equally, meaning that every prediction error contributes to the overall error in the same way, regardless of its size or direction (whether it's overestimated or underestimated).
      - **Example:** Predicting house prices; if the predicted price is off by $10,000, MAE gives a penalty of $10,000, no matter the direction of the error.

- <span>Classification Loss Functions:</span> Used when predicting categories
   1. Binary Cross-Entropy **(BSE)** (Log Loss) (For Binary Classification)
      - **Formula:** *$Loss = -\frac{1}{n} \sum_{i=1}^{n} [y_i log(\hat{y}_i) + (1- y_i) log(1 - \hat{y}_i)]$*
      - If the true label is 1, it penalizes small predicted probabilities ($\hat{y}$).
      - **Example:** Classifying emails as spam or not spam. If the model predicts a low probability for an email being spam but it is actually spam, the loss increases.
   
   2. Categorical Cross-Entropy **(CCE)** (For Multi-class Classification)
      - **Formula:** *$Loss = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})$*
         - Where:
            - $C$ is the number of classes.
            - $y_{ic}$ is the true label for class $c$ (1 if the sample belongs to class $c$, 0 otherwise).
            - $\hat{y}_{ic}$ is the predicted probability for class $c$.
      - Works similarly to binary cross-entropy but for multiple categories.

      - **Example:** Classifying images of animals. If the true label is "dog" but the model predicts "cat," it will penalize the model according to how close the predicted probability for "cat" is to the actual label "dog."

### Optimization: Improving the Model

- Optimization is about adjusting the model’s parameters *(weights and biases)* to minimize the loss function.

