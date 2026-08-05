# Neste estudo sobre as Redes Neurais, eu resolvo o famoso problema da porta XOR, onde esta estará ligada somente se suas entradas forem diferentes, Ex: [0 , 1] ou [1 e 0].

## Materiais Utilizados:
1. **"Deep Learning"** - *Ian Goodfellow, Yoshua Bengio e Aaron Courville*
2. **"Inteligência Artificial: Uma Abordagem Moderna" (AIMA)** - *Stuart Russell, Peter Norvig*
3. **The spelled-out intro to neural networks and backpropagation: building micrograd** - *Andrej Karpathy*

Isso é feito através de **backpropagation**, que é uma técnica de cálculo da derivada do erro relacionado à cada peso, e é feito o ajuste dos pesos e biases(viés) pelo **gradient descent**. Também com a ideia do feedforward, usando a soma ponderada com a função de ativação(neste caso, a função sigmóide). Eu implementei um **MLP (multi-layer perceptron)** do zero usando apenas o numpy e uma matemática de ensino superior bem tranquila! (ps: não tão tranquila assimkkkk), cada neurônio é retroalimentado com o output do anterior, e assim sucessivamente.

No fim, há um treinamento com a **taxa de aprendizado ($\eta$): 0,5** e o uso de épocas(o ciclo feedforward -> erro -> backpropagation -> atualização dos pesos), em detalhes é basicamente o cálculo do gradiente na camada de saída (nesse caso:  $$\frac{d\sigma(z)}{dz} = \sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$) e faz um dot product com o erro, em notação matemática:

$$\delta_2 = \frac{\partial E}{\partial Z_2} = (A_2 - y) \odot [A_2 \cdot (1 - A_2)]$$

Logo após propaga o erro de volta para a camada oculta:

$$\delta_1 = (\delta_2 \cdot W_2^T) \odot [A_1 \cdot (1 - A_1)]$$

Por fim, há a atualização dos pesos com base no gradient descent, onde visa achar o mínimo local, pela minimização da loss function.
