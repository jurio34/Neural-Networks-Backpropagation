
import numpy as np
from engine import NeuralNetwork, mean_squared_error


def main():
    # Conjunto de dados (Porta Lógica XOR)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    # Configuração dos hiperparâmetros
    input_size = 2
    hidden_size = 4
    output_size = 1
    learning_rate = 0.5
    epocas = 10000

    # Instancia o modelo importado do engine.py
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    print("--- Treinando a Rede Neural (engine.py) ---")
    for epoca in range(1, epocas + 1):
        # Forward Pass
        output = nn.forward(X)
        
        # Backward Pass
        nn.backward(X, y, output, learning_rate)

        # Monitoramento do Erro
        if epoca % 2000 == 0:
            loss = mean_squared_error(y, output)
            print(f"Época {epoca}/{epocas} - Perda (MSE): {loss:.6f}")

    print("\n--- Resultados Finais ---")
    previsoes = nn.forward(X)
    for entrada, pred in zip(X, previsoes):
        classe = int(pred[0] > 0.5)
        print(f"Entrada: {entrada} -> Saída: {pred[0]:.4f} -> Classe: {classe}")


if __name__ == "__main__":
    main()