# Rede Neural Siamese para Reconhecimento Facial

Este projeto implementa uma **Rede Neural Siamese** para realizar reconhecimento facial. A solução é baseada na construção de embeddings de imagens utilizando uma arquitetura convolucional e no cálculo da distância L1 para comparar imagens de entrada com imagens de validação. O modelo é treinado para identificar pares positivos (mesma pessoa) e negativos (pessoas diferentes) com uma camada de classificação binária.
<br>
<br>

## Estrutura do Código

1. **Pré-requisitos e Configuração Inicial**
   - Importação de bibliotecas como TensorFlow, OpenCV, NumPy e Pandas.
   - Montagem do Google Drive para acesso a arquivos armazenados no diretório `/gdrive`.

2. **Definição dos Caminhos**
   - Diretórios organizados para armazenar imagens de:
     - Ancoragem (`anchor`)
     - Positivas (`positive`)
     - Negativas (`negative`)

3. **Carregamento e Pré-processamento dos Dados**
   - Criação de datasets utilizando `tf.data.Dataset` para leitura eficiente de imagens.
   - Função `preprocess` para redimensionar imagens para 100x100 e normalizá-las (valores entre 0 e 1).
   - Combinação de imagens ancoradas, positivas e negativas, gerando dados balanceados com seus respectivos rótulos.

4. **Divisão dos Dados**
   - Separação em dados de treino e teste com proporções de 70% e 30%, respectivamente.
   - Configuração de batches para treinamento e prefetch para otimização.

5. **Construção da Rede Neural Siamese**
   - Definição do modelo de **embedding facial** com 4 blocos de camadas convolucionais, seguidas de pooling e uma camada densa com 4096 neurônios.
   - Implementação da camada `L1Dist` para calcular a distância absoluta entre os embeddings de entrada e validação.
   - Construção do modelo Siamese final com um classificador binário (camada `Dense` com função de ativação sigmoide).

6. **Treinamento do Modelo**
   - Função `train_step`: executa o passo de treino com cálculo de perda (Binary Crossentropy) e ajuste dos pesos via gradientes.
   - Função `train`: gerencia o loop de treinamento por múltiplas épocas, salvando checkpoints a cada 10 épocas.

7. **Avaliação**
   - Cálculo de métricas de precisão e recall para medir o desempenho do modelo.
   - Utilização de dados de teste para validação após o treinamento.

8. **Exportação e Carregamento do Modelo**
   - Salvamento do modelo treinado no formato `.h5`.
   - Recarregamento do modelo para futuras inferências, com suporte a camadas customizadas.
<br>

## Como Utilizar

1. **Configuração Inicial**
   - Monte o Google Drive para acessar as imagens localizadas nos diretórios especificados.
   - Organize suas imagens nas pastas `anchor`, `positive` e `negative`.

2. **Treinamento**
   - Execute o notebook para processar os dados e iniciar o treinamento.
   - Modifique o número de épocas na variável `EPOCHS` conforme necessário.

3. **Avaliação**
   - Utilize os dados de teste para calcular precisão e recall, garantindo o desempenho desejado.

4. **Salvamento**
   - Salve o modelo treinado no Google Drive para reutilização futura.

5. **Inferência**
   - Carregue o modelo salvo para realizar predições em novas imagens.
<br>

## Métricas de Desempenho

- **Precisão (Precision):** Avalia a proporção de imagens preditas corretamente como similares.
- **Recall:** Mede a capacidade do modelo de identificar todas as imagens similares corretamente.
