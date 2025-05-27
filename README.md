# Projeto de Classificação de Imagens: Gatos vs Cães

## Descrição do Problema
O objetivo é criar uma pipeline capaz de classificar imagens de gatos e cães. O fluxo inclui carregamento das imagens, pré-processamento (GaussianBlur + equalizeHist), extração de características via HOG e classificação usando SVM.

## Justificativa das Técnicas Utilizadas
- **Redimensionamento (128×128):** Redimensiona as imagens para 128 x 128  
- **Pré-processamento (GaussianBlur + equalizeHist):** reduz ruído e melhora o contraste, o que facilita a detecção de texturas e bordas.  
- **HOG (Histogram of Oriented Gradients):** extrai padrões de orientação de gradientes, sendo robusto a variações de iluminação e pose.  
- **SVM (Support Vector Machine):** classificador eficaz para problemas binários em espaços de alta dimensionalidade, com boa capacidade de generalização.  
- **Augmentação (flip horizontal):** dobra o conjunto de dados, tentando evitar overfitting e variando a posição dos animais nas imagens.

## Etapas Realizadas
1. **Aquisição de dados**  
   - Dataset público do Kaggle: [Cats and Dogs Image Classification](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification/data)  
   - Organização em pastas: `imagens/cats/` e `imagens/dogs/`  
2. **Demo de Filtros**  
   - Visualização em grid (3×6) das 6 primeiras imagens de cães e as 6 primeiras de gatos, mostrando original, GaussianBlur e equalizeHist.  
3. **Pré-processamento e Extração de Features**  
   - Conversão para tons de cinza e redimensionamento para 128×128.  
   - Aplicação de GaussianBlur (kernel 5×5) e equalização de histograma.  
   - Cálculo do descritor HOG via `cv2.HOGDescriptor`.  
4. **Data Augmentation**  
   - Flip horizontal para cada imagem original.  
5. **Split Treino/Teste**  
   - Divisão em 80% treino e 20% teste.  
6. **Treinamento**  
   - `GridSearchCV` para busca de hiperparâmetros (`C`, `kernel`, `gamma`) em SVM com validação cruzada 5-fold.

7. **Avaliação**  
   - Cálculo de acurácia, precision, recall, F1-score (via `classification_report`) e matriz de confusão.

## Resultados Obtidos

* **Melhores parâmetros (SVM):**

  * `C = 1`
  * `kernel = 'rbf'`
  * `gamma = 'scale'`
* **Acurácia no conjunto de teste:** 74.55%
* **Classification Report:**

  ```
               precision    recall  f1-score   support
  cat            0.7949      0.6643     0.7237       140
  dog            0.7099      0.8273     0.7641       139
  accuracy                          0.7455      279
  ```
* **Matriz de Confusão:**

  ```
         Predicted
         cat   dog
  real
  cat     93    47
  dog     24    115
  ```

## Tempo Total Gasto

2 horas para o desenvolvimento e realização de testes, além da escrita do README.MD
e o treinamento leva de 3 a 5 minutos aproximadamente.

## Dificuldades encontradas
- Poucas imagens do dataset podia gerar overfitting. Para evitar isso, realizei uma data augmentation (flip) para dobrar a quantidade das imagens.
-   Validação cruzada em 5 folds aumentou o tempo do treino;
