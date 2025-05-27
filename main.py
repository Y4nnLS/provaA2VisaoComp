import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Diretórios das imagens
CATS_DIR = 'imagens/cats'
DOGS_DIR = 'imagens/dogs'

# Configurações de HOG
WIN_SIZE      = (128, 128)
BLOCK_SIZE    = (32, 32)
BLOCK_STRIDE  = (16, 16)
CELL_SIZE     = (16, 16)
NBINS         = 9
hog = cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)

def demo_filters_grid(folder, label_name, sample_size=6):
    """
    Exibe num grid 3×6 as primeiras sample_size imagens de folder,
    inserindo acima de cada linha uma faixa com o texto de legenda.
    """
    files = sorted(os.listdir(folder))[:sample_size]
    orig_list, blur_list, eq_list = [], [], []

    for fname in files:
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, WIN_SIZE)

        # versões
        orig_list.append(img)
        blur_list.append( cv2.GaussianBlur(img, (5,5), 0) )
        eq_list.append( cv2.equalizeHist(blur_list[-1]) )

    # monta cada linha e converte para BGR
    row1 = cv2.cvtColor(np.hstack(orig_list), cv2.COLOR_GRAY2BGR)
    row2 = cv2.cvtColor(np.hstack(blur_list), cv2.COLOR_GRAY2BGR)
    row3 = cv2.cvtColor(np.hstack(eq_list),  cv2.COLOR_GRAY2BGR)

    rows = [row1, row2, row3]
    labels = ["Original", "GaussianBlur", "equalizeHist"]

    banner_h = 30  # altura da faixa de texto
    banner_color = (255, 255, 255)  # preto (você pode usar branco (255,255,255) se preferir)
    text_color   = (0, 0, 0)
    font         = cv2.FONT_HERSHEY_SIMPLEX
    font_scale   = 0.6
    thickness    = 1

    # Para cada linha, cria uma faixa + texto + a própria linha
    bands = []
    for row_img, label in zip(rows, labels):
        h, w = row_img.shape[:2]
        # faixa em preto
        banner = np.full((banner_h, w, 3), banner_color, dtype=np.uint8)
        text = f"{'-'*12} {label} {'-'*12}"
        # desenha o texto centralizado verticalmente na faixa, com 10px de margem à esquerda
        cv2.putText(
            banner,
            text,
            (10, banner_h - 10),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )
        bands.extend([banner, row_img])

    # empilha tudo: banner1, row1, banner2, row2, banner3, row3
    grid_with_banners = np.vstack(bands)

    window_name = f"Demo {label_name} (Original | GaussianBlur | equalizeHist)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, grid_with_banners)
    print(f"[DEMO GRID] Mostrando {sample_size} imagens de {label_name} em grid com legendas.")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def preprocess_and_hog(img):
    """
    Recebe imagem já redimensionada em tons de cinza,
    aplica blur, equaliza e retorna vetor HOG.
    """
    blur = cv2.GaussianBlur(img, (5,5), 0)
    eq   = cv2.equalizeHist(blur)
    return hog.compute(eq).flatten()


def load_data_with_augmentation(folder, label, augment=True):
    features, labels = [], []
    files = os.listdir(folder)
    total = len(files)
    for idx, fname in enumerate(files, start=1):
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, WIN_SIZE)

        # HOG do original
        desc = preprocess_and_hog(img)
        features.append(desc); labels.append(label)

        # augmentação: flip horizontal
        if augment:
            flip = cv2.flip(img, 1)
            features.append(preprocess_and_hog(flip))
            labels.append(label)

        if idx % 50 == 0 or idx == total:
            print(f"[{os.path.basename(folder)}] {idx}/{total} imagens processadas")

    return np.array(features), np.array(labels)

    
def main():
    # 1) Demo GRID de filtros na tela
    print("=== Demo GRID de 6 gatos ===")
    demo_filters_grid(CATS_DIR, 'cat', sample_size=6)

    print("=== Demo GRID de 6 cães ===")
    demo_filters_grid(DOGS_DIR, 'dog', sample_size=6)

    # 2) Carregamento e pré-processamento completo
    print("\n=== Carregando e pré-processando TODO o dataset ===")
    Xc, yc = load_data_with_augmentation(CATS_DIR, 0, augment=True)
    Xd, yd = load_data_with_augmentation(DOGS_DIR, 1, augment=True)
    X = np.vstack([Xc, Xd]); y = np.hstack([yc, yd])
    print(f"Total de amostras (c/ augmentação): {X.shape[0]}")

    # 3) Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

    # 4) Busca de hiperparâmetros via GridSearchCV
    print("=== Iniciando GridSearchCV ===")
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print("Melhores parâmetros:", grid.best_params_)

    # 5) Avaliação final
    print("\n=== Avaliando no conjunto de teste ===")
    y_pred = best.predict(X_test)
    print(f"Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['cat','dog'], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 6) Treinamento e avaliação com class_weight balanced
    print("\n=== Treinando SVM balanceado ===")
    clf_bal = SVC(
        C=best.C, kernel=best.kernel, gamma=best.gamma,
        class_weight={0:1,1:2}, probability=True, random_state=42
    )
    clf_bal.fit(X_train, y_train)

    print("\n=== Avaliando no conjunto de teste (balanced) ===")
    y_pred_bal = clf_bal.predict(X_test)
    print(f"Acurácia (balanced): {accuracy_score(y_test, y_pred_bal) * 100:.2f}%")
    print("\nClassification Report (balanced):")
    print(classification_report(y_test, y_pred_bal, target_names=['cat','dog'], digits=4))
    print("Confusion Matrix (balanced):")
    print(confusion_matrix(y_test, y_pred_bal))

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
