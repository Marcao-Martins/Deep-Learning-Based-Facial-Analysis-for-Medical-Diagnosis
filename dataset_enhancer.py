"""
dataset_enhancer.py
===================

1. FaceCropper ........ detecta o rosto principal e faz o crop + padding
2. CustomFeatureExtractor .. extrai features customizadas das imagens (sem PyRadiomics)
3. EnhancedDataset ..... Dataset PyTorch que devolve (imagem, features, label)
4. get_data_loaders .... cria DataLoaders prontos para treino/validação/teste

Dependências:
    pip install opencv-python face-recognition albumentations pandas tqdm scikit-image
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# 1. FACE CROPPER
# --------------------------------------------------------------------------- #


class FaceCropper:
    """Detecta e recorta o maior rosto da imagem, aplicando um padding opcional."""

    def __init__(self, min_size: int = 80, pad: float = 0.3, out_size: Tuple[int, int] = (224, 224)):
        self.min_size = min_size
        self.pad = pad
        self.out_size = out_size
        self.detector_cv = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _detect(self, img: np.ndarray):
        """Tenta face_recognition, depois HaarCascade. Retorna lista (x, y, w, h)."""
        # Primeiro: face_recognition
        try:
            import face_recognition  # type: ignore

            boxes = face_recognition.face_locations(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return [(l, t, r - l, b - t) for (t, r, b, l) in boxes]
        except Exception:
            pass

        # Fallback: HaarCascade
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector_cv.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(self.min_size, self.min_size),
        )
        return faces

    def crop(self, img_path: Path, dst_path: Path) -> bool:
        img = cv2.imread(str(img_path))
        if img is None:
            return False

        faces = self._detect(img)
        if len(faces) == 0:
            # nenhuma face → apenas redimensiona e salva
            img_resized = cv2.resize(img, self.out_size)
            cv2.imwrite(str(dst_path), img_resized)
            return False

        # Seleciona a maior face
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        pad_x, pad_y = int(w * self.pad), int(h * self.pad)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(img.shape[1], x + w + pad_x), min(img.shape[0], y + h + pad_y)

        crop_img = img[y0:y1, x0:x1]
        crop_img = cv2.resize(crop_img, self.out_size)
        cv2.imwrite(str(dst_path), crop_img)
        return True


# --------------------------------------------------------------------------- #
# 2. CUSTOM FEATURE EXTRACTOR (sem PyRadiomics)
# --------------------------------------------------------------------------- #


class CustomFeatureExtractor:
    """Extrai features customizadas sem depender do PyRadiomics."""

    def __init__(self):
        try:
            from skimage import feature, filters, measure
            self.skimage_available = True
        except ImportError:
            self.skimage_available = False
            print("⚠️  scikit-image não encontrado. Instale com: pip install scikit-image")

    def _get_face_roi(self, img: np.ndarray) -> np.ndarray:
        """Obtém a região do rosto para análise."""
        h, w = img.shape[:2]
        try:
            import face_recognition  # type: ignore
            boxes = face_recognition.face_locations(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if boxes:
                t, r, b, l = boxes[0]
                return img[t:b, l:r]
        except Exception:
            pass
        # Fallback: usar toda a imagem
        return img

    def _extract_color_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extrai features de cor."""
        features = {}
        
        # Converter para diferentes espaços de cor
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Estatísticas por canal
        for i, (name, channel) in enumerate([('b', img[:,:,0]), ('g', img[:,:,1]), ('r', img[:,:,2])]):
            features[f'{name}_mean'] = float(np.mean(channel))
            features[f'{name}_std'] = float(np.std(channel))
            features[f'{name}_min'] = float(np.min(channel))
            features[f'{name}_max'] = float(np.max(channel))
            features[f'{name}_median'] = float(np.median(channel))
        
        # Estatísticas HSV
        for i, (name, channel) in enumerate([('h', hsv[:,:,0]), ('s', hsv[:,:,1]), ('v', hsv[:,:,2])]):
            features[f'hsv_{name}_mean'] = float(np.mean(channel))
            features[f'hsv_{name}_std'] = float(np.std(channel))
        
        # Estatísticas LAB
        for i, (name, channel) in enumerate([('l', lab[:,:,0]), ('a', lab[:,:,1]), ('b', lab[:,:,2])]):
            features[f'lab_{name}_mean'] = float(np.mean(channel))
            features[f'lab_{name}_std'] = float(np.std(channel))
        
        # Histograma de cores
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
        
        features['hist_b_mean'] = float(np.mean(hist_b))
        features['hist_g_mean'] = float(np.mean(hist_g))
        features['hist_r_mean'] = float(np.mean(hist_r))
        features['hist_b_std'] = float(np.std(hist_b))
        features['hist_g_std'] = float(np.std(hist_g))
        features['hist_r_std'] = float(np.std(hist_r))
        
        return features

    def _extract_texture_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extrai features de textura."""
        features = {}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.skimage_available:
            try:
                from skimage import feature, filters
                
                # GLCM (Gray Level Co-occurrence Matrix)
                glcm = feature.graycomatrix(gray, [1], [0, 45, 90, 135], levels=256, symmetric=True, normed=True)
                
                # Propriedades GLCM
                contrast = feature.graycoprops(glcm, 'contrast')
                dissimilarity = feature.graycoprops(glcm, 'dissimilarity')
                homogeneity = feature.graycoprops(glcm, 'homogeneity')
                energy = feature.graycoprops(glcm, 'energy')
                correlation = feature.graycoprops(glcm, 'correlation')
                
                for i, angle in enumerate([0, 45, 90, 135]):
                    features[f'glcm_contrast_{angle}'] = float(contrast[0, i])
                    features[f'glcm_dissimilarity_{angle}'] = float(dissimilarity[0, i])
                    features[f'glcm_homogeneity_{angle}'] = float(homogeneity[0, i])
                    features[f'glcm_energy_{angle}'] = float(energy[0, i])
                    features[f'glcm_correlation_{angle}'] = float(correlation[0, i])
                
                # Local Binary Pattern
                lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
                features['lbp_mean'] = float(np.mean(lbp))
                features['lbp_std'] = float(np.std(lbp))
                
                # Gabor filters
                gabor_responses = []
                for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                    for sigma in [1, 2]:
                        for frequency in [0.1, 0.3]:
                            gabor = filters.gabor(gray, frequency=frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)[0]
                            gabor_responses.append(np.mean(gabor))
                
                for i, resp in enumerate(gabor_responses):
                    features[f'gabor_{i}'] = float(resp)
                    
            except Exception as e:
                print(f"Erro ao extrair features de textura: {e}")
        
        # Features básicas de textura (sempre disponíveis)
        # Gradientes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features['gradient_mean'] = float(np.mean(grad_magnitude))
        features['gradient_std'] = float(np.std(grad_magnitude))
        features['gradient_max'] = float(np.max(grad_magnitude))
        
        # Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['laplacian_mean'] = float(np.mean(laplacian))
        features['laplacian_std'] = float(np.std(laplacian))
        features['laplacian_var'] = float(np.var(laplacian))
        
        return features

    def _extract_shape_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extrai features de forma."""
        features = {}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Dimensões
        h, w = gray.shape
        features['aspect_ratio'] = float(w / h)
        features['area'] = float(h * w)
        features['perimeter'] = float(2 * (h + w))
        
        # Contornos
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Maior contorno
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            features['contour_area'] = float(area)
            features['contour_perimeter'] = float(perimeter)
            features['circularity'] = float(4 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
            
            # Retângulo delimitador
            x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
            features['bounding_box_area'] = float(w_rect * h_rect)
            features['extent'] = float(area / (w_rect * h_rect)) if w_rect * h_rect > 0 else 0.0
        else:
            features['contour_area'] = 0.0
            features['contour_perimeter'] = 0.0
            features['circularity'] = 0.0
            features['bounding_box_area'] = 0.0
            features['extent'] = 0.0
        
        return features

    def _extract_statistical_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extrai features estatísticas."""
        features = {}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Estatísticas básicas
        features['mean_intensity'] = float(np.mean(gray))
        features['std_intensity'] = float(np.std(gray))
        features['min_intensity'] = float(np.min(gray))
        features['max_intensity'] = float(np.max(gray))
        features['median_intensity'] = float(np.median(gray))
        features['var_intensity'] = float(np.var(gray))
        features['skewness'] = float(self._skewness(gray))
        features['kurtosis'] = float(self._kurtosis(gray))
        
        # Percentis
        for p in [10, 25, 50, 75, 90]:
            features[f'percentile_{p}'] = float(np.percentile(gray, p))
        
        # Entropia
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()  # Normalizar
        entropy = -np.sum(hist * np.log2(hist + 1e-10))  # +1e-10 para evitar log(0)
        features['entropy'] = float(entropy)
        
        return features

    def _skewness(self, data: np.ndarray) -> float:
        """Calcula assimetria."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    def _kurtosis(self, data: np.ndarray) -> float:
        """Calcula curtose."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)

    def extract(self, img_path: Path) -> Dict[str, float]:
        """Extrai todas as features de uma imagem."""
        img = cv2.imread(str(img_path))
        if img is None:
            return {}

        # Obter ROI do rosto
        face_roi = self._get_face_roi(img)
        
        # Extrair features
        features = {}
        features.update(self._extract_color_features(face_roi))
        features.update(self._extract_texture_features(face_roi))
        features.update(self._extract_shape_features(face_roi))
        features.update(self._extract_statistical_features(face_roi))
        
        # Filtrar valores inválidos
        valid_features = {}
        for k, v in features.items():
            if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):
                valid_features[k] = float(v)
        
        return valid_features


# --------------------------------------------------------------------------- #
# 3. PIPELINE ENHANCE (crop + features customizadas)
# --------------------------------------------------------------------------- #

def enhance_dataset(
    src_dir: str,
    dst_dir: str = "Dataset/processed",
    features_csv: str = "Dataset/radiomics_features.csv",
    img_size: Tuple[int, int] = (224, 224),
    pad: float = 0.3,
) -> pd.DataFrame:
    """Processa dataset => imagens croppadas + CSV de features customizadas."""

    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    cropper = FaceCropper(pad=pad, out_size=img_size)
    feature_extractor = CustomFeatureExtractor()

    records: List[Dict[str, float]] = []
    for class_dir in src.iterdir():
        if not class_dir.is_dir():
            continue
        out_class = dst / class_dir.name
        out_class.mkdir(exist_ok=True)
        for img_path in tqdm(list(class_dir.glob("*.*")), desc=f"{class_dir.name}"):
            out_img = out_class / img_path.name
            cropper.crop(img_path, out_img)
            feats = feature_extractor.extract(out_img)
            
            # Adicionar caminho da imagem e classe explicitamente
            feats["image_path"] = str(out_img)
            feats["class"] = class_dir.name
            records.append(feats)

    df = pd.DataFrame(records)
    
    # Reorganizar colunas para garantir que image_path e class sejam as últimas
    feature_cols = [col for col in df.columns if col not in ["image_path", "class"]]
    df = df[feature_cols + ["image_path", "class"]]
    
    # Salvar CSV com index=False para evitar problemas
    df.to_csv(features_csv, index=False)
    
    print(
        f"✔ Imagens croppadas salvas em: {dst}\n"
        f"✔ Features customizadas salvas em: {features_csv} ({len(feature_cols)} features)\n"
        f"✔ Estrutura do CSV: {len(df)} linhas, {len(df.columns)} colunas\n"
        f"✔ Últimas colunas: {list(df.columns[-2:])}"
    )
    return df


# --------------------------------------------------------------------------- #
# 4. DATASET & DATALOADERS
# --------------------------------------------------------------------------- #


class EnhancedDataset(Dataset):
    """Dataset que retorna (imagem_tensor, features_tensor, label)"""

    def __init__(
        self,
        img_df: pd.DataFrame,
        label_map: Dict[str, int],
        features_df: pd.DataFrame,
        transform: Optional[Compose] = None,
    ):
        self.img_df = img_df.reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform
        feats_cols = [c for c in features_df.columns if c not in ("image_path", "class")]
        self.feats = features_df.set_index("image_path")[feats_cols]

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        row = self.img_df.iloc[idx]
        img = cv2.imread(row.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]

        features = torch.tensor(self.feats.loc[row.image_path].values, dtype=torch.float32)
        label = torch.tensor(self.label_map[row["class"]], dtype=torch.long)
        return img, features, label


def _build_df(root: str) -> pd.DataFrame:
    records = []
    for cls in os.listdir(root):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for img in os.listdir(cls_dir):
            if img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                records.append({"image_path": os.path.join(cls_dir, img), "class": cls})
    return pd.DataFrame(records)


def get_data_loaders(
    images_root: str,
    features_csv: str,
    batch_size: int = 32,
    valid_split: float = 0.15,
    test_split: float = 0.15,
    img_size: Tuple[int, int] = (224, 224),
    shuffle: bool = True,
    seed: int = 42,
):
    """Cria DataLoaders (train, valid, test) usando EnhancedDataset."""

    full_df = _build_df(images_root)
    feats_df = pd.read_csv(features_csv)

    # Split
    random.seed(seed)
    idx = list(range(len(full_df)))
    random.shuffle(idx)
    test_n = int(test_split * len(idx))
    valid_n = int(valid_split * len(idx))
    train_idx = idx[test_n + valid_n :]
    valid_idx = idx[test_n : test_n + valid_n]
    test_idx = idx[:test_n]

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    valid_df = full_df.iloc[valid_idx].reset_index(drop=True)
    test_df = full_df.iloc[test_idx].reset_index(drop=True)

    classes = sorted(full_df["class"].unique())
    label_map = {c: i for i, c in enumerate(classes)}

    transform = Compose(
        [
            Resize(*img_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    def _dl(df, shuff):
        return DataLoader(
            EnhancedDataset(df, label_map, feats_df, transform),
            batch_size=batch_size,
            shuffle=shuff,
            num_workers=0 if os.name == "nt" else 4,
            pin_memory=torch.cuda.is_available(),
        )

    return _dl(train_df, shuffle), _dl(valid_df, False), _dl(test_df, False), len(classes)


# --------------------------------------------------------------------------- #
# CLI BÁSICO
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhance ASD dataset (crop faces + custom features)")
    parser.add_argument("--input", required=True, help="Diretório Dataset/consolidated original")
    parser.add_argument("--output", default="Dataset/processed", help="Dir das imagens croppadas")
    parser.add_argument("--features", default="Dataset/custom_features.csv", help="CSV de saída")
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224], help="Dimensão final")
    parser.add_argument("--pad", type=float, default=0.3, help="Padding (%) ao redor do rosto")
    args = parser.parse_args()

    enhance_dataset(
        args.input, args.output, args.features, tuple(args.img_size), args.pad
    ) 