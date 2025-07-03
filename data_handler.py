import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
# suporte a features tabulares
import pandas as pd


class MedicalImageAugmentation:
    """Classe para data augmentation específico para imagens médicas."""
    
    @staticmethod
    def get_medical_augmentation_pipeline(image_size: int = 224, intensity: str = 'medium') -> A.Compose:
        """
        Retorna pipeline de augmentation otimizado para imagens médicas.
        
        Args:
            image_size: Tamanho da imagem de saída
            intensity: Intensidade do augmentation ('light', 'medium', 'strong')
            
        Returns:
            Pipeline de augmentation Albumentations
        """
        if intensity == 'light':
            return A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.1),
                A.Rotate(limit=15, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                ], p=0.2),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
                ], p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        
        elif intensity == 'medium':
            return A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=25, p=0.4),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.4),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                    A.MotionBlur(blur_limit=3, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15, p=0.5),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                    A.OpticalDistortion(distort_limit=0.2, shift_limit=0.15, p=0.5),
                ], p=0.2),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        
        else:  # strong
            return A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=35, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=5, p=0.5),
                    A.MotionBlur(blur_limit=5, p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                ], p=0.4),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
                    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                ], p=0.4),
                A.OneOf([
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                    A.GridDistortion(num_steps=5, distort_limit=0.4, p=0.5),
                    A.OpticalDistortion(distort_limit=0.3, shift_limit=0.2, p=0.5),
                    A.Perspective(scale=(0.05, 0.1), p=0.5),
                ], p=0.3),
                A.CoarseDropout(max_holes=12, max_height=48, max_width=48, p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    @staticmethod
    def get_test_transform(image_size: int = 224) -> A.Compose:
        """
        Retorna transformação para teste/validação (sem augmentation).
        
        Args:
            image_size: Tamanho da imagem de saída
            
        Returns:
            Pipeline de transformação para teste
        """
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


class AlbumentationsDataset(Dataset):
    """Dataset customizado que usa Albumentations para augmentation."""
    
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


# ------------------------- Dataset com features extras -------------------- #


class AlbumentationsDatasetWithFeatures(AlbumentationsDataset):
    """Extende AlbumentationsDataset para também retornar vetor de features."""

    def __init__(self, data_dir: str, features_df: pd.DataFrame, transform=None):
        super().__init__(data_dir, transform)
        
        # Verificar se a coluna image_path existe
        if 'image_path' not in features_df.columns:
            # Se não existir, usar a penúltima coluna como image_path
            print("⚠️  Coluna 'image_path' não encontrada no CSV. Usando penúltima coluna.")
            features_df = features_df.copy()
            features_df['image_path'] = features_df.iloc[:, -2]  # Penúltima coluna
            features_df['class'] = features_df.iloc[:, -1]       # Última coluna
        
        # indexar features pelo caminho relativo
        self.features_df = features_df.set_index('image_path')
        # lista de colunas de features
        self.feature_cols = [c for c in self.features_df.columns if c not in ('image_path', 'class')]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Tentar diferentes formatos de caminho para encontrar as features
        try:
            # 1. Caminho absoluto (como está no CSV)
            feats = self.features_df.loc[img_path][self.feature_cols].values.astype(np.float32)
            if idx < 3:
                print(f"✅ Features encontradas (absoluto): {len(feats)} features")
        except KeyError:
            # 2. Caminho relativo
            rel_path = os.path.relpath(img_path, os.path.dirname(self.data_dir))
            rel_path = rel_path.replace('\\', '/')
            if idx < 3:
                print(f"🔍 Debug - img_path: {img_path}")
                print(f"🔍 Debug - rel_path: {rel_path}")
                print(f"🔍 Debug - CSV keys: {list(self.features_df.index)[:5]}")
            try:
                feats = self.features_df.loc[rel_path][self.feature_cols].values.astype(np.float32)
                if idx < 3:
                    print(f"✅ Features encontradas (relativo): {len(feats)} features")
            except KeyError:
                # 3. Caminho sem o prefixo Dataset/ ou Dataset\
                normalized_path = img_path.replace('Dataset\\', '').replace('Dataset/', '')
                try:
                    feats = self.features_df.loc[normalized_path][self.feature_cols].values.astype(np.float32)
                    if idx < 3:
                        print(f"✅ Features encontradas (normalizado): {len(feats)} features")
                except KeyError:
                    print(f"⚠️  Features não encontradas para: {img_path}")
                    feats = np.zeros(len(self.feature_cols), dtype=np.float32)
        feats_tensor = torch.tensor(feats, dtype=torch.float32)
        return image, feats_tensor, label


class DataHandler:
    """Classe responsável por toda a manipulação de dados usando PyTorch."""
    
    def __init__(self, data_config):
        """
        Inicializa o DataHandler com as configurações de dados.
        
        Args:
            data_config: Dicionário com as configurações de dados OU string com caminho do dataset
        """
        # Se data_config for string, criar configuração padrão
        if isinstance(data_config, str):
            self.config = {
                'dataset_path': data_config,
                'image_height': 224,
                'image_width': 224,
                'channels': 3,
                'batch_size': 32,
                'validation_split': 0.15,
                'test_split': 0.15,
                'shuffle': True,
                'seed': 42,
                # 'features_csv': None  # caminho opcional para CSV com features
            }
        else:
            self.config = data_config
            
        self.dataset_path = self.config['dataset_path']
        self.image_height = self.config['image_height']
        self.image_width = self.config['image_width']
        self.channels = self.config['channels']
        self.batch_size = self.config['batch_size']
        self.validation_split = self.config['validation_split']
        self.test_split = self.config['test_split']
        self.shuffle = self.config['shuffle']
        self.seed = self.config['seed']
        # arquivo opcional de features
        self.features_csv = self.config.get('features_csv')
        
        # Definir seeds para reprodutibilidade
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.num_classes = None
        self.class_names = None
        
    def load_dataset(self) -> Dict[str, Any]:
        """
        Carrega o dataset a partir do caminho especificado.
        
        Returns:
            Dicionário com informações sobre o dataset carregado
        """
        # Verificar se o diretório existe
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path {self.dataset_path} does not exist")
        
        # Procurar por estrutura de pastas existente (train/valid/test)
        train_dir = os.path.join(self.dataset_path, 'train')
        valid_dir = os.path.join(self.dataset_path, 'valid')
        test_dir = os.path.join(self.dataset_path, 'test')
        
        # Se existir estrutura separada, usar ela
        if os.path.exists(train_dir) and os.path.exists(valid_dir) and os.path.exists(test_dir):
            print("Usando estrutura de dados existente (train/valid/test)")
            self.train_dir = train_dir
            self.valid_dir = valid_dir
            self.test_dir = test_dir
            self.use_existing_split = True
        else:
            # Caso contrário, usar o diretório consolidado
            consolidated_dir = os.path.join(self.dataset_path, 'consolidated')
            if os.path.exists(consolidated_dir):
                print("Usando diretório consolidado para divisão de dados")
                self.data_dir = consolidated_dir
                self.use_existing_split = False
            else:
                # Verificar se o diretório atual tem classes diretamente
                classes_in_current_dir = [d for d in os.listdir(self.dataset_path) 
                                        if os.path.isdir(os.path.join(self.dataset_path, d))]
                if len(classes_in_current_dir) >= 2:  # Pelo menos 2 classes
                    print(f"Usando estrutura de classes direta: {classes_in_current_dir}")
                    self.data_dir = self.dataset_path
                    self.use_existing_split = False
                else:
                    raise ValueError("Nenhuma estrutura de dados válida encontrada")
        
        # Detectar classes
        if self.use_existing_split:
            self.class_names = sorted(os.listdir(self.train_dir))
        else:
            self.class_names = sorted(os.listdir(self.data_dir))
            
        self.num_classes = len(self.class_names)
        
        print(f"Classes detectadas: {self.class_names}")
        print(f"Número de classes: {self.num_classes}")
        
        # Verificar se há GPU disponível
        if torch.cuda.is_available():
            print(f"🎮 GPU detectada: {torch.cuda.get_device_name(0)}")
            print(f"   Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠️  GPU não detectada. Usando CPU.")
        
        return {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'use_existing_split': self.use_existing_split
        }
    
    def get_albumentations_transforms(self, model_config: Dict[str, Any] = None) -> Tuple[A.Compose, A.Compose]:
        """
        Cria transformações usando Albumentations.
        
        Args:
            model_config: Configurações específicas do modelo
            
        Returns:
            Tupla com (train_transform, test_transform)
        """
        if model_config is None:
            model_config = {}
            
        use_data_augmentation = model_config.get('use_data_augmentation', True)
        augmentation_intensity = model_config.get('augmentation_intensity', 'medium')
        
        if use_data_augmentation:
            train_transform = MedicalImageAugmentation.get_medical_augmentation_pipeline(
                image_size=self.image_height,
                intensity=augmentation_intensity
            )
        else:
            # Sem augmentation, apenas redimensionar e normalizar
            train_transform = A.Compose([
                A.Resize(self.image_height, self.image_width),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        
        test_transform = MedicalImageAugmentation.get_test_transform(
            image_size=self.image_height
        )
        
        return train_transform, test_transform
    
    def get_data_loaders(self, model_config: Dict[str, Any] = None) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """
        Retorna DataLoaders para treino, validação e teste usando Albumentations.
        
        Args:
            model_config: Configurações específicas do modelo
            
        Returns:
            Tupla com (train_loader, valid_loader, test_loader, num_classes)
        """
        # Carregar dataset
        dataset_info = self.load_dataset()
        
        # Carregar CSV de features se existir
        features_df = None
        if self.features_csv and os.path.exists(self.features_csv):
            print(f"📑 Carregando features de: {self.features_csv}")
            features_df = pd.read_csv(self.features_csv)

        # Sempre usar Albumentations
        train_transform, test_transform = self.get_albumentations_transforms(model_config)
        print("Usando Albumentations para data augmentation")
        
        # Determinar número de workers baseado no sistema
        num_workers = 0 if os.name == 'nt' else 4  # Windows = 0, Linux/Mac = 4
        
        if self.use_existing_split:
            # Usar estrutura existente
            ds_class = AlbumentationsDatasetWithFeatures if features_df is not None else AlbumentationsDataset

            train_dataset = ds_class(
                data_dir=self.train_dir,
                transform=train_transform,
                **({'features_df': features_df} if features_df is not None else {})
            )
            
            valid_dataset = ds_class(
                data_dir=self.valid_dir,
                transform=test_transform,
                **({'features_df': features_df} if features_df is not None else {})
            )
            
            test_dataset = ds_class(
                data_dir=self.test_dir,
                transform=test_transform,
                **({'features_df': features_df} if features_df is not None else {})
            )
        else:
            # Criar divisão a partir do diretório consolidado
            ds_class_full = AlbumentationsDatasetWithFeatures if features_df is not None else AlbumentationsDataset

            full_dataset = ds_class_full(
                data_dir=self.data_dir,
                transform=test_transform,
                **({'features_df': features_df} if features_df is not None else {})  # para __getitem__ consistente
            )
            
            # Calcular tamanhos
            total_size = len(full_dataset)
            test_size = int(self.test_split * total_size)
            valid_size = int(self.validation_split * total_size)
            train_size = total_size - test_size - valid_size
            
            # Dividir dataset
            train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, 
                [train_size, valid_size, test_size],
                generator=torch.Generator().manual_seed(self.seed)
            )
            
            print("AVISO: Usando divisão simplificada. Para melhor controle, "
                  "considere organizar os dados em pastas train/valid/test")
        
        # Criar DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Mostrar informações sobre augmentation
        if model_config and model_config.get('use_data_augmentation', True):
            intensity = model_config.get('augmentation_intensity', 'medium')
            print(f"✅ Data augmentation ativado (intensidade: {intensity})")
            print("Usando Albumentations")
        else:
            print("⚠️  Data augmentation desativado")
        
        print(f"\nInformações dos DataLoaders:")
        print(f"Amostras de treino: {len(train_dataset)}")
        print(f"Amostras de validação: {len(valid_dataset)}")
        print(f"Amostras de teste: {len(test_dataset)}")
        print(f"Batches de treino: {len(self.train_loader)}")
        print(f"Batches de validação: {len(self.valid_loader)}")
        print(f"Batches de teste: {len(self.test_loader)}")
        
        return (self.train_loader, 
                self.valid_loader, 
                self.test_loader, 
                self.num_classes)
    
    # Manter compatibilidade com código antigo
    def get_data_generators(self, model_config: Dict[str, Any] = None):
        """Método de compatibilidade que chama get_data_loaders."""
        return self.get_data_loaders(model_config) 