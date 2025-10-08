import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import collections
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from rdkit.Chem import rdMolDescriptors
import time
import subprocess
import sys
from tqdm.auto import trange, tqdm

# Попробуем импортировать MAP4, если не установлен - установим
try:
    from map4 import MAP4
except ImportError:
    print("MAP4 not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "map4"])
    from map4 import MAP4


# Класс для расчета разных типов фингерпринтов
class FingerprintCalculator:
    @staticmethod
    def compute_morgan(smiles, radius=2, n_bits=512):
        """Фингерпринт Моргана (ECFP)"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return torch.tensor(np.array(fp), dtype=torch.float32)

    @staticmethod
    def compute_maccs(smiles):
        """MACCS Keys - 166 бинарных фич"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        return torch.tensor(np.array(fp), dtype=torch.float32)

    @staticmethod
    def compute_rdkit(smiles, n_bits=512):
        """RDKit Topological Fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = RDKFingerprint(mol, fpSize=n_bits)
        return torch.tensor(np.array(fp), dtype=torch.float32)

    @staticmethod
    def compute_atom_pairs(smiles, n_bits=512):
        """Atom Pairs Fingerprint (Hashed)"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        return torch.tensor(np.array(fp), dtype=torch.float32)

    @staticmethod
    def compute_topological_torsions(smiles, n_bits=512):
        """Topological Torsions Fingerprint (Hashed)"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, targetSize=n_bits)
        return torch.tensor(np.array(fp), dtype=torch.float32)

    @staticmethod
    def compute_pattern(smiles, n_bits=512):
        """Pattern Fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = RDKFingerprint(mol, fpSize=n_bits, minPath=1, maxPath=7)
        return torch.tensor(np.array(fp), dtype=torch.float32)

    @staticmethod
    def compute_avalon(smiles, n_bits=512):
        """Avalon Fingerprint"""
        try:
            from rdkit.Avalon import pyAvalonTools
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
            return torch.tensor(np.array(fp), dtype=torch.float32)
        except ImportError:
            print("Avalon fingerprints not available. Install rdkit with Avalon support.")
            return None

    @staticmethod
    def compute_layered(smiles, n_bits=512):
        """Layered Fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = RDKFingerprint(mol, fpSize=n_bits, minPath=1, maxPath=7, layerFlags=0xFFFFFFFF)
        return torch.tensor(np.array(fp), dtype=torch.float32)

    @staticmethod
    def compute_map4(smiles, dimensions=512, radius=2, is_counted=False):
        """MAP4 Fingerprint - Modern alternative to Morgan"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            map4_calculator = MAP4(dimensions=dimensions, radius=radius, is_counted=is_counted)
            fp = map4_calculator.calculate(mol)
            return torch.tensor(np.array(fp), dtype=torch.float32)
        except Exception as e:
            print(f"MAP4 calculation error for {smiles}: {e}")
            return None

    @staticmethod
    def compute_rdkit_2d_descriptors(smiles):
        """RDKit 2D Descriptors (continuous values)"""
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Выбираем основные дескрипторы
        descriptor_list = [
            Descriptors.MolWt,
            Descriptors.MolLogP,
            Descriptors.NumHDonors,
            Descriptors.NumHAcceptors,
            Descriptors.TPSA,
            Descriptors.NumRotatableBonds,
            Descriptors.NumAromaticRings,
            Descriptors.FractionCsp3,
            Descriptors.NumHeteroatoms,
            Descriptors.NumRadicalElectrons
        ]

        descriptors = []
        for desc_func in descriptor_list:
            try:
                value = desc_func(mol)
                descriptors.append(value)
            except:
                descriptors.append(0.0)

        return torch.tensor(np.array(descriptors), dtype=torch.float32)


# Универсальный класс датасета для любого типа фингерпринтов
class FingerprintReactionDataset(Dataset):
    def __init__(self, true_map, false_map, fingerprint_func, fp_name, **fp_kwargs):
        """
        Parameters:
        true_map: dict {substrate_smiles: set(product_smiles)} - истинные продукты
        false_map: dict {substrate_smiles: set(product_smiles)} - ложные продукты
        fingerprint_func: функция для расчета фингерпринта
        fp_name: название фингерпринта (для отладки)
        fp_kwargs: параметры для функции фингерпринта
        """
        self.data = []
        self.labels = []
        self.fp_name = fp_name

        true_count = 0
        false_count = 0
        missing_smiles = 0

        # Собираем все уникальные SMILES
        all_smiles = set()
        for substrate, products in true_map.items():
            all_smiles.add(substrate)
            all_smiles.update(products)
        for substrate, products in false_map.items():
            all_smiles.add(substrate)
            all_smiles.update(products)

        # Предварительно вычисляем фингерпринты
        print(f"Calculating {fp_name} fingerprints for {len(all_smiles)} molecules...")
        fp_dict = {}
        for i, smiles in enumerate(tqdm(all_smiles)):
            fp = fingerprint_func(smiles, **fp_kwargs)
            if fp is not None:
                fp_dict[smiles] = fp
            else:
                missing_smiles += 1

        print(f"Successfully calculated {len(fp_dict)}/{len(all_smiles)} fingerprints")
        if missing_smiles > 0:
            print(f"Warning: {missing_smiles} molecules could not be processed")

        # Добавляем истинные пары (метка 1)
        for substrate, products in true_map.items():
            if substrate in fp_dict:
                substrate_fp = fp_dict[substrate]
                for product in products:
                    if product in fp_dict:
                        product_fp = fp_dict[product]
                        # Конкатенируем фингерпринты
                        combined_fp = torch.cat([substrate_fp, product_fp])
                        self.data.append(combined_fp)
                        self.labels.append(1)
                        true_count += 1

        # Добавляем ложные пары (метка 0)
        for substrate, products in false_map.items():
            if substrate in fp_dict:
                substrate_fp = fp_dict[substrate]
                for product in products:
                    if product in fp_dict:
                        product_fp = fp_dict[product]
                        # Конкатенируем фингерпринты
                        combined_fp = torch.cat([substrate_fp, product_fp])
                        self.data.append(combined_fp)
                        self.labels.append(0)
                        false_count += 1

        print(f"{fp_name} dataset created: {len(self.data)} samples")
        print(f"True pairs: {true_count}, False pairs: {false_count}")

        # Сохраняем размерность входа
        if len(self.data) > 0:
            self.input_dim = self.data[0].shape[0]
            print(f"Input dimension: {self.input_dim}")
        else:
            self.input_dim = 0
            print("Warning: No data in dataset!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Универсальная модель классификатора
class ReactionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256], dropout_rate=0.3):
        super(ReactionClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 1)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        features = self.features(x)
        output = self.classifier(features)
        return output


# Функция обучения
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_auc = 0
    best_model_state = None

    for epoch in trange(num_epochs):
        # Обучение
        model.train()
        epoch_loss = 0
        batch_count = 0

        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()

            if batch_data.dtype != torch.float32:
                batch_data = batch_data.float()
            batch_labels = batch_labels.float()

            outputs = model(batch_data)

            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)

            loss = criterion(outputs, batch_labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
        else:
            avg_loss = 0

        # Валидация
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                if batch_data.dtype != torch.float32:
                    batch_data = batch_data.float()
                batch_labels = batch_labels.float()

                outputs = model(batch_data)
                probs = torch.sigmoid(outputs)

                if probs.dim() > 1:
                    probs = probs.squeeze(1)

                probs_np = probs.cpu().numpy()
                labels_np = batch_labels.cpu().numpy()

                if probs_np.size == 1:
                    val_preds.append(float(probs_np))
                    val_true.append(float(labels_np))
                else:
                    val_preds.extend(probs_np.tolist())
                    val_true.extend(labels_np.tolist())

        if len(val_preds) > 0:
            val_preds = np.array(val_preds)
            val_true = np.array(val_true)

            if len(np.unique(val_true)) < 2:
                val_accuracy = 0.5
                val_auc = 0.5
            else:
                val_accuracy = accuracy_score(val_true, (val_preds > 0.5).astype(int))
                val_auc = roc_auc_score(val_true, val_preds)
        else:
            val_accuracy = 0.5
            val_auc = 0.5

        scheduler.step(avg_loss)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_auc


# Функция для оценки одного типа фингерпринта
def evaluate_fingerprint(fingerprint_func, fp_name, true_map, false_map, **fp_kwargs):
    print(f"\n{'=' * 50}")
    print(f"Evaluating {fp_name}")
    print(f"{'=' * 50}")

    start_time = time.time()

    # Создаем датасет
    dataset = FingerprintReactionDataset(true_map, false_map, fingerprint_func, fp_name, **fp_kwargs)

    if len(dataset) == 0:
        print(f"No data available for {fp_name}!")
        return None

    # Проверяем распределение классов
    label_counter = collections.Counter(dataset.labels)
    print(f"Label distribution: {dict(label_counter)}")

    if len(label_counter) < 2:
        print(f"Warning: Only one class in {fp_name} dataset!")
        return None

    # Разделяем на train/val
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Создаем DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Создаем модель
    model = ReactionClassifier(input_dim=dataset.input_dim)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Обучаем модель
    model, best_auc = train_model(model, train_loader, val_loader, num_epochs=10)

    # Финальная оценка
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            if batch_data.dtype != torch.float32:
                batch_data = batch_data.float()
            batch_labels = batch_labels.float()

            outputs = model(batch_data)
            probs = torch.sigmoid(outputs)

            if probs.dim() > 1:
                probs = probs.squeeze(1)

            probs_np = probs.cpu().numpy()
            labels_np = batch_labels.cpu().numpy()

            if probs_np.size == 1:
                all_preds.append(float(probs_np))
                all_true.append(float(labels_np))
            else:
                all_preds.extend(probs_np.tolist())
                all_true.extend(labels_np.tolist())

    if len(all_true) > 0 and len(np.unique(all_true)) >= 2:
        final_accuracy = accuracy_score(all_true, (np.array(all_preds) > 0.5).astype(int))
        final_auc = roc_auc_score(all_true, all_preds)
    else:
        final_accuracy = 0.5
        final_auc = 0.5

    end_time = time.time()
    training_time = end_time - start_time

    print(f"\n{fp_name} Results:")
    print(f"Final Accuracy: {final_accuracy:.4f}")
    print(f"Final AUC: {final_auc:.4f}")
    print(f"Best AUC during training: {best_auc:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    return {
        'fingerprint': fp_name,
        'accuracy': final_accuracy,
        'auc': final_auc,
        'best_auc': best_auc,
        'training_time': training_time,
        'input_dim': dataset.input_dim,
        'dataset_size': len(dataset)
    }


# Основная функция сравнения
def compare_fingerprints():
    """Сравнивает разные типы фингерпринтов"""

    # Определяем конфигурации фингерпринтов для сравнения
    fingerprint_configs = [
        {
            'name': 'Morgan_512',
            'func': FingerprintCalculator.compute_morgan,
            'kwargs': {'n_bits': 512, 'radius': 2}
        },
        {
            'name': 'Morgan_1024',
            'func': FingerprintCalculator.compute_morgan,
            'kwargs': {'n_bits': 1024, 'radius': 2}
        },
        {
            'name': 'MACCS',
            'func': FingerprintCalculator.compute_maccs,
            'kwargs': {}
        },
        {
            'name': 'RDKit_512',
            'func': FingerprintCalculator.compute_rdkit,
            'kwargs': {'n_bits': 512}
        },
        {
            'name': 'AtomPairs_512',
            'func': FingerprintCalculator.compute_atom_pairs,
            'kwargs': {'n_bits': 512}
        },
        {
            'name': 'TopologicalTorsions_512',
            'func': FingerprintCalculator.compute_topological_torsions,
            'kwargs': {'n_bits': 512}
        },
        {
            'name': 'Pattern_512',
            'func': FingerprintCalculator.compute_pattern,
            'kwargs': {'n_bits': 512}
        },
        {
            'name': 'MAP4_512',
            'func': FingerprintCalculator.compute_map4,
            'kwargs': {'dimensions': 512, 'radius': 2}
        },
        {
            'name': 'MAP4_1024',
            'func': FingerprintCalculator.compute_map4,
            'kwargs': {'dimensions': 1024, 'radius': 2}
        }
    ]

    # Проверяем доступность Avalon fingerprints
    try:
        from rdkit.Avalon import pyAvalonTools
        fingerprint_configs.append({
            'name': 'Avalon_512',
            'func': FingerprintCalculator.compute_avalon,
            'kwargs': {'n_bits': 512}
        })
    except ImportError:
        print("Avalon fingerprints not available, skipping...")

    # Проверяем доступность Layered fingerprints
    fingerprint_configs.append({
        'name': 'Layered_512',
        'func': FingerprintCalculator.compute_layered,
        'kwargs': {'n_bits': 512}
    })

    results = []

    for config in fingerprint_configs:
        try:
            result = evaluate_fingerprint(
                config['func'],
                config['name'],
                train.map,
                train.gen_map,
                **config['kwargs']
            )
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"Error evaluating {config['name']}: {e}")
            continue

    # Выводим сводную таблицу результатов
    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'Fingerprint':<20} {'Accuracy':<10} {'AUC':<10} {'Best AUC':<10} {'Time (s)':<12} {'Input Dim':<12} {'Samples':<10}")
    print(f"{'-' * 80}")

    # Сортируем результаты по AUC
    results.sort(key=lambda x: x['auc'], reverse=True)

    for result in results:
        print(f"{result['fingerprint']:<20} {result['accuracy']:<10.4f} {result['auc']:<10.4f} "
              f"{result['best_auc']:<10.4f} {result['training_time']:<12.1f} "
              f"{result['input_dim']:<12} {result['dataset_size']:<10}")

    # Находим лучший фингерпринт по AUC
    if results:
        best_result = results[0]  # Уже отсортированы
        print(f"\nBEST FINGERPRINT: {best_result['fingerprint']}")
        print(f"   AUC: {best_result['auc']:.4f}, Accuracy: {best_result['accuracy']:.4f}")


if __name__ == '__main__':
    compare_fingerprints()