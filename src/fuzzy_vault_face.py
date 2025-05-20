import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
from sympy import Poly, symbols
import hashlib
import pickle
from insightface.app import FaceAnalysis
import signal
import configparser

def load_config(config_file="config.ini"):
    """Učitava konfiguraciju iz config.ini datoteke."""
    config = configparser.ConfigParser()
    
    # Postavi zadane vrijednosti
    config['FeatureTransformer'] = {
        'n_intervals': '2',
        'binarization_method': 'LSSC',
        'use_equal_probable': 'True'
    }
    config['FuzzyVault'] = {
        'field_size': '65537',
        'polynomial_degree': '50',
        'n_chaff': '5120',
        'max_unlock_attempts': '100'
    }
    config['FaceExtractor'] = {
        'det_size': '320',
        'num_features': '128'
    }
    
    # Pokušaj učitati konfiguraciju iz datoteke
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:  # Dodaj encoding
            config.read_file(f)
        print(f"Konfiguracija učitana iz {config_file}")
    else:
        # Ako datoteka ne postoji, kreiraj je sa zadanim vrijednostima
        with open(config_file, 'w', encoding='utf-8') as f:  # Dodaj encoding
            config.write(f)
        print(f"Kreirana nova config.ini datoteka sa zadanim vrijednostima")
    
    return config


def get_directory_hash(directory):
    """Generira hash za sve datoteke u direktoriju (ime + vrijeme izmjene)"""
    hash_str = ""
    for root, dirs, files in os.walk(directory):
        for file in sorted(files):
            path = os.path.join(root, file)
            if file.lower().endswith(('.jpg', '.png')):
                hash_str += f"{file}:{os.path.getmtime(path)}|"
    return hashlib.sha256(hash_str.encode()).hexdigest()

class FaceFeatureExtractor:
    def __init__(self, config=None):
        if config is None:
            config = load_config()
        
        det_size = int(config['FaceExtractor']['det_size'])
        self.num_features = int(config['FaceExtractor']['num_features'])
        
        self.model = FaceAnalysis(name="buffalo_l", root="./weights")
        self.model.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def extract_features(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Ne mogu učitati sliku: {image_path}")
        
        img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0,0,0))
        faces = self.model.get(img)
        
        if not faces:
            raise ValueError("Nije detektirano lice na slici.")
        embedding = faces[0].embedding
        return embedding[:self.num_features]

class FeatureTransformer:
    def __init__(self, config=None):
        if config is None:
            config = load_config()
            
        self.n_intervals = int(config['FeatureTransformer']['n_intervals'])
        self.binarization_method = config['FeatureTransformer']['binarization_method']
        self.use_equal_probable = config['FeatureTransformer'].getboolean('use_equal_probable')
        
        self.intervals = None
        self.feature_min = None
        self.feature_max = None

    def fit(self, feature_vectors):
        if len(feature_vectors) == 0:
            raise ValueError("Nema podataka za treniranje")
        
        n_features = feature_vectors.shape[1]
        self.intervals = []
        self.feature_min = np.min(feature_vectors, axis=0)
        self.feature_max = np.max(feature_vectors, axis=0)

        for i in range(n_features):
            feature_values = feature_vectors[:, i]
            if self.use_equal_probable:
                quantiles = np.linspace(0, 1, self.n_intervals + 1)
                intervals = np.quantile(feature_values, quantiles)
            else:
                intervals = np.linspace(self.feature_min[i], self.feature_max[i], self.n_intervals + 1)
            self.intervals.append(intervals)

    def quantize(self, feature_vector):
        if self.intervals is None:
            raise ValueError("Transformer nije treniran. Prvo pozovite fit()")
        
        n_features = len(feature_vector)
        quantized = np.zeros(n_features, dtype=np.int32)
        
        for i in range(n_features):
            for j in range(self.n_intervals):
                if self.intervals[i][j] <= feature_vector[i] < self.intervals[i][j+1]:
                    quantized[i] = j
                    break
            else:
                quantized[i] = self.n_intervals - 1
        return quantized

    def binarize(self, quantized_vector):
        n_features = len(quantized_vector)
        
        if self.binarization_method == 'LSSC':
            m = self.n_intervals - 1
            binary_vector = np.zeros(n_features * m, dtype=np.int32)
            for i in range(n_features):
                for j in range(m):
                    if quantized_vector[i] > j:
                        binary_vector[i*m + j] = 1
            return binary_vector
        else:
            raise ValueError(f"Nepodržana metoda binarizacije: {self.binarization_method}")

    def transform(self, feature_vector):
        quantized = self.quantize(feature_vector)
        binary = self.binarize(quantized)
        return np.where(binary == 1)[0].tolist()

class FuzzyVault:
    def __init__(self, config=None):
        if config is None:
            config = load_config()
        
        self.field_size = int(config['FuzzyVault']['field_size'])
        self.polynomial_degree = int(config['FuzzyVault']['polynomial_degree'])
        self.n_chaff = int(config['FuzzyVault']['n_chaff'])
        self.max_unlock_attempts = int(config['FuzzyVault']['max_unlock_attempts'])
        
        self.X = symbols('X')
        self.interrupted = False

    def _handle_interrupt(self, signum, frame):
        self.interrupted = True

    def _create_random_polynomial(self):
        coeffs = [random.randint(0, self.field_size-1) for _ in range(self.polynomial_degree)]
        while coeffs[0] == 0:
            coeffs[0] = random.randint(1, self.field_size-1)
        return Poly(coeffs, self.X)

    def _hash_polynomial(self, poly):
        return hashlib.sha256(str(poly.all_coeffs()).encode()).hexdigest()

    def _generate_bijection(self, seed):
        random.seed(seed)
        perm = list(range(self.field_size))
        random.shuffle(perm)
        return lambda x: perm[x % self.field_size]

    def lock(self, feature_set, key=None):
        poly = self._create_random_polynomial() if key is None else Poly(key, self.X)
        hash_value = self._hash_polynomial(poly)
        bijection = self._generate_bijection(hash_value)
        
        transformed_set = [bijection(x) for x in feature_set]
        genuine_points = [(x, poly.eval(x) % self.field_size) for x in transformed_set]
        
        chaff_points = []
        existing_x = set(transformed_set)
        while len(chaff_points) < self.n_chaff:
            x = random.randint(0, self.field_size-1)
            if x not in existing_x:
                y = random.randint(0, self.field_size-1)
                while y == poly.eval(x) % self.field_size:
                    y = random.randint(0, self.field_size-1)
                chaff_points.append((x, y))
                existing_x.add(x)
        
        vault_points = genuine_points + chaff_points
        random.shuffle(vault_points)
        
        print(f"\nGeneriran vault za korisnika:")
        print(f"- Broj genuine točaka: {len(genuine_points)}")
        print(f"- Broj chaff točaka: {len(chaff_points)}")
        print(f"- Hash vrijednost: {hash_value[:16]}...{hash_value[-16:]}")
        print(f"- Stupanj polinoma: {self.polynomial_degree}")
        
        return (vault_points, hash_value, poly.all_coeffs())

    def unlock(self, vault_points, feature_set, hash_value, max_attempts=None):
        max_attempts = max_attempts or self.max_unlock_attempts
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        bijection = self._generate_bijection(hash_value)
        transformed_set = [bijection(x) for x in feature_set]
        
        candidate_points = []
        for x in transformed_set:
            for (vx, vy) in vault_points:
                if vx == x:
                    candidate_points.append((vx, vy))
        
        for attempt in range(max_attempts):
            if self.interrupted:
                print("\nPrekinuto korisnikom!")
                return None
            
            print(f"\rPokušaj {attempt+1}/{max_attempts}...", end="")
            
            if len(candidate_points) < self.polynomial_degree:
                continue
                
            try:
                points = random.sample(candidate_points, self.polynomial_degree)
                interp_poly = Poly(0, self.X)
                
                for i, (xi, yi) in enumerate(points):
                    term = Poly(yi, self.X)
                    for j, (xj, _) in enumerate(points):
                        if i != j:
                            denom = (xi - xj) % self.field_size
                            denom_inv = pow(denom, self.field_size-2, self.field_size)
                            term *= (self.X - xj) * denom_inv
                    interp_poly += term
                    interp_poly = Poly([c % self.field_size for c in interp_poly.all_coeffs()], self.X)
                
                interp_coeffs = [c % self.field_size for c in interp_poly.all_coeffs()]
                if self._hash_polynomial(Poly(interp_coeffs, self.X)) == hash_value:
                    print("\n\nUspješno otključan vault!")
                    print(f"- Pronađeni ključ: {interp_coeffs[:3]}...{interp_coeffs[-3:]}")
                    return interp_coeffs
            except:
                continue
        
        print("\nNeuspješno otključavanje vaulta!")
        return None

class BiometricTemplateProtectionSystem:
    def __init__(self, config=None):
        if config is None:
            config = load_config()
        
        self.config = config
        self.feature_extractor = FaceFeatureExtractor(config)
        self.feature_transformer = FeatureTransformer(config)
        self.fuzzy_vault = FuzzyVault(config)
        self.is_trained = False
        self.training_hash = None

    def train(self, image_paths, force_retrain=False):
        current_hash = get_directory_hash('./data/training')
        
        if not force_retrain and self._is_training_unchanged(current_hash):
            print("\nNema promjena u training podacima. Učitavam postojeći model.")
            return
        
        features = []
        for path in image_paths:
            try:
                features.append(self.feature_extractor.extract_features(path))
            except Exception as e:
                print(f"Greška pri obradi {path}: {str(e)}")
        
        if not features:
            raise ValueError("Nema valjanih podataka za treniranje")
        
        self.feature_transformer.fit(np.array(features))
        self._save_training_state(current_hash, features)
        self.is_trained = True
        print(f"\nSustav treniran na {len(features)} slika")

    def _is_training_unchanged(self, current_hash):
        try:
            with open("training_state.pkl", "rb") as f:
                saved_hash, _ = pickle.load(f)
                return saved_hash == current_hash
        except FileNotFoundError:
            return False

    def _save_training_state(self, current_hash, features):
        with open("training_state.pkl", "wb") as f:
            pickle.dump((current_hash, self.feature_transformer), f)
        self.training_hash = current_hash

    def enroll(self, image_path):
        if not self.is_trained:
            raise ValueError("Sustav nije treniran")
        
        print(f"\nPokrećem registraciju za: {os.path.basename(image_path)}")
        features = self.feature_extractor.extract_features(image_path)
        feature_set = self.feature_transformer.transform(features)
        return self.fuzzy_vault.lock(feature_set)

    def verify(self, image_path, vault_data, hash_value):
        print(f"\nPokrećem verifikaciju za: {os.path.basename(image_path)}")
        features = self.feature_extractor.extract_features(image_path)
        feature_set = self.feature_transformer.transform(features)
        return self.fuzzy_vault.unlock(vault_data, feature_set, hash_value)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump((self.feature_transformer, self.fuzzy_vault), f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.feature_transformer, self.fuzzy_vault = pickle.load(f)
        self.is_trained = True

def main():
    # Učitaj konfiguraciju
    config = load_config()
    
    training_dir = './data/training'
    gallery_dir = './data/gallery'
    probe_dir = './data/probe'

    try:
        # Pokušaj učitati postojeći model
        with open("training_state.pkl", "rb") as f:
            saved_hash, feature_transformer = pickle.load(f)
            system = BiometricTemplateProtectionSystem(config)
            system.feature_transformer = feature_transformer
            system.training_hash = saved_hash
            system.is_trained = True
            print("\nUčitavanje postojećeg modela...")
    except FileNotFoundError:
        # Kreiraj novi sustav ako nema spremljenog modela
        system = BiometricTemplateProtectionSystem(config)

    # Treniraj sustav (samo ako ima promjena)
    training_images = [os.path.join(training_dir, f) for f in os.listdir(training_dir) 
                      if f.lower().endswith(('.jpg', '.png'))]
    system.train(training_images)

    # Registracija korisnika
    enrollments = {}
    gallery_images = [os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir) 
                     if f.lower().endswith(('.jpg', '.png'))]
    
    for img_path in gallery_images:
        user_id = os.path.splitext(os.path.basename(img_path))[0]
        try:
            vault_data, hash_val, _ = system.enroll(img_path)
            enrollments[user_id] = (vault_data, hash_val)
            print(f"Registriran korisnik: {user_id}")
        except Exception as e:
            print(f"Greška pri registraciji {user_id}: {str(e)}")

    # Evaluacija
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    probe_images = [os.path.join(probe_dir, f) for f in os.listdir(probe_dir) 
                   if f.lower().endswith(('.jpg', '.png'))]
    
    for probe_path in probe_images:
        user_id = os.path.splitext(os.path.basename(probe_path))[0].split('_')[0]
        try:
            for enrolled_id, (vault_data, hash_val) in enrollments.items():
                key = system.verify(probe_path, vault_data, hash_val)
                success = key is not None
                
                if success:
                    if user_id == enrolled_id:
                        print(f"✓ Točna potvrda: {user_id}")
                        true_positives += 1
                    else:
                        print(f"✗ Lažna potvrda: {user_id} kao {enrolled_id}")
                        false_positives += 1
                else:
                    if user_id == enrolled_id:
                        print(f"✗ Netočno odbijanje: {user_id}")
                        false_negatives += 1
                    else:
                        true_negatives += 1
        except Exception as e:
            print(f"Greška pri verifikaciji {probe_path}: {str(e)}")

    # Evaluacijske metrike
    total_tests = true_positives + false_positives + true_negatives + false_negatives
    if total_tests > 0:
        print("\nRezultati evaluacije:")
        print(f"Točne potvrde (TP): {true_positives}")
        print(f"Lažne potvrde (FP): {false_positives}")
        print(f"Točna odbijanja (TN): {true_negatives}")
        print(f"Netočna odbijanja (FN): {false_negatives}")
        
        if (true_positives + false_negatives) > 0:
            gmr = true_positives / (true_positives + false_negatives)
            print(f"\nGenuine Match Rate (GMR): {gmr:.2%}")
        
        if (false_positives + true_negatives) > 0:
            fmr = false_positives / (false_positives + true_negatives)
            print(f"False Match Rate (FMR): {fmr:.2%}")

if __name__ == "__main__":
    main()
