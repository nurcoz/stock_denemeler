"""
═══════════════════════════════════════════════════════════════════════════
COMPLEXITY (2021) MAKALE EXACT REPLIKASYON - PYTHON
═══════════════════════════════════════════════════════════════════════════

Makale: Ali, M., et al. (2021). Predicting the Direction Movement of 
Financial Time Series Using Artificial Neural Network and Support 
Vector Machine. Complexity, 2021.

EXACT METODOLOJI:
- 4-Fold Cross-Validation (Figure 2'de belirtilmiş)
- Kronolojik Train/Test Split
- Min-Max Normalization
- Grid Search with RMSE evaluation

KSE-100 Hedef Sonuçlar:
- Linear SVM: 85.19%
- RBF SVM: 76.88%
- Polynomial SVM: 84.38%
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# 1. VERİ ÇEKME
# ═══════════════════════════════════════════════════════════════════════════

def fetch_data(ticker="^KSE", start="2011-01-01", end="2020-09-27"):
    """KSE-100 verisini çek"""
    print(f"📥 Çekiliyor: {ticker}...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    print(f"✅ {len(data)} günlük veri çekildi\n")
    return data

# ═══════════════════════════════════════════════════════════════════════════
# 2. TEKNİK GÖSTERGELER (15 ADET)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_technical_indicators(data):
    """15 teknik gösterge hesapla"""
    df = data.copy()
    
    print("🔧 Teknik göstergeler hesaplanıyor...")
    
    # 1-2. Stochastic Oscillator
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stochastic_D'] = df['Stochastic_K'].rolling(3).mean()
    
    # 3. ROC
    df['ROC'] = ((df['Close'] / df['Close'].shift(10)) - 1) * 100
    
    # 4. Williams %R
    df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
    
    # 5. Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    
    # 6-7. Disparity Index
    ma5 = df['Close'].rolling(5).mean()
    ma14 = df['Close'].rolling(14).mean()
    df['Disparity_5'] = ((df['Close'] - ma5) / ma5) * 100
    df['Disparity_14'] = ((df['Close'] - ma14) / ma14) * 100
    
    # 8. OSCP
    ma10 = df['Close'].rolling(10).mean()
    df['OSCP'] = ((ma5 - ma10) / ma5) * 100
    
    # 9. CCI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma_tp = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['CCI'] = (tp - ma_tp) / (0.015 * md)
    
    # 10. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 11-15. Pivot Points
    df['Pivot_Point'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['S1'] = (df['Pivot_Point'] * 2) - df['High'].shift(1)
    df['S2'] = df['Pivot_Point'] - (df['High'].shift(1) - df['Low'].shift(1))
    df['R1'] = (df['Pivot_Point'] * 2) - df['Low'].shift(1)
    df['R2'] = df['Pivot_Point'] + (df['High'].shift(1) - df['Low'].shift(1))
    
    # Target: Yarın yükselecek mi?
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # NaN temizle
    df = df.dropna()
    
    print(f"✅ {len(df)} satır veri hazırlandı\n")
    return df

# ═══════════════════════════════════════════════════════════════════════════
# 3. VERİ HAZIRLAMA - KRONOLOJİK SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def prepare_data_chronological(df, test_ratio=0.2):
    """
    KRONOLOJIK SPLIT (Time-Series için doğru yöntem)
    İlk %80 → Train
    Son %20 → Test
    """
    
    feature_cols = ['Stochastic_K', 'Stochastic_D', 'ROC', 'Williams_R',
                    'Momentum', 'Disparity_5', 'Disparity_14', 'OSCP',
                    'CCI', 'RSI', 'Pivot_Point', 'S1', 'S2', 'R1', 'R2']
    
    X = df[feature_cols].values
    y = df['Target'].values
    
    # KRONOLOJIK SPLIT
    train_size = int(len(X) * (1 - test_ratio))
    
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    # Normalizasyon
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("📊 VERİ HAZIRLANDI")
    print("═" * 60)
    print(f"Train: {len(X_train)} samples ({(1-test_ratio)*100:.0f}%)")
    print(f"  Up: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
    print(f"Test: {len(X_test)} samples ({test_ratio*100:.0f}%)")
    print(f"  Up: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

# ═══════════════════════════════════════════════════════════════════════════
# 4. SVM MODELLER - 4-FOLD CV ile EXACT parametreler
# ═══════════════════════════════════════════════════════════════════════════

def train_svm_exact_params(X_train, y_train, X_test, y_test):
    """
    Makaledeki EXACT parametrelerle SVM eğit
    KSE-100 için:
    - Linear: C = 964.7736
    - RBF: C = 137.20, sigma = 0.0909 → gamma = 60.51
    - Polynomial: C = 314.52, coef0 = 0.5554, degree = 2
    """
    
    results = {}
    
    # ═══════════════════════════════════════════════════════════════════
    # LINEAR SVM
    # ═══════════════════════════════════════════════════════════════════
    print("🔧 [1/3] Linear SVM")
    print("─" * 60)
    
    svm_linear = SVC(kernel='linear', C=964.7736)
    svm_linear.fit(X_train, y_train)
    y_pred = svm_linear.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results['Linear'] = {'Accuracy': acc, 'F1': f1}
    
    print(f"📊 Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"📊 F1-Score: {f1:.4f}")
    print(f"🎯 Makale Hedefi: 85.19%")
    print(f"📈 Fark: {(acc - 0.8519)*100:+.2f}% puan\n")
    
    # ═══════════════════════════════════════════════════════════════════
    # RBF SVM
    # ═══════════════════════════════════════════════════════════════════
    print("🔧 [2/3] RBF SVM")
    print("─" * 60)
    
    # sigma = 0.0909 → gamma = 1/(2*sigma^2) = 60.51
    gamma_val = 1 / (2 * 0.0909**2)
    print(f"Sigma = 0.0909 → Gamma = {gamma_val:.4f}")
    
    svm_rbf = SVC(kernel='rbf', C=137.20, gamma=gamma_val)
    svm_rbf.fit(X_train, y_train)
    y_pred = svm_rbf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results['RBF'] = {'Accuracy': acc, 'F1': f1}
    
    print(f"📊 Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"📊 F1-Score: {f1:.4f}")
    print(f"🎯 Makale Hedefi: 76.88%")
    print(f"📈 Fark: {(acc - 0.7688)*100:+.2f}% puan\n")
    
    # ═══════════════════════════════════════════════════════════════════
    # POLYNOMIAL SVM
    # ═══════════════════════════════════════════════════════════════════
    print("🔧 [3/3] Polynomial SVM")
    print("─" * 60)
    
    svm_poly = SVC(kernel='poly', C=314.52, degree=2, coef0=0.5554)
    svm_poly.fit(X_train, y_train)
    y_pred = svm_poly.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results['Polynomial'] = {'Accuracy': acc, 'F1': f1}
    
    print(f"📊 Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"📊 F1-Score: {f1:.4f}")
    print(f"🎯 Makale Hedefi: 84.38%")
    print(f"📈 Fark: {(acc - 0.8438)*100:+.2f}% puan\n")
    
    return results

# ═══════════════════════════════════════════════════════════════════════════
# 5. GRID SEARCH - 4-FOLD CV
# ═══════════════════════════════════════════════════════════════════════════

def grid_search_4fold(X_train, y_train, X_test, y_test):
    """
    Figure 2'deki gibi 4-Fold CV ile Grid Search
    """
    
    print("\n")
    print("═" * 60)
    print("  GRID SEARCH - 4-FOLD CV (Makale Yöntemi)")
    print("═" * 60)
    print()
    
    # 4-Fold CV
    cv = KFold(n_splits=4, shuffle=False)  # Kronolojik fold'lar
    
    results_grid = {}
    
    # ═══════════════════════════════════════════════════════════════════
    # LINEAR
    # ═══════════════════════════════════════════════════════════════════
    print("🔍 [1/3] Linear SVM Grid Search...")
    param_grid_linear = {'C': [1, 10, 100, 500, 964.7736, 1000, 5000]}
    
    grid_linear = GridSearchCV(
        SVC(kernel='linear'),
        param_grid_linear,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_linear.fit(X_train, y_train)
    
    y_pred = grid_linear.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"   Best Params: {grid_linear.best_params_}")
    print(f"   Best CV Score: {grid_linear.best_score_:.4f}")
    print(f"   Test Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    
    results_grid['Linear'] = {'Accuracy': acc, 'Best_Params': grid_linear.best_params_}
    
    # ═══════════════════════════════════════════════════════════════════
    # RBF
    # ═══════════════════════════════════════════════════════════════════
    print("🔍 [2/3] RBF SVM Grid Search...")
    param_grid_rbf = {
        'C': [1, 10, 100, 137.20, 500],
        'gamma': [0.001, 0.01, 0.1, 1, 60.51, 100]
    }
    
    grid_rbf = GridSearchCV(
        SVC(kernel='rbf'),
        param_grid_rbf,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_rbf.fit(X_train, y_train)
    
    y_pred = grid_rbf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"   Best Params: {grid_rbf.best_params_}")
    print(f"   Best CV Score: {grid_rbf.best_score_:.4f}")
    print(f"   Test Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    
    results_grid['RBF'] = {'Accuracy': acc, 'Best_Params': grid_rbf.best_params_}
    
    # ═══════════════════════════════════════════════════════════════════
    # POLYNOMIAL
    # ═══════════════════════════════════════════════════════════════════
    print("🔍 [3/3] Polynomial SVM Grid Search...")
    param_grid_poly = {
        'C': [10, 100, 314.52, 500],
        'degree': [2, 3],
        'coef0': [0.1, 0.5554, 1.0]
    }
    
    grid_poly = GridSearchCV(
        SVC(kernel='poly'),
        param_grid_poly,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_poly.fit(X_train, y_train)
    
    y_pred = grid_poly.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"   Best Params: {grid_poly.best_params_}")
    print(f"   Best CV Score: {grid_poly.best_score_:.4f}")
    print(f"   Test Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    
    results_grid['Polynomial'] = {'Accuracy': acc, 'Best_Params': grid_poly.best_params_}
    
    return results_grid

# ═══════════════════════════════════════════════════════════════════════════
# 6. ANA FONKSİYON
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Ana çalıştırma fonksiyonu"""
    
    print("\n")
    print("═" * 70)
    print("  COMPLEXITY (2021) - EXACT REPLICATION")
    print("  KSE-100 Stock Index Prediction")
    print("═" * 70)
    print()
    
    # 1. Veri çek
    data = fetch_data("^KSE", "2011-01-01", "2020-09-27")
    
    # 2. Teknik göstergeler
    df = calculate_technical_indicators(data)
    
    # 3. Veri hazırla (KRONOLOJIK)
    X_train, X_test, y_train, y_test, features = prepare_data_chronological(df)
    
    # 4. EXACT parametrelerle test
    print("═" * 70)
    print("  EXACT MAKALE PARAMETRELERİYLE TEST")
    print("═" * 70)
    print()
    
    results_exact = train_svm_exact_params(X_train, y_train, X_test, y_test)
    
    # 5. Grid Search ile en iyi parametreleri bul
    results_grid = grid_search_4fold(X_train, y_train, X_test, y_test)
    
    # 6. Sonuç özeti
    print("\n")
    print("═" * 70)
    print("  SONUÇ ÖZETİ")
    print("═" * 70)
    print()
    
    print("📚 MAKALE HEDEFLERİ vs GERÇEKLEŞEN:")
    print("─" * 70)
    
    article_targets = {'Linear': 0.8519, 'RBF': 0.7688, 'Polynomial': 0.8438}
    
    for model in ['Linear', 'RBF', 'Polynomial']:
        exact_acc = results_exact[model]['Accuracy']
        grid_acc = results_grid[model]['Accuracy']
        target = article_targets[model]
        
        print(f"\n{model} SVM:")
        print(f"  Makale:          {target*100:.2f}%")
        print(f"  Exact Params:    {exact_acc*100:.2f}% ({(exact_acc-target)*100:+.2f}%)")
        print(f"  Grid Search:     {grid_acc*100:.2f}% ({(grid_acc-target)*100:+.2f}%)")
        print(f"  Best Params:     {results_grid[model]['Best_Params']}")
    
    # Ortalama karşılaştırma
    avg_article = np.mean(list(article_targets.values()))
    avg_exact = np.mean([results_exact[m]['Accuracy'] for m in article_targets.keys()])
    avg_grid = np.mean([results_grid[m]['Accuracy'] for m in article_targets.keys()])
    
    print("\n" + "─" * 70)
    print(f"\nORTALAMA:")
    print(f"  Makale:       {avg_article*100:.2f}%")
    print(f"  Exact Params: {avg_exact*100:.2f}% ({(avg_exact-avg_article)*100:+.2f}%)")
    print(f"  Grid Search:  {avg_grid*100:.2f}% ({(avg_grid-avg_article)*100:+.2f}%)")
    
    if avg_exact >= avg_article * 0.90:
        print("\n✅ BAŞARILI: Makaleye çok yakın performans!")
    elif avg_exact >= avg_article * 0.80:
        print("\n⚠️  Kabul Edilebilir: İyileştirme mümkün")
    else:
        print("\n❌ Düşük Performans: Veri kaynağı farklılığı olabilir")
    
    print("\n" + "═" * 70)
    print()

if __name__ == "__main__":
    main()
