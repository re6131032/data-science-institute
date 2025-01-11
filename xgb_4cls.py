import pandas as pd
import numpy as np
import torch
import pickle
import json
import os
from datetime import datetime
from sklearn.metrics import classification_report
import xgboost as xgb

def load_data():
    """載入數據和索引"""
    print("\nLoading data...")
    print("Reading features data...")
    features_data = pd.read_csv('data/processed/features_data.csv')
    print(f"Features data shape: {features_data.shape}")
    
    print("Loading split indices...")
    split_indices = torch.load('data/processed/split_indices.pt')
    train_indices = split_indices['train_indices']
    test_indices = split_indices['test_indices']

    print("Loading label encoder...")
    with open('base_models/encoder/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return features_data, train_indices, test_indices, label_encoder

def prepare_targeted_data(features_data, train_indices, test_indices, label_encoder):
    """準備針對性訓練數據，只使用特定類別"""
    print("\nPreparing targeted training data...")
    target_categories = ['socialanxiety', 'mentalhealth', 'COVID19_support', 'lonely']
    
    # 建立目標類別到新索引的映射
    target_to_index = {cat: idx for idx, cat in enumerate(target_categories)}
    
    # 取得訓練集的資料
    train_data = features_data.iloc[train_indices]
    
    # 只選擇目標類別的資料
    train_mask = train_data['subreddit'].isin(target_categories)
    train_data = train_data[train_mask]
    
    # 準備特徵和標籤
    X_train = train_data.drop('subreddit', axis=1)
    # 使用新的映射重新編碼標籤
    y_train = train_data['subreddit'].map(target_to_index)
    
    # 測試集保持完整
    test_data = features_data.iloc[test_indices]
    X_test = test_data.drop('subreddit', axis=1)
    y_test = label_encoder.transform(test_data['subreddit'])
    
    # 保存映射關係，供之後使用
    global TARGET_MAPPING
    TARGET_MAPPING = {
        'original_to_new': {
            label_encoder.transform([cat])[0]: target_to_index[cat] 
            for cat in target_categories
        },
        'new_to_original': {
            target_to_index[cat]: label_encoder.transform([cat])[0]
            for cat in target_categories
        }
    }
    
    print('\nTarget categories training data:')
    values, counts = np.unique(train_data['subreddit'], return_counts=True)
    for v, c in zip(values, counts):
        print(f'{v}: {c} (new index: {target_to_index[v]})')
    
    print(f"\nFinal training data shape: {X_train.shape}")
    print(f"Final test data shape: {X_test.shape}")
    print(f"Number of unique classes in training: {len(np.unique(y_train))}")
    print("Label mapping:", TARGET_MAPPING)
        
    return X_train, X_test, y_train, y_test

def train_targeted_xgboost(X_train, y_train, num_classes):
    """訓練針對性的XGBoost模型"""
    print("\nTraining targeted XGBoost model...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of classes in training data: {len(np.unique(y_train))}")
    print("Initializing XGBoost model...")
    
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,  # 保持26類輸出
        random_state=42,
        tree_method='hist',
        device='cuda',
        max_delta_step=1,
        min_child_weight=1,
        verbosity=2  # 增加輸出詳細度
    )
    
    print("Starting model training...")
    try:
        model.fit(
            X_train, 
            y_train,
            verbose=True
        )
        print("Model training completed successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    
    return model

def evaluate_targeted_model(model, X_test, y_test, label_encoder):
    """評估模型，特別關注目標類別的表現"""
    print("\nEvaluating model...")
    target_categories = ['socialanxiety', 'mentalhealth', 'COVID19_support', 'lonely']
    global TARGET_MAPPING
    
    # 定義轉換函數
    def map_predictions(preds):
        return np.array([TARGET_MAPPING['new_to_original'].get(p, p) for p in preds])
    
    # 預測
    print("Making predictions...")
    y_pred = map_predictions(model.predict(X_test))
    y_pred_proba = model.predict_proba(X_test)
    
    # 計算目標類別的平均信心度
    print("\nCalculating confidence scores for target categories...")
    confidence_scores = {}
    for cat, new_idx in enumerate(range(len(target_categories))):  # 使用新的類別索引 0-3
        # 找出原始標籤索引
        orig_idx = TARGET_MAPPING['new_to_original'][new_idx]
        mask = y_test == orig_idx
        if np.any(mask):
            confidence = np.mean(y_pred_proba[mask, new_idx])  # 使用新的類別索引
            confidence_scores[target_categories[new_idx]] = confidence
            print(f"{target_categories[new_idx]}: {confidence:.4f}")
    
    # 生成分類報告
    print("\nGenerating classification report...")
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    # 計算目標類別的平均性能
    target_performance = {
        'f1': np.mean([report[cat]['f1-score'] for cat in target_categories]),
        'precision': np.mean([report[cat]['precision'] for cat in target_categories]),
        'recall': np.mean([report[cat]['recall'] for cat in target_categories])
    }
    
    return {
        'classification_report': report,
        'target_performance': target_performance,
        'confidence_scores': confidence_scores
    }

def save_targeted_model(model, metrics, save_dir='base_models/targeted_xgboost'):
    """保存模型和相關信息"""
    print(f"\nSaving model to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(save_dir, 'model.json')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # 將 numpy 類型轉換為 Python 原生類型
    def convert_to_native_types(obj):
        if isinstance(obj, dict):
            return {key: convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_native_types(obj.tolist())
        else:
            return obj
    
    # 轉換指標數據
    converted_metrics = convert_to_native_types(metrics)
    
    # 保存配置信息
    config = {
        'target_categories': ['socialanxiety', 'mentalhealth', 'COVID19_support', 'lonely'],
        'metrics': converted_metrics,
        'saved_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_version': '1.0'
    }
    
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_path}")

def main():
    print("=== Starting Targeted XGBoost Training ===")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 載入數據
        features_data, train_indices, test_indices, label_encoder = load_data()
        
        # 準備針對性訓練數據
        X_train, X_test, y_train, y_test = prepare_targeted_data(
            features_data, train_indices, test_indices, label_encoder
        )
        
        # 訓練模型
        model = train_targeted_xgboost(
            X_train, y_train, num_classes=len(label_encoder.classes_)
        )
        
        # 評估模型
        metrics = evaluate_targeted_model(
            model, X_test, y_test, label_encoder
        )
        
        # 保存模型和結果
        save_targeted_model(model, metrics)
        
        # 轉換數據類型用於輸出
        def convert_to_native_types(obj):
            if isinstance(obj, dict):
                return {key: convert_to_native_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_to_native_types(obj.tolist())
            else:
                return obj
        
        print("\n=== Training Summary ===")
        print("\nTarget categories performance:")
        print(json.dumps(convert_to_native_types(metrics['target_performance']), indent=2))
        print("\nConfidence scores for target categories:")
        print(json.dumps(convert_to_native_types(metrics['confidence_scores']), indent=2))
        
        print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()